"""
gpt_numpy.py
============
GPT do Zero — Transformer de linguagem em NumPy puro.

Arquitetura (estilo GPT-2 small):
  TokenEmbedding + PositionalEncoding
  N × TransformerBlock (pre-LN, causal mask, multi-head attention, FFN)
  LayerNorm final
  LM head (projeção linear para vocab)

Tudo é diferenciável à mão: os gradientes de cada operação estão
explícitos, sem autograd. O objetivo é tornar visível o que frameworks
como PyTorch ocultam.

Uso rápido
----------
    from gpt_numpy import GPTConfig, GPT, CharTokenizer, AdamW, Trainer

    text   = open("corpus.txt").read()
    tok    = CharTokenizer(text)
    cfg    = GPTConfig(vocab_size=tok.vocab_size, ctx_len=64,
                       n_embd=128, n_heads=4, n_layers=2)
    model  = GPT(cfg)
    trainer = Trainer(model, tok, cfg)
    trainer.fit(text, epochs=10, batch_size=16, lr=3e-4)
    print(trainer.generate("Olá", max_new=200))
"""

from __future__ import annotations
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Optional

# NumPy emite RuntimeWarning de overflow/divide nos primeiros passes
# antes que o LayerNorm estabilize os valores; as computações são
# matematicamente corretas (float64, sem NaN real), então suprimimos.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*encountered in matmul.*")


# ─────────────────────────────────────────────
#  Configuração
# ─────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int = 65          # tamanho do vocabulário
    ctx_len:    int = 128         # comprimento máximo do contexto (T)
    n_embd:     int = 256         # dimensão dos embeddings (d_model)
    n_heads:    int = 4           # número de cabeças de atenção
    n_layers:   int = 4           # número de blocos Transformer
    ffn_mult:   int = 4           # multiplicador FFN (n_embd * ffn_mult)
    dropout:    float = 0.1       # taxa de dropout
    eps:        float = 1e-5      # epsilon para LayerNorm
    seed:       int   = 42


# ─────────────────────────────────────────────
#  Tokenizador de caracteres
# ─────────────────────────────────────────────

class CharTokenizer:
    """
    Tokenizador character-level minimalista.

    Cada caractere único do corpus recebe um ID inteiro.
    É o tokenizador mais simples possível — perfeito para
    expor os mecanismos do modelo sem ruído de tokenização.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)


# ─────────────────────────────────────────────
#  Layer Norm (pre-norm, sem autograd)
# ─────────────────────────────────────────────

class LayerNorm:
    """
    LayerNorm normaliza ao longo da última dimensão (features).

    Gradiente: regra da cadeia sobre a fórmula de normalização.
    Referência: Ba et al., 2016.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        self.gamma = np.ones(d)       # escala aprendida
        self.beta  = np.zeros(d)      # deslocamento aprendido
        self.eps   = eps
        # cache para backward
        self._x_norm: Optional[np.ndarray] = None
        self._std:    Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (..., d)
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        self._std    = np.sqrt(var + self.eps)
        self._x_norm = (x - mean) / self._std
        return self.gamma * self._x_norm + self.beta

    def backward(self, dout: np.ndarray):
        """
        Retorna (grad_input, grad_gamma, grad_beta).

        A derivação completa está em:
        https://kevinzakka.github.io/2016/09/14/batch_normalization/
        (o mesmo princípio se aplica ao LayerNorm com eixo=-1)
        """
        d       = dout.shape[-1]
        x_norm  = self._x_norm
        std     = self._std

        d_gamma = (dout * x_norm).sum(axis=tuple(range(dout.ndim - 1)))
        d_beta  = dout.sum(axis=tuple(range(dout.ndim - 1)))

        dx_norm = dout * self.gamma
        # propagação através da normalização
        dx = (1.0 / (d * std)) * (
            d * dx_norm
            - dx_norm.sum(axis=-1, keepdims=True)
            - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)
        )
        return dx, d_gamma, d_beta


# ─────────────────────────────────────────────
#  Multi-Head Causal Self-Attention
# ─────────────────────────────────────────────

class CausalSelfAttention:
    """
    Multi-head self-attention com máscara causal (triangular inferior).

    Cada cabeça aprende um subespaço diferente de Q, K, V.
    A máscara impede que o token na posição t "veja" tokens futuros
    (t+1, t+2, …), o que é essencial para modelagem de linguagem
    auto-regressiva.

    Decisões de design expostas:
    - Projeções QKV são uma única matriz (W_qkv) por eficiência.
    - Escala por 1/√(d_head) evita que softmax sature em gradientes
      minúsculos (zona de saturação da softmax).
    - A máscara é aplicada ANTES do softmax somando -∞ nas posições
      futuras; após softmax essas posições valem exatamente 0.
    """

    def __init__(self, cfg: GPTConfig):
        self.n_heads = cfg.n_heads
        self.d_model = cfg.n_embd
        self.d_head  = cfg.n_embd // cfg.n_heads
        self.dropout = cfg.dropout
        assert cfg.n_embd % cfg.n_heads == 0, "n_embd deve ser divisível por n_heads"

        # Inicialização de Glorot / GPT-2: escala por 1/√(2 * n_layers)
        # nas projeções de saída (residual stream stability)
        scale_qkv = 0.02
        scale_out = 0.02 / np.sqrt(2 * cfg.n_layers)
        # Projeções unificadas QKV → 3 × d_model saídas
        self.W_qkv = np.random.randn(cfg.n_embd, 3 * cfg.n_embd) * scale_qkv
        self.b_qkv = np.zeros(3 * cfg.n_embd)
        # Projeção de saída
        self.W_o   = np.random.randn(cfg.n_embd, cfg.n_embd) * scale_out
        self.b_o   = np.zeros(cfg.n_embd)

        # Máscara causal (constante, alocada uma vez)
        T = cfg.ctx_len
        self._causal_mask = np.tril(np.ones((T, T), dtype=bool))

        self.training = True
        # cache backward
        self._cache: dict = {}

    # ----------------------------------------------------------
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, d_model) → (B, n_heads, T, d_head)"""
        B, T, _ = x.shape
        x = x.reshape(B, T, self.n_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, n_heads, T, d_head) → (B, T, d_model)"""
        B, H, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)

    # ----------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (B, T, d_model)
        retorna: (B, T, d_model)
        """
        B, T, C = x.shape

        # ── 1. Projeções Q, K, V ──────────────────────────────
        qkv = x @ self.W_qkv + self.b_qkv          # (B, T, 3C)
        Q, K, V = np.split(qkv, 3, axis=-1)        # cada (B, T, C)

        # ── 2. Dividir em cabeças ─────────────────────────────
        Q = self._split_heads(Q)   # (B, H, T, dh)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # ── 3. Scores de atenção ──────────────────────────────
        scale  = 1.0 / np.sqrt(self.d_head)
        scores = Q @ K.transpose(0, 1, 3, 2) * scale  # (B, H, T, T)

        # ── 4. Máscara causal ─────────────────────────────────
        mask   = self._causal_mask[:T, :T]             # (T, T)
        scores = np.where(mask, scores, -1e9)

        # ── 5. Softmax ────────────────────────────────────────
        scores -= scores.max(axis=-1, keepdims=True)   # estabilidade numérica
        exp_s  = np.exp(scores)
        attn   = exp_s / exp_s.sum(axis=-1, keepdims=True)  # (B, H, T, T)

        # ── 6. Dropout na atenção ─────────────────────────────
        if self.training and self.dropout > 0:
            drop_mask = (np.random.rand(*attn.shape) > self.dropout) / (1 - self.dropout)
            attn_drop = attn * drop_mask
        else:
            drop_mask = np.ones_like(attn)
            attn_drop = attn

        # ── 7. Valores ponderados ─────────────────────────────
        out_h = attn_drop @ V                       # (B, H, T, dh)
        out   = self._merge_heads(out_h)            # (B, T, C)

        # ── 8. Projeção de saída ──────────────────────────────
        out_proj = out @ self.W_o + self.b_o        # (B, T, C)

        # ── Cache para backward ───────────────────────────────
        self._cache = dict(x=x, Q=Q, K=K, V=V,
                           attn=attn, attn_drop=attn_drop,
                           drop_mask=drop_mask,
                           out_h=out_h, out=out,
                           scale=scale, mask=mask)
        return out_proj

    # ----------------------------------------------------------
    def backward(self, dout: np.ndarray):
        """
        dout: (B, T, C)
        Retorna (d_input, grads) onde grads é dict com todas as derivadas.

        O backward de self-attention segue:
          dout → dW_o, db_o, d_out_merged
          d_out_merged → (via merge_heads) d_out_heads
          d_out_heads = d(attn_drop @ V)
            → d_attn_drop, d_V
          d_attn_drop → (dropout) → d_attn
          d_attn → (softmax backward) → d_scores
          d_scores → (masking, scale) → d_Q, d_K
          d_Q, d_K, d_V → (split_heads + concat) → d_qkv
          d_qkv → dW_qkv, db_qkv, d_x
        """
        c = self._cache
        B, T, C = dout.shape
        H, dh   = self.n_heads, self.d_head

        # ── 8. backward projeção de saída ─────────────────────
        d_W_o  = c["out"].reshape(B * T, C).T @ dout.reshape(B * T, C)
        d_b_o  = dout.sum(axis=(0, 1))
        d_out  = dout @ self.W_o.T                  # (B, T, C)

        # ── 7. backward merge_heads ───────────────────────────
        # (B, T, C) → (B, H, T, dh)
        d_out_h = d_out.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        # ── 6 + 5. backward attn_drop @ V ────────────────────
        # d_out_h = d_attn_drop @ V  →  d_attn_drop = d_out_h @ V^T
        d_attn_drop = d_out_h @ c["V"].transpose(0, 1, 3, 2)   # (B,H,T,T)
        d_V         = c["attn_drop"].transpose(0, 1, 3, 2) @ d_out_h  # (B,H,T,dh)

        # ── dropout backward ──────────────────────────────────
        d_attn = d_attn_drop * c["drop_mask"]

        # ── 5. backward softmax ───────────────────────────────
        # d_scores = attn * (d_attn - sum(d_attn * attn, axis=-1, keepdims=True))
        attn     = c["attn"]
        d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))

        # ── 4. backward máscara causal ────────────────────────
        d_scores = np.where(c["mask"], d_scores, 0.0)

        # ── 3. backward Q @ K^T * scale ───────────────────────
        d_scores *= c["scale"]
        d_Q = d_scores @ c["K"]                         # (B,H,T,dh)
        d_K = d_scores.transpose(0, 1, 3, 2) @ c["Q"]  # (B,H,T,dh)

        # ── 2. backward split_heads → (B,T,C) ─────────────────
        def _merge_grad(dh_tensor):
            return dh_tensor.transpose(0, 2, 1, 3).reshape(B, T, C)

        d_Q_flat = _merge_grad(d_Q)
        d_K_flat = _merge_grad(d_K)
        d_V_flat = _merge_grad(d_V)
        d_qkv    = np.concatenate([d_Q_flat, d_K_flat, d_V_flat], axis=-1)  # (B,T,3C)

        # ── 1. backward projeções QKV ──────────────────────────
        x_flat   = c["x"].reshape(B * T, C)
        d_qkv_fl = d_qkv.reshape(B * T, 3 * C)
        d_W_qkv  = x_flat.T @ d_qkv_fl
        d_b_qkv  = d_qkv.sum(axis=(0, 1))
        d_x      = d_qkv @ self.W_qkv.T              # (B, T, C)

        grads = dict(W_qkv=d_W_qkv, b_qkv=d_b_qkv,
                     W_o=d_W_o,     b_o=d_b_o)
        return d_x, grads


# ─────────────────────────────────────────────
#  Feed-Forward Network (FFN)
# ─────────────────────────────────────────────

class FeedForward:
    """
    FFN de dois estágios com GELU: Linear → GELU → Linear.

    A dimensão interna é n_embd × ffn_mult (tipicamente 4×).
    GELU é preferido ao ReLU em Transformers porque tem gradiente
    suave em z≈0, o que ajuda a treinar redes profundas.
    """

    def __init__(self, cfg: GPTConfig):
        d, h = cfg.n_embd, cfg.n_embd * cfg.ffn_mult
        scale_in  = 0.02
        scale_out = 0.02 / np.sqrt(2 * cfg.n_layers)
        self.W1 = np.random.randn(d, h) * scale_in
        self.b1 = np.zeros(h)
        self.W2 = np.random.randn(h, d) * scale_out
        self.b2 = np.zeros(d)
        self._cache: dict = {}

    # GELU e sua derivada (aproximação de Hendrycks & Gimpel, 2016)
    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _gelu_grad(x: np.ndarray) -> np.ndarray:
        c   = np.sqrt(2.0 / np.pi)
        t   = np.tanh(c * (x + 0.044715 * x ** 3))
        sech2 = 1.0 - t ** 2
        return 0.5 * (1.0 + t) + 0.5 * x * sech2 * c * (1.0 + 3 * 0.044715 * x ** 2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1   = x @ self.W1 + self.b1          # (B, T, 4d)
        act  = self._gelu(z1)
        out  = act @ self.W2 + self.b2        # (B, T, d)
        self._cache = dict(x=x, z1=z1, act=act)
        return out

    def backward(self, dout: np.ndarray):
        c  = self._cache
        B, T, d = dout.shape

        # Linear 2
        d_W2  = c["act"].reshape(B * T, -1).T @ dout.reshape(B * T, d)
        d_b2  = dout.sum(axis=(0, 1))
        d_act = dout @ self.W2.T              # (B, T, 4d)

        # GELU
        d_z1  = d_act * self._gelu_grad(c["z1"])

        # Linear 1
        d_W1  = c["x"].reshape(B * T, d).T @ d_z1.reshape(B * T, -1)
        d_b1  = d_z1.sum(axis=(0, 1))
        d_x   = d_z1 @ self.W1.T             # (B, T, d)

        grads = dict(W1=d_W1, b1=d_b1, W2=d_W2, b2=d_b2)
        return d_x, grads


# ─────────────────────────────────────────────
#  Transformer Block (pre-LN)
# ─────────────────────────────────────────────

class TransformerBlock:
    """
    Um bloco Transformer completo com pre-layer normalization.

    Pré-LN (normalizar ANTES da sublayer) é mais estável que
    a formulação original pós-LN do Attention is All You Need —
    é o que GPT-2 e versões posteriores usam.

    Fluxo:
        x  →  LN1  →  CausalSelfAttention  →  x + residual
           →  LN2  →  FeedForward          →  x + residual
    """

    def __init__(self, cfg: GPTConfig):
        self.ln1  = LayerNorm(cfg.n_embd, cfg.eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = LayerNorm(cfg.n_embd, cfg.eps)
        self.ffn  = FeedForward(cfg)
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Sublayer 1: atenção com conexão residual
        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out)
        x = x + attn_out

        # Sublayer 2: FFN com conexão residual
        ln2_out = self.ln2.forward(x)
        ffn_out = self.ffn.forward(ln2_out)
        x = x + ffn_out

        self._cache = dict(ln1_out=ln1_out, attn_out=attn_out,
                           ln2_out=ln2_out,  ffn_out=ffn_out)
        return x

    def backward(self, dout: np.ndarray):
        """
        Retorna (d_input, grads_dict).
        Todos os gradientes dos parâmetros internos são coletados em grads_dict.
        """
        grads = {}
        c = self._cache

        # ── Sublayer 2 (FFN) ──────────────────────────────────
        d_ffn_out = dout                           # gradiente do residual: soma
        d_ln2_out, grads_ffn = self.ffn.backward(d_ffn_out)
        grads.update({f"ffn.{k}": v for k, v in grads_ffn.items()})

        d_x_after_attn, d_gamma2, d_beta2 = self.ln2.backward(d_ln2_out)
        grads["ln2.gamma"] = d_gamma2
        grads["ln2.beta"]  = d_beta2
        # soma do residual: o grad flui pela LN E pelo atalho
        d_x_after_attn = d_x_after_attn + dout

        # ── Sublayer 1 (Atenção) ──────────────────────────────
        d_attn_out = d_x_after_attn
        d_ln1_out, grads_attn = self.attn.backward(d_attn_out)
        grads.update({f"attn.{k}": v for k, v in grads_attn.items()})

        d_x_in, d_gamma1, d_beta1 = self.ln1.backward(d_ln1_out)
        grads["ln1.gamma"] = d_gamma1
        grads["ln1.beta"]  = d_beta1
        d_x_in = d_x_in + d_x_after_attn  # atalho residual

        return d_x_in, grads

    def train(self):
        self.attn.training = True

    def eval(self):
        self.attn.training = False


# ─────────────────────────────────────────────
#  Embeddings (Token + Posicional)
# ─────────────────────────────────────────────

class Embeddings:
    """
    Soma de embedding de token + embedding posicional aprendido.

    Positional encoding aprendido (vs. sinusoidal do artigo original)
    é a escolha do GPT-2: mais flexível, mas requer um corpus
    suficientemente grande para aprender as posições.
    """

    def __init__(self, cfg: GPTConfig):
        scale = 0.02
        self.tok_emb = np.random.randn(cfg.vocab_size, cfg.n_embd) * scale
        self.pos_emb = np.random.randn(cfg.ctx_len,    cfg.n_embd) * scale
        self._cache: dict = {}

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        idx: (B, T) inteiros
        retorna: (B, T, n_embd)
        """
        B, T = idx.shape
        te = self.tok_emb[idx]                      # (B, T, C)
        pe = self.pos_emb[:T]                       # (T, C)  broadcast
        self._cache = dict(idx=idx, B=B, T=T)
        return te + pe

    def backward(self, dout: np.ndarray):
        """
        dout: (B, T, C)
        Retorna grads para tok_emb e pos_emb (scatter add).
        """
        idx = self._cache["idx"]
        B, T, C = dout.shape

        d_tok = np.zeros_like(self.tok_emb)
        np.add.at(d_tok, idx, dout)                 # scatter add

        d_pos = dout.sum(axis=0)                    # (T, C)

        return dict(tok_emb=d_tok, pos_emb=d_pos)


# ─────────────────────────────────────────────
#  Modelo GPT completo
# ─────────────────────────────────────────────

class GPT:
    """
    Modelo GPT completo em NumPy puro.

    Camadas:
        Embeddings  (token + posicional)
        N × TransformerBlock
        LayerNorm final
        LM head  (projeção linear → logits sobre vocab)

    O LM head compartilha pesos com tok_emb (weight tying),
    o que reduz parâmetros e é padrão desde o GPT-1.
    """

    def __init__(self, cfg: GPTConfig):
        np.random.seed(cfg.seed)
        self.cfg   = cfg
        self.emb   = Embeddings(cfg)
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.ln_f  = LayerNorm(cfg.n_embd, cfg.eps)
        # LM head: sem bias (padrão GPT-2); weight tying com tok_emb
        # Nota: o weight tying significa que self.emb.tok_emb e a
        #       projeção final compartilham o MESMO array NumPy.
        #       Gradientes de ambos são acumulados em d_tok_emb.
        self._cache: dict = {}

    # ── forward ────────────────────────────────────────────────
    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        idx: (B, T) inteiros em [0, vocab_size)
        retorna logits: (B, T, vocab_size)
        """
        x = self.emb.forward(idx)                   # (B, T, C)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)                    # (B, T, C)
        # weight tying: usa tok_emb como projeção final
        logits = x @ self.emb.tok_emb.T             # (B, T, V)
        self._cache = dict(x_before_head=x)
        return logits

    # ── loss ───────────────────────────────────────────────────
    def loss(self, logits: np.ndarray, targets: np.ndarray):
        """
        Cross-entropy causal: prediz token t+1 dado t.

        logits:  (B, T, V)
        targets: (B, T) inteiros

        Retorna (loss_escalar, d_logits (B, T, V)).
        """
        B, T, V = logits.shape
        # ── Softmax numericamente estável ──────────────────────
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_l   = np.exp(shifted)
        probs   = exp_l / exp_l.sum(axis=-1, keepdims=True)

        # ── NLL nos tokens alvo ────────────────────────────────
        log_probs = np.log(probs + 1e-9)
        # flatten batch e tempo
        flat_log   = log_probs.reshape(B * T, V)
        flat_tgt   = targets.reshape(B * T)
        loss_val   = -flat_log[np.arange(B * T), flat_tgt].mean()

        # ── Gradiente: (probs - one_hot) / (B*T) ──────────────
        d_logits          = probs.reshape(B * T, V).copy()
        d_logits[np.arange(B * T), flat_tgt] -= 1.0
        d_logits          /= B * T
        d_logits           = d_logits.reshape(B, T, V)

        return loss_val, d_logits

    # ── backward ───────────────────────────────────────────────
    def backward(self, d_logits: np.ndarray) -> dict:
        """
        d_logits: (B, T, V)
        Retorna um dicionário plano com todos os gradientes.
        """
        all_grads: dict = {}
        x = self._cache["x_before_head"]

        # ── LM head (weight tying) ─────────────────────────────
        B, T, V = d_logits.shape
        C = x.shape[-1]
        # d_x = d_logits @ tok_emb  (de x @ tok_emb.T)
        d_x_head     = d_logits @ self.emb.tok_emb          # (B, T, C)
        # gradiente nos pesos do LM head == tok_emb
        d_tok_emb_lm = d_logits.reshape(B * T, V).T @ x.reshape(B * T, C)  # (V, C)
        all_grads["lm_head.tok_emb"] = d_tok_emb_lm         # acumulado no optimizer

        # ── LayerNorm final ────────────────────────────────────
        d_x, d_gamma_f, d_beta_f = self.ln_f.backward(d_x_head)
        all_grads["ln_f.gamma"] = d_gamma_f
        all_grads["ln_f.beta"]  = d_beta_f

        # ── Blocos Transformer (na ordem inversa) ──────────────
        for i, block in enumerate(reversed(self.blocks)):
            d_x, block_grads = block.backward(d_x)
            for k, v in block_grads.items():
                all_grads[f"block{self.cfg.n_layers - 1 - i}.{k}"] = v

        # ── Embeddings ─────────────────────────────────────────
        emb_grads = self.emb.backward(d_x)
        all_grads["emb.tok_emb"] = emb_grads["tok_emb"]
        all_grads["emb.pos_emb"] = emb_grads["pos_emb"]

        # ── Weight tying: acumula gradientes do LM head em tok_emb
        # d_tok_emb_lm tem shape (V, C) == tok_emb.shape → soma direta
        all_grads["emb.tok_emb"] = (
            all_grads["emb.tok_emb"] + d_tok_emb_lm
        )
        del all_grads["lm_head.tok_emb"]   # evita duplicidade

        return all_grads

    # ── train / eval ───────────────────────────────────────────
    def train(self):
        for b in self.blocks:
            b.train()

    def eval(self):
        for b in self.blocks:
            b.eval()

    # ── geração ────────────────────────────────────────────────
    def generate(self, idx: np.ndarray, max_new: int,
                 temperature: float = 1.0, top_k: int = 0) -> np.ndarray:
        """
        Geração auto-regressiva token a token.

        idx: (1, T) inteiros — prompt inicial
        temperature: divide os logits antes do softmax.
                     < 1  → distribuição mais concentrada (mais determinístico)
                     > 1  → distribuição mais plana (mais criativo)
        top_k: se > 0, restringe amostragem aos k tokens mais prováveis
               (top-k sampling do GPT-2).
        """
        self.eval()
        for _ in range(max_new):
            T = idx.shape[1]
            ctx = idx[:, -self.cfg.ctx_len:]        # janela de contexto
            logits = self.forward(ctx)              # (1, T', V)
            logits_last = logits[:, -1, :] / temperature  # (1, V)

            if top_k > 0:
                # zera tudo exceto os top-k logits
                topk_vals = np.sort(logits_last, axis=-1)[:, -top_k:]
                thresh = topk_vals[:, 0:1]
                logits_last = np.where(logits_last >= thresh, logits_last, -1e9)

            # Softmax → probabilidades
            shifted = logits_last - logits_last.max(axis=-1, keepdims=True)
            probs   = np.exp(shifted)
            probs  /= probs.sum(axis=-1, keepdims=True)

            # Amostragem multinomial
            next_tok = np.array([[np.random.choice(self.cfg.vocab_size, p=probs[0])]])
            idx = np.concatenate([idx, next_tok], axis=1)

        return idx


# ─────────────────────────────────────────────
#  Otimizador AdamW
# ─────────────────────────────────────────────

class AdamW:
    """
    AdamW = Adam + weight decay L2 aplicado ANTES da atualização
    (Loshchilov & Hutter, 2019).

    A diferença em relação ao Adam padrão é que o weight decay é
    aplicado diretamente nos pesos, não nos gradientes — o que é
    matematicamente mais correto para regularização L2 em Adam.

    Parâmetros sem peso (biases, gains de LayerNorm, embeddings)
    não sofrem weight decay (no_decay_keys).
    """

    NO_DECAY_SUFFIXES = (".beta", ".gamma", "emb.tok_emb", "emb.pos_emb")

    def __init__(self, lr: float = 3e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.1):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.wd  = weight_decay
        self.t   = 0
        self.m:  dict = {}
        self.v:  dict = {}

    def step(self, params: dict, grads: dict):
        """
        params: dict  key → np.ndarray  (referências diretas aos pesos)
        grads:  dict  key → np.ndarray  (mesmos keys)
        """
        self.t += 1
        for key, g in grads.items():
            if key not in params:
                continue
            p = params[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)

            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g ** 2

            m_hat = self.m[key] / (1 - self.b1 ** self.t)
            v_hat = self.v[key] / (1 - self.b2 ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Weight decay (não aplicado a biases / layer norms / embeddings)
            if not any(key.endswith(s) or key.startswith(s.lstrip("."))
                       for s in self.NO_DECAY_SUFFIXES):
                update += self.lr * self.wd * p

            p -= update


# ─────────────────────────────────────────────
#  Construção do dicionário de parâmetros
# ─────────────────────────────────────────────

def _collect_params(model: GPT) -> dict:
    """
    Retorna um dicionário {nome_chave: array_NumPy} com referência
    direta a todos os parâmetros treináveis do modelo.
    Usado pelo AdamW para atualização in-place.
    """
    p: dict = {}
    p["emb.tok_emb"] = model.emb.tok_emb
    p["emb.pos_emb"] = model.emb.pos_emb

    for i, block in enumerate(model.blocks):
        prefix = f"block{i}"
        p[f"{prefix}.attn.W_qkv"] = block.attn.W_qkv
        p[f"{prefix}.attn.b_qkv"] = block.attn.b_qkv
        p[f"{prefix}.attn.W_o"]   = block.attn.W_o
        p[f"{prefix}.attn.b_o"]   = block.attn.b_o
        p[f"{prefix}.ffn.W1"]     = block.ffn.W1
        p[f"{prefix}.ffn.b1"]     = block.ffn.b1
        p[f"{prefix}.ffn.W2"]     = block.ffn.W2
        p[f"{prefix}.ffn.b2"]     = block.ffn.b2
        p[f"{prefix}.ln1.gamma"]  = block.ln1.gamma
        p[f"{prefix}.ln1.beta"]   = block.ln1.beta
        p[f"{prefix}.ln2.gamma"]  = block.ln2.gamma
        p[f"{prefix}.ln2.beta"]   = block.ln2.beta

    p["ln_f.gamma"] = model.ln_f.gamma
    p["ln_f.beta"]  = model.ln_f.beta
    return p


def count_params(model: GPT) -> int:
    return sum(p.size for p in _collect_params(model).values())


# ─────────────────────────────────────────────
#  Loop de Treinamento
# ─────────────────────────────────────────────

class Trainer:
    """
    Encapsula o loop de treinamento do GPT.

    O loop é intencionalmente simples:
      1. Corta o texto em janelas de (ctx_len + 1) tokens.
      2. Forward pass → logits.
      3. Calcula cross-entropy entre logits[t] e target[t+1].
      4. Backward pass → gradientes.
      5. AdamW atualiza os pesos.

    Grad clipping por norma global (max_grad_norm) evita explosão
    de gradiente — especialmente crítico nas primeiras épocas.
    """

    def __init__(self, model: GPT, tokenizer: CharTokenizer,
                 cfg: GPTConfig):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.params    = _collect_params(model)
        self.optimizer = AdamW()

    # ----------------------------------------------------------
    def _make_batches(self, ids: np.ndarray, batch_size: int):
        """Gera (X, Y) onde Y = X deslocado em 1 (next-token prediction)."""
        T    = self.cfg.ctx_len
        N    = (len(ids) - 1) // T
        ids  = ids[: N * T + 1]
        xs   = []
        ys   = []
        for i in range(0, N * T, T):
            xs.append(ids[i:     i + T])
            ys.append(ids[i + 1: i + T + 1])
        xs = np.array(xs)   # (N, T)
        ys = np.array(ys)   # (N, T)
        # embaralha
        perm = np.random.permutation(N)
        xs, ys = xs[perm], ys[perm]
        for i in range(0, N, batch_size):
            yield xs[i:i + batch_size], ys[i:i + batch_size]

    # ----------------------------------------------------------
    def _clip_grads(self, grads: dict, max_norm: float = 1.0):
        """Global gradient clipping por norma L2."""
        total_norm = np.sqrt(sum(np.sum(g ** 2)
                                 for g in grads.values() if g is not None))
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for k in grads:
                if grads[k] is not None:
                    grads[k] = grads[k] * clip_coef
        return grads

    # ----------------------------------------------------------
    def fit(self, text: str, epochs: int = 10, batch_size: int = 16,
            lr: float = 3e-4, max_grad_norm: float = 1.0,
            eval_interval: int = 1, verbose: bool = True):
        """Treina o modelo no texto fornecido."""
        self.optimizer.lr = lr
        ids = np.array(self.tokenizer.encode(text), dtype=np.int32)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches  = 0

            for X, Y in self._make_batches(ids, batch_size):
                logits         = self.model.forward(X)
                loss, d_logits = self.model.loss(logits, Y)
                grads          = self.model.backward(d_logits)
                grads          = self._clip_grads(grads, max_grad_norm)
                self.optimizer.step(self.params, grads)

                total_loss += loss
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)

            if verbose and epoch % eval_interval == 0:
                print(f"Epoch {epoch:3d}/{epochs} | loss {avg_loss:.4f} "
                      f"| perplexity {np.exp(avg_loss):.2f}")

    # ----------------------------------------------------------
    def generate(self, prompt: str, max_new: int = 200,
                 temperature: float = 0.8, top_k: int = 40) -> str:
        """Gera texto a partir de um prompt."""
        ids  = np.array([self.tokenizer.encode(prompt)], dtype=np.int32)
        out  = self.model.generate(ids, max_new=max_new,
                                   temperature=temperature, top_k=top_k)
        return self.tokenizer.decode(out[0].tolist())


# ─────────────────────────────────────────────
#  Demo / Smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Corpus mínimo para validar que tudo compila e treina
    _CORPUS = (
        "Ser ou não ser, eis a questão. "
        "Se mais não sou, não sou mais eu. "
        "A língua é a pátria. "
        "No princípio era o Verbo. "
    ) * 40   # repete para ter tokens suficientes

    print("── GPT do Zero — NumPy puro ──────────────────")
    tok = CharTokenizer(_CORPUS)
    cfg = GPTConfig(
        vocab_size = tok.vocab_size,
        ctx_len    = 32,
        n_embd     = 64,
        n_heads    = 2,
        n_layers   = 2,
        dropout    = 0.0,   # desligado no smoke test para reprodutibilidade
    )
    model   = GPT(cfg)
    trainer = Trainer(model, tok, cfg)

    print(f"Parâmetros totais: {count_params(model):,}")
    print(f"Vocab size:        {tok.vocab_size}")

    trainer.fit(_CORPUS, epochs=30, batch_size=8, lr=3e-3, verbose=True)

    prompt = "Ser ou não"
    result = trainer.generate(prompt, max_new=80, temperature=0.7, top_k=10)
    print(f"\nPrompt : {prompt!r}")
    print(f"Gerado : {result!r}")
    print("──────────────────────────────────────────────")
