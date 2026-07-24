"""
run_report.py
=============
Treina o GPT em NumPy e imprime os artefatos necessários para o artigo:
  1. Dataset usado no treino
  2. Tamanho do modelo (parâmetros, camadas, heads)
  3. Loss final de treino
  4. Texto gerado (prompt → continuação)
"""

import json
import numpy as np
from gpt_numpy import GPTConfig, GPT, CharTokenizer, Trainer, count_params

# ─────────────────────────────────────────────────────────────
#  1. CORPUS DE TREINO
#     Trechos literários em português — Machado de Assis,
#     Fernando Pessoa e Guimarães Rosa — domínio público.
# ─────────────────────────────────────────────────────────────
CORPUS = """
Dom Casmurro, de Machado de Assis:
Minha mãe era boa criatura. Quando ficou viúva, aos trinta e um anos,
com dois filhos pequenos, o maior dos quais tinha sete anos, passou noites
e dias a chorar. Que eu me lembre, nunca ouvi dela uma palavra áspera.
Era dócil, amiga dos filhos, crente em Deus e nos santos.

Capitú tinha os olhos de ressaca. Olhos oblíquos e dissimulados.
Não pense o leitor que os olhos de ressaca são para intimidar;
ao contrário, são para seduzir. É a ondulação do mar que convida
o nadador a mergulhar nela. Capitú olhava assim.

O tempo passa, os homens mudam, os costumes mudam, a língua muda,
mas a alma humana é sempre a mesma. Os vícios e as virtudes dos nossos
avós são os nossos vícios e as nossas virtudes. Tudo muda de forma,
nada muda de essência.

Fernando Pessoa:
Navegar é preciso, viver não é preciso.
Não sou nada. Nunca serei nada. Não posso querer ser nada.
À parte isso, tenho em mim todos os sonhos do mundo.
Fingir é conhecer-se. Sou o intervalo entre o meu desejo
e aquilo que a vida fez de mim.

O que está em mim é a consciência de que não há nada.
Sou aquele que não chegou. O caminho é o mesmo,
o viajante é diferente. O mundo é uma ideia que temos do mundo.
Nada é real exceto a nossa percepção do real.

Guimarães Rosa:
O sertão está em toda parte. O sertão é do tamanho do mundo.
Viver é muito perigoso. A vida é assim: esquenta e esfria,
aperta e daí afrouxa, sossega e depois desinquieta.
O que ela quer da gente é coragem.

Riobaldo pensava: o diabo não existe, mas é preciso que exista para
que o homem possa se salvar. A coragem não é a ausência do medo;
é a decisão de que outra coisa é mais importante que o medo.
O amor é o princípio de tudo. O amor salva o que parece perdido.

A linguagem e a vida são a mesma coisa. Quem não entende a linguagem
não entende a vida. O sertão aceita todos os nomes — é no sertão
que as palavras adquirem o peso das pedras e a leveza das aves.
""".strip()

# Replica o corpus para ter volume suficiente de tokens de treino
CORPUS = CORPUS * 6

# ─────────────────────────────────────────────────────────────
#  2. CONFIGURAÇÃO DO MODELO
# ─────────────────────────────────────────────────────────────
tok = CharTokenizer(CORPUS)

cfg = GPTConfig(
    vocab_size = tok.vocab_size,
    ctx_len    = 128,
    n_embd     = 128,
    n_heads    = 4,
    n_layers   = 3,
    ffn_mult   = 4,
    dropout    = 0.0,
    seed       = 42,
)

model   = GPT(cfg)
trainer = Trainer(model, tok, cfg)

# ─────────────────────────────────────────────────────────────
#  3. TREINO
# ─────────────────────────────────────────────────────────────
N_EPOCHS   = 200
BATCH_SIZE = 32
LR         = 3e-3

print("Treinando GPT do Zero…")
loss_history = []

# Monkey-patch para capturar loss por época
_orig_fit = trainer.fit.__func__

def fit_tracked(self, text, epochs, batch_size, lr, max_grad_norm=1.0,
                eval_interval=1, verbose=True):
    from gpt_numpy import AdamW
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
        loss_history.append(float(avg_loss))
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {avg_loss:.4f} "
                  f"| perplexity {np.exp(avg_loss):.2f}")

import types
trainer.fit = types.MethodType(fit_tracked, trainer)
trainer.fit(CORPUS, epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR, verbose=True)

final_loss = loss_history[-1]

# ─────────────────────────────────────────────────────────────
#  4. GERAÇÃO
# ─────────────────────────────────────────────────────────────
PROMPTS = [
    ("Navegar é preciso", 0.4, 5),
    ("O sertão", 0.4, 5),
    ("Capitú tinha", 0.4, 5),
]

generations = {}
for p, temp, k in PROMPTS:
    out = trainer.generate(p, max_new=150, temperature=temp, top_k=k)
    generations[p] = out

# ─────────────────────────────────────────────────────────────
#  5. RELATÓRIO
# ─────────────────────────────────────────────────────────────
total_params = count_params(model)
param_detail = {
    "embedding (tok + pos)": (cfg.vocab_size + cfg.ctx_len) * cfg.n_embd,
    "por bloco (attn QKV + O + FFN + LNs)": (
        cfg.n_embd * 3 * cfg.n_embd + 3 * cfg.n_embd  # W_qkv + b_qkv
        + cfg.n_embd * cfg.n_embd + cfg.n_embd          # W_o + b_o
        + cfg.n_embd * cfg.n_embd * cfg.ffn_mult + cfg.n_embd * cfg.ffn_mult  # W1+b1
        + cfg.n_embd * cfg.ffn_mult * cfg.n_embd + cfg.n_embd                 # W2+b2
        + 4 * cfg.n_embd                                                       # 2x LN
    ),
    "LN final": 2 * cfg.n_embd,
}

report = {
    "dataset": {
        "descricao": "Trechos literários em português — Machado de Assis, Fernando Pessoa e Guimarães Rosa (domínio público)",
        "corpus_original_chars": len(CORPUS) // 6,
        "corpus_total_chars":    len(CORPUS),
        "vocab_size":            tok.vocab_size,
        "vocab_chars":           sorted(tok.stoi.keys()),
    },
    "modelo": {
        "arquitetura":       "GPT (decoder-only Transformer, pre-LayerNorm)",
        "n_layers":          cfg.n_layers,
        "n_heads":           cfg.n_heads,
        "d_model":           cfg.n_embd,
        "d_head":            cfg.n_embd // cfg.n_heads,
        "ctx_len":           cfg.ctx_len,
        "ffn_hidden":        cfg.n_embd * cfg.ffn_mult,
        "total_params":      total_params,
        "breakdown_params":  param_detail,
        "weight_tying":      True,
        "activation":        "GELU",
        "optimizer":         "AdamW (β1=0.9, β2=0.999, wd=0.1)",
    },
    "treino": {
        "epochs":        N_EPOCHS,
        "batch_size":    BATCH_SIZE,
        "learning_rate": LR,
        "loss_inicial":  round(loss_history[0],  4),
        "loss_epoch10":  round(loss_history[9],  4),
        "loss_epoch40":  round(loss_history[39], 4),
        "loss_final":    round(final_loss,        4),
        "perplexidade_final": round(float(np.exp(final_loss)), 4),
    },
    "geracao": {p: g for p, g in generations.items()},
}

# Salva JSON
with open("gpt_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# Imprime resumo legível
print("\n" + "═" * 62)
print("  RELATÓRIO — GPT DO ZERO (NumPy puro)")
print("═" * 62)

print("\n── 1. DATASET ──────────────────────────────────────────────")
print(f"  Corpus     : {report['dataset']['descricao']}")
print(f"  Chars únicos (vocab): {tok.vocab_size}")
print(f"  Total de chars treinados: {len(CORPUS):,}")
print(f"  Vocab: {''.join(report['dataset']['vocab_chars'])!r}")

print("\n── 2. TAMANHO DO MODELO ────────────────────────────────────")
print(f"  Arquitetura   : {report['modelo']['arquitetura']}")
print(f"  Camadas       : {cfg.n_layers}  |  Heads de atenção: {cfg.n_heads}")
print(f"  d_model       : {cfg.n_embd}  |  d_head: {cfg.n_embd // cfg.n_heads}")
print(f"  ctx_len       : {cfg.ctx_len}  |  FFN hidden: {cfg.n_embd * cfg.ffn_mult}")
print(f"  Total parâm.  : {total_params:,}")
for k, v in param_detail.items():
    print(f"    {k}: {v:,}")

print("\n── 3. LOSS DE TREINO ───────────────────────────────────────")
print(f"  Loss inicial (ep.1)  : {loss_history[0]:.4f}  (perplexidade {np.exp(loss_history[0]):.2f})")
print(f"  Loss ep.10           : {loss_history[9]:.4f}  (perplexidade {np.exp(loss_history[9]):.2f})")
print(f"  Loss ep.40           : {loss_history[39]:.4f}  (perplexidade {np.exp(loss_history[39]):.2f})")
print(f"  Loss final (ep.{N_EPOCHS})  : {final_loss:.4f}  (perplexidade {np.exp(final_loss):.2f})")

print("\n── 4. TEXTO GERADO ─────────────────────────────────────────")
for prompt, text in generations.items():
    print(f"\n  Prompt   : {prompt!r}")
    continuation = text[len(prompt):]
    print(f"  Gerado   : {continuation!r}")

print("\n" + "═" * 62)
print("  Relatório completo salvo em: gpt_report.json")
print("═" * 62)
