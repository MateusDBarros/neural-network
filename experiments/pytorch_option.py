import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Data ─────────────────────────────────────────────────────────────────────
#
# Load and preprocess MNIST dataset, matching NeuralNetwork.py lines 241-256

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader handles batching (line 274: batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
#
# Matches NeuralNetwork.py line 271: [784, 128, 'dropout', 64, 'dropout', 10]
#
# nn.Sequential stacks layers just like our Network class
# nn.Linear(in, out) is our Layer — holds W and b internally
# nn.LeakyReLU(0.01) matches our relu() with alpha=0.01
# nn.Dropout(0.2) matches our DropoutLayer with rate=0.2
# nn.Softmax(dim=1) matches our SoftmaxOutputLayer
#
# PyTorch uses Kaiming/He initialization by default for Linear layers,
# matching our np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.2),
    nn.Linear(64, 10),
    nn.Softmax(dim=1),
)

# ── Loss & optimizer ──────────────────────────────────────────────────────────
#
# CrossEntropyLoss combines softmax + categorical cross-entropy
# But since we already have softmax in the model, we use NLLLoss
# (Negative Log Likelihood) which expects log probabilities
#
# Actually, better to remove softmax from model and use CrossEntropyLoss directly
# Let's rebuild the model properly:

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.LeakyReLU(0.01),
    nn.Dropout(0.2),
    nn.Linear(64, 10),
    # No activation here - CrossEntropyLoss includes softmax
)

# CrossEntropyLoss matches our categorical_cross_entropy from line 232
# Adam(lr=0.001) matches our Adam optimizer from line 272
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── Training loop ─────────────────────────────────────────────────────────────
#
# Mirrors our training loop from lines 276-292
# Structure: forward → loss → backward → update weights
#
# Key differences from our manual implementation:
#   - optimizer.zero_grad() clears gradients (PyTorch accumulates by default)
#   - loss.backward() replaces our manual categorical_cross_entropy_grad + net.backward()
#     PyTorch's autograd builds computation graph during forward() and walks it automatically
#   - optimizer.step() applies weight updates (our W -= lr * grad_W)
#   - model.train() / model.eval() handle dropout behavior automatically

epochs = 20
test_acc = 0.0  # Initialize to avoid unbound variable

for epoch in range(epochs):
    model.train()  # Enable dropout
    total_loss = 0
    batches = 0
    
    for X_batch, y_batch in train_loader:
        # Forward pass
        predicted = model(X_batch)
        loss = loss_fn(predicted, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    # Evaluation on test set
    model.eval()  # Disable dropout
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predicted = model(X_batch)
            pred_classes = predicted.argmax(dim=1)
            correct += (pred_classes == y_batch).sum().item()
            total += y_batch.size(0)
    
    test_acc = correct / total
    avg_loss = total_loss / batches
    
    # Match output format from line 293
    print(f"Epoch {epoch+1:2d} — Loss: {avg_loss:.4f}  Test accuracy: {test_acc*100:.1f}%")

# ── Results ───────────────────────────────────────────────────────────────────

print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")