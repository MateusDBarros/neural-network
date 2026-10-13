"""
Example: Training a Convolutional Neural Network on MNIST
This demonstrates how to use the Conv2D, MaxPool2D, and other CNN layers
"""

import numpy as np
from keras.datasets import mnist
from NeuralNetwork import (
    Conv2D, MaxPool2D, FlattenLayer, ReLULayer,
    Layer, SoftmaxOutputLayer, DropoutLayer, BatchNormLayer,
    ConvNetwork, Adam,
    categorical_cross_entropy, categorical_cross_entropy_grad
)

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape for CNN: (N, C, H, W) format - batch, channels, height, width
X_train_cnn = X_train.reshape(-1, 1, 28, 28) / 255.0
X_test_cnn = X_test.reshape(-1, 1, 28, 28) / 255.0

# One-hot encode labels
def one_hot(y, n_classes=10):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

# Helper functions
def get_batches(X, Y, batch_size):
    indices = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idx = indices[i:i + batch_size]
        yield X[idx], Y[idx]

def accuracy(predicted, actual):
    pred_classes = np.argmax(predicted, axis=1)
    actual_classes = np.argmax(actual, axis=1)
    return np.mean(pred_classes == actual_classes)

# Build CNN architecture
print("Building Convolutional Neural Network...")
print("Architecture:")
print("  Input: (N, 1, 28, 28)")
print("  Conv2D(1->32, 3x3) + ReLU -> (N, 32, 28, 28)")
print("  MaxPool(2x2) -> (N, 32, 14, 14)")
print("  Conv2D(32->64, 3x3) + ReLU -> (N, 64, 14, 14)")
print("  MaxPool(2x2) -> (N, 64, 7, 7)")
print("  Flatten -> (N, 3136)")
print("  Dense(3136->128) + Dropout(0.5)")
print("  Dense(128->10) + Softmax")
print()

cnn = ConvNetwork([
    Conv2D(1, 32, 3, padding='same'),      # 32 filters, 3x3 kernel
    ReLULayer(),
    MaxPool2D(2),                           # 2x2 pooling, stride 2
    Conv2D(32, 64, 3, padding='same'),     # 64 filters, 3x3 kernel
    ReLULayer(),
    MaxPool2D(2),
    FlattenLayer(),                         # Flatten to 1D
    Layer(3136, 128),                       # Fully connected layer
    DropoutLayer(0.5),                      # 50% dropout
    SoftmaxOutputLayer(128, 10)             # Output layer
])

# Initialize optimizer
optimizer = Adam(learning_rate=0.001)

# Training parameters
epochs = 10
batch_size = 64

print(f"Training for {epochs} epochs with batch size {batch_size}...")
print()

# Training loop
for epoch in range(epochs):
    cnn.train()
    total_loss = 0
    batches = 0
    
    for X_batch, Y_batch in get_batches(X_train_cnn, Y_train, batch_size):
        # Forward pass
        predicted = cnn.forward(X_batch)
        
        # Calculate loss
        loss = categorical_cross_entropy(predicted, Y_batch)
        total_loss += loss
        
        # Backward pass
        grad = categorical_cross_entropy_grad(predicted, Y_batch)
        cnn.backward(grad, optimizer)
        
        batches += 1
    
    # Evaluation
    cnn.eval()
    
    # Calculate training accuracy (on a subset for speed)
    train_sample_size = 1000
    train_sample_idx = np.random.choice(len(X_train_cnn), train_sample_size, replace=False)
    train_pred = cnn.forward(X_train_cnn[train_sample_idx])
    train_acc = accuracy(train_pred, Y_train[train_sample_idx])
    
    # Calculate test accuracy
    test_pred = cnn.forward(X_test_cnn)
    test_acc = accuracy(test_pred, Y_test)
    
    avg_loss = total_loss / batches
    
    print(f"Epoch {epoch+1:2d}/{epochs} — "
          f"Loss: {avg_loss:.4f}  "
          f"Train Acc: {train_acc*100:.2f}%  "
          f"Test Acc: {test_acc*100:.2f}%")

print()
print("Training complete!")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

# Made with Bob
