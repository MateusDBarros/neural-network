# Convolutional Neural Network Implementation

This document describes the complete implementation of Convolutional Neural Networks (CNNs) in the `NeuralNetwork.py` file.

## Overview

The CNN implementation includes the following components:

### 1. **Conv2D Layer**
Performs 2D convolution operations on image data.

**Features:**
- Configurable number of input/output channels
- Adjustable kernel size
- Stride support
- Padding modes: 'same', 'valid', or custom integer padding
- Efficient im2col algorithm for fast convolution
- Full backward pass with gradients for weights, biases, and inputs

**Usage:**
```python
conv = Conv2D(in_channels=1, out_channels=32, kernel=3, stride=1, padding='same')
output = conv.forward(input)  # input shape: (N, C, H, W)
```

**Parameters:**
- `in_ch`: Number of input channels
- `out_ch`: Number of output channels (filters)
- `kernel`: Kernel size (assumes square kernel)
- `stride`: Stride for convolution (default: 1)
- `padding`: 'same', 'valid', or integer (default: 'same')

### 2. **MaxPool2D Layer**
Performs max pooling to reduce spatial dimensions.

**Features:**
- Configurable pool size
- Adjustable stride
- Tracks max value positions for backward pass
- Handles gradient distribution for tied max values

**Usage:**
```python
pool = MaxPool2D(pool_size=2, stride=2)
output = pool.forward(input)  # reduces spatial dimensions by factor of pool_size
```

**Parameters:**
- `pool_size`: Size of pooling window (default: 2)
- `stride`: Stride for pooling (default: same as pool_size)

### 3. **FlattenLayer**
Converts 4D tensor (N, C, H, W) to 2D tensor (N, C*H*W) for fully connected layers.

**Usage:**
```python
flatten = FlattenLayer()
output = flatten.forward(input)  # (N, C, H, W) -> (N, C*H*W)
```

### 4. **ReLULayer**
Applies ReLU activation function element-wise.

**Usage:**
```python
relu = ReLULayer()
output = relu.forward(input)  # applies max(0, x)
```

### 5. **ConvNetwork Class**
Container class for building and training CNNs.

**Features:**
- Accepts list of layer instances
- Automatic forward/backward propagation
- Training/evaluation mode switching
- Compatible with existing optimizers (Adam)

**Usage:**
```python
cnn = ConvNetwork([
    Conv2D(1, 32, 3, padding='same'),
    ReLULayer(),
    MaxPool2D(2),
    Conv2D(32, 64, 3, padding='same'),
    ReLULayer(),
    MaxPool2D(2),
    FlattenLayer(),
    Layer(3136, 128),
    DropoutLayer(0.5),
    SoftmaxOutputLayer(128, 10)
])
```

## Architecture Example: MNIST Classification

### Network Structure
```
Input: (N, 1, 28, 28)
    ↓
Conv2D(1→32, 3×3) + ReLU → (N, 32, 28, 28)
    ↓
MaxPool(2×2) → (N, 32, 14, 14)
    ↓
Conv2D(32→64, 3×3) + ReLU → (N, 64, 14, 14)
    ↓
MaxPool(2×2) → (N, 64, 7, 7)
    ↓
Flatten → (N, 3136)
    ↓
Dense(3136→128) + Dropout(0.5)
    ↓
Dense(128→10) + Softmax
    ↓
Output: (N, 10)
```

### Complete Training Example

```python
from NeuralNetwork import (
    Conv2D, MaxPool2D, FlattenLayer, ReLULayer,
    Layer, SoftmaxOutputLayer, DropoutLayer,
    ConvNetwork, Adam,
    categorical_cross_entropy, categorical_cross_entropy_grad
)

# Prepare data (reshape to CNN format)
X_train_cnn = X_train.reshape(-1, 1, 28, 28) / 255.0
X_test_cnn = X_test.reshape(-1, 1, 28, 28) / 255.0

# Build network
cnn = ConvNetwork([
    Conv2D(1, 32, 3, padding='same'),
    ReLULayer(),
    MaxPool2D(2),
    Conv2D(32, 64, 3, padding='same'),
    ReLULayer(),
    MaxPool2D(2),
    FlattenLayer(),
    Layer(3136, 128),
    DropoutLayer(0.5),
    SoftmaxOutputLayer(128, 10)
])

# Initialize optimizer
optimizer = Adam(learning_rate=0.001)

# Training loop
for epoch in range(epochs):
    cnn.train()
    for X_batch, Y_batch in get_batches(X_train_cnn, Y_train, batch_size):
        predicted = cnn.forward(X_batch)
        loss = categorical_cross_entropy(predicted, Y_batch)
        grad = categorical_cross_entropy_grad(predicted, Y_batch)
        cnn.backward(grad, optimizer)
    
    cnn.eval()
    test_pred = cnn.forward(X_test_cnn)
    test_acc = accuracy(test_pred, Y_test)
    print(f"Epoch {epoch+1} — Test Acc: {test_acc*100:.2f}%")
```

## Implementation Details

### Im2Col Algorithm
The Conv2D layer uses the im2col (image to column) algorithm for efficient convolution:

1. **Forward Pass:**
   - Pad input if needed
   - Convert image patches to columns
   - Perform matrix multiplication with reshaped weights
   - Reshape output to (N, out_ch, out_h, out_w)

2. **Backward Pass:**
   - Compute gradients for weights and biases
   - Convert column gradients back to image format (col2im)
   - Return gradients for input, weights, and biases

### Memory Efficiency
- Uses numpy's efficient array operations
- Minimizes temporary allocations
- Reuses buffers where possible

### Numerical Stability
- Proper weight initialization (He initialization)
- Gradient clipping in loss functions
- Careful handling of padding edge cases

## Performance Considerations

### Speed Optimization
- Im2col algorithm is faster than naive nested loops
- Matrix multiplication leverages optimized BLAS libraries
- Batch processing for efficient GPU utilization (if using GPU-enabled numpy)

### Memory Usage
- Im2col creates temporary column matrices (memory-intensive for large images)
- Consider using smaller batch sizes for large images
- MaxPool2D stores masks for backward pass

## Extending the Implementation

### Adding New Layer Types

To add a new layer type, implement:

```python
class CustomLayer:
    def __init__(self, ...):
        # Initialize parameters
        self.W = ...
        self.b = ...
        self.last_input = None
    
    def forward(self, X):
        # Store input for backward pass
        self.last_input = X
        # Compute output
        return output
    
    def backward(self, grad_out):
        # Compute gradients
        grad_input = ...
        grad_W = ...
        grad_b = ...
        return grad_input, grad_W, grad_b
```

### Tips for Custom Architectures

1. **Always match dimensions:** Ensure output of one layer matches input of next
2. **Use padding='same':** Keeps spatial dimensions constant (easier to track)
3. **Add ReLU after Conv2D:** Non-linearity is crucial for learning
4. **Use MaxPool for downsampling:** Reduces computation and adds translation invariance
5. **Flatten before fully connected:** Required to transition from conv to dense layers
6. **Add Dropout before output:** Helps prevent overfitting

## Common Issues and Solutions

### Issue: Shape Mismatch
**Problem:** Dimensions don't match between layers
**Solution:** Print shapes after each layer during forward pass to debug

### Issue: Gradient Explosion/Vanishing
**Problem:** Loss becomes NaN or doesn't decrease
**Solution:** 
- Use proper weight initialization (already implemented)
- Reduce learning rate
- Add batch normalization
- Use gradient clipping

### Issue: Slow Training
**Problem:** Training takes too long
**Solution:**
- Reduce batch size (but not too small)
- Use fewer filters or smaller kernels
- Reduce image resolution
- Use stride > 1 in convolutions

### Issue: Poor Accuracy
**Problem:** Model doesn't learn well
**Solution:**
- Increase model capacity (more filters/layers)
- Train for more epochs
- Adjust learning rate
- Add data augmentation
- Check data preprocessing

## Running the Example

To run the CNN example:

```bash
python cnn_example.py
```

This will train a CNN on MNIST and display training progress.

## References

- **Im2Col Algorithm:** Used for efficient convolution implementation
- **He Initialization:** Weight initialization for ReLU networks
- **Adam Optimizer:** Adaptive learning rate optimization
- **Dropout:** Regularization technique to prevent overfitting

## Future Enhancements

Potential improvements to the implementation:

1. **Batch Normalization for Conv2D:** Add BN specifically for convolutional layers
2. **Dilated Convolutions:** Support for dilated/atrous convolutions
3. **Depthwise Separable Convolutions:** More efficient convolutions
4. **Residual Connections:** Skip connections for deeper networks
5. **Average Pooling:** Alternative to max pooling
6. **Global Average Pooling:** Replace flatten + dense for classification
7. **1x1 Convolutions:** Channel-wise transformations
8. **Transposed Convolutions:** For upsampling (useful in autoencoders)