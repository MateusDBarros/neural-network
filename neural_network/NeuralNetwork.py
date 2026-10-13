import numpy as np
from keras.datasets import mnist
from torch.nn import Flatten


# Layers

class Layer:
    """"Hidden layer - ReLU"""

    def __init__(self, n_inputs, n_neurons):
        self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros(n_neurons)
        self.last_input = None
        self.last_z = None


    def relu(self, z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    def relu_derivative(self, z, alpha=0.01):
        return np.where(z > 0, 1.0, alpha)

    def forward(self, X):
        self.last_input = X
        self.last_z = X @ self.W + self.b # batch, n_neurons
        return self.relu(self.last_z)


    def backward(self, grad_outputs):

        grad_z = grad_outputs * self.relu_derivative(self.last_z)

        grad_W = self.last_input.T @ grad_z
        grad_b = grad_z.sum(axis=0)
        grad_input = grad_z @ self.W.T

        return grad_input, grad_W, grad_b


class OutputLayer(Layer):
    """"Output Layer - Sigmoid activation for binary classification."""

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        self.last_input = X
        self.last_z = X @ self.W + self.b
        return self.sigmoid(self.last_z)

    def backward(self, grad_outputs):

        grad_z = grad_outputs * self.sigmoid_derivative(self.last_z)
        grad_W = self.last_input.T @ grad_z
        grad_b = grad_z.sum(axis=0)
        grad_input = grad_z @ self.W.T
        return grad_input, grad_W, grad_b


class SoftmaxOutputLayer(Layer):

    def softmax(self, z):
        # subtract max per row for numerical stability to prevent exp overflow
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.last_input = X
        self.last_z = X @ self.W + self.b
        return self.softmax(self.last_z)

    def backward(self, grad_output):
        # gradient of softmax + cross entropy simplifies to (predict - actual)
        # which is already computed outside, so grad_output passes through

        grad_W = self.last_input.T @ grad_output
        grad_b = grad_output.sum(axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input, grad_W, grad_b


class DropoutLayer:

    def __init__(self, rate=0.2):
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.rate) / (1 - self.rate)
            return X * self.mask

        return X

    def backward(self, grad_output):
        return grad_output * self.mask, None, None


class BatchNormLayer:
    def __init__(self, n_inputs, momentum=0.9, epsilon=1e-8):
        self.gamma = np.ones(n_inputs)   # learned scale
        self.beta = np.zeros(n_inputs)  # learned shift
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros(n_inputs)
        self.running_var  = np.ones(n_inputs)
        self.x_norm  = None
        self.std     = None
        self.last_input = None
        self.training = True

    def forward(self, X):
        if self.training:
            mean = X.mean(axis=0)
            var  = X.var(axis=0)
            self.std = np.sqrt(var + self.epsilon)
            self.x_norm = (X - mean) / self.std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            std = np.sqrt(self.running_var + self.epsilon)
            self.x_norm = (X - self.running_mean) / std

        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        N = grad_output.shape[0]
        grad_gamma = (grad_output * self.x_norm).sum(axis=0)
        grad_beta = grad_output.sum(axis=0)
        dx_norm = grad_output * self.gamma
        grad_input = (1 / (N * self.std)) * (
            N * dx_norm - dx_norm.sum(axis=0) - self.x_norm * (dx_norm * self.x_norm).sum(axis=0)
        )
        return grad_input, grad_gamma, grad_beta

# Network

class Network:

    def __init__(self, layer_size, output='sigmoid'):

        output_map = {
            'sigmoid': OutputLayer,
            'softmax': SoftmaxOutputLayer,
        }
        self.layers = []
        i = 0

        while i < len(layer_size) - 1:
            size = layer_size[i]

            if layer_size[i] == 'dropout':
                self.layers.append(DropoutLayer())
                i += 1
                continue

            if layer_size[i] == 'batchnorm':
                prev = next(s for s in reversed(layer_size[:i]) if isinstance(s, int))
                self.layers.append(BatchNormLayer(prev))
                i += 1
                continue

            next_i = i + 1
            while next_i < len(layer_size) and not isinstance(layer_size[next_i], int):
                next_i += 1

            if next_i >= len(layer_size):
                break

            is_output = (next_i == len(layer_size) - 1)
            LayerClass = output_map[output] if is_output else Layer
            self.layers.append(LayerClass(layer_size[i], layer_size[next_i]))
            i = next_i

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad, optimizer):
        opt_i = 0
        for layer in reversed(self.layers):
            grad, grad_W, grad_b = layer.backward(grad)
            if grad_W is not None:
                optimizer.update(layer, grad_W, grad_b, layer_id=opt_i)
                opt_i += 1

    def train(self):
        for layer in self.layers:
            if isinstance(layer, (DropoutLayer, BatchNormLayer)):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, (DropoutLayer, BatchNormLayer)):
                layer.training = False




# ADAM optimizer

class Adam:
    """"
    Adaptive Moment Estimation

    Tracks a running average of gradients (m) and squared gradients (v)
    per weight. Weights that have been moving a lot get smaller steps;
    weights that haven't moved get larger steps.

    Default hyperparameters are the ones from the original paper and work
    well for most problems without tuning.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1 # smoothing gradient mean
        self.beta2 = beta2 # smoothing gradient variance
        self.epsilon = epsilon # prevent division by zero
        self.t = 0 # global timestamp
        self.m = {} #first moment (per layer)
        self.v = {} #second moment (per layer)


    def update(self, layer, grad_W, grad_b, layer_id):
        self.t += 1

        if layer_id not in self.m:
            self.m[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(grad_b)}
            self.v[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(grad_b)}

        # update biased moments
        self.m[layer_id]['W'] = self.beta1 * self.m[layer_id]['W'] + (1 - self.beta1) * grad_W
        self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * grad_b
        self.v[layer_id]['W'] = self.beta2 * self.v[layer_id]['W'] + (1 - self.beta2) * grad_W ** 2
        self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * grad_b ** 2

        # bias correction
        m_hat_W = self.m [layer_id]['W'] / (1 - self.beta1 ** self.t)
        m_hat_b = self.m [layer_id]['b'] / (1 - self.beta1 ** self.t)
        v_hat_W = self.v [layer_id]['W'] / (1 - self.beta2 ** self.t)
        v_hat_b = self.v [layer_id]['b'] / (1 - self.beta2 ** self.t)

        grad_W_attr = 'gamma' if hasattr(layer, 'gamma') else 'W'
        grad_b_attr = 'beta' if hasattr(layer, 'beta') else 'b'

        getattr(layer, grad_W_attr)[:] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        getattr(layer, grad_b_attr)[:] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


# Loss

def binary_cross_entropy(predicted, actual):
    predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
    return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

def binary_cross_entropy_grad(predicted, actual):
    predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
    return (predicted - actual) / (predicted * (1 - predicted)) / predicted.shape[0]

def mse_loss(predicted, actual):
    return np.mean((predicted - actual) ** 2)

def mse_grad(predicted, actual):
    return 2 * (predicted - actual) / predicted.shape[0]

def categorical_cross_entropy(predicted, actual):
    predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
    return -np.mean(np.sum(actual * np.log(predicted), axis=1))

def categorical_cross_entropy_grad(predicted, actual):
    return (predicted - actual) / predicted.shape[0]


class Conv2D:
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding='same'):
        self.W = np.random.randn(out_ch, in_ch, kernel, kernel) * np.sqrt(2.0/(in_ch*kernel*kernel))
        self.b = np.zeros(out_ch)
        self.stride = stride
        self.padding = padding
        self.kernel = kernel
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.last_input = None
        self.last_input_shape = None
        
    def _get_padding(self, H, W):
        """Calculate padding based on padding mode"""
        if self.padding == 'same':
            pad_h = ((H - 1) * self.stride + self.kernel - H) // 2
            pad_w = ((W - 1) * self.stride + self.kernel - W) // 2
            return max(0, pad_h), max(0, pad_w)
        elif self.padding == 'valid':
            return 0, 0
        else:
            return self.padding, self.padding
    
    def _pad_input(self, X, pad_h, pad_w):
        """Pad input tensor"""
        if pad_h == 0 and pad_w == 0:
            return X
        return np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    def _im2col(self, X, pad_h, pad_w):
        """Convert image to column matrix for efficient convolution"""
        N, C, H, W = X.shape
        
        # Pad input
        X_padded = self._pad_input(X, pad_h, pad_w)
        
        # Calculate output dimensions
        out_h = (H + 2 * pad_h - self.kernel) // self.stride + 1
        out_w = (W + 2 * pad_w - self.kernel) // self.stride + 1
        
        # Create column matrix
        col = np.zeros((N, C, self.kernel, self.kernel, out_h, out_w))
        
        for y in range(self.kernel):
            y_max = y + self.stride * out_h
            for x in range(self.kernel):
                x_max = x + self.stride * out_w
                col[:, :, y, x, :, :] = X_padded[:, :, y:y_max:self.stride, x:x_max:self.stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col, out_h, out_w
    
    def _col2im(self, col, input_shape, pad_h, pad_w):
        """Convert column matrix back to image"""
        N, C, H, W = input_shape
        H_padded, W_padded = H + 2 * pad_h, W + 2 * pad_w
        
        out_h = (H + 2 * pad_h - self.kernel) // self.stride + 1
        out_w = (W + 2 * pad_w - self.kernel) // self.stride + 1
        
        col = col.reshape(N, out_h, out_w, C, self.kernel, self.kernel).transpose(0, 3, 4, 5, 1, 2)
        
        img = np.zeros((N, C, H_padded, W_padded))
        
        for y in range(self.kernel):
            y_max = y + self.stride * out_h
            for x in range(self.kernel):
                x_max = x + self.stride * out_w
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            return img[:, :, pad_h:-pad_h if pad_h > 0 else None,
                      pad_w:-pad_w if pad_w > 0 else None]
        return img

    def forward(self, X):
        """
        Forward pass for 2D convolution
        X shape: (N, C, H, W) - batch, channels, height, width
        Returns: (N, out_ch, out_h, out_w)
        """
        self.last_input_shape = X.shape
        N, C, H, W = X.shape
        
        # Get padding
        pad_h, pad_w = self._get_padding(H, W)
        
        # Convert to column matrix
        col, out_h, out_w = self._im2col(X, pad_h, pad_w)
        self.last_input = col
        
        # Reshape weights for matrix multiplication
        W_col = self.W.reshape(self.out_ch, -1)
        
        # Perform convolution via matrix multiplication
        out = (col @ W_col.T) + self.b
        
        # Reshape output
        out = out.reshape(N, out_h, out_w, self.out_ch).transpose(0, 3, 1, 2)
        
        return out

    def backward(self, grad_out):
        """
        Backward pass for 2D convolution
        grad_out shape: (N, out_ch, out_h, out_w)
        Returns: grad_input, grad_W, grad_b
        """
        N, _, out_h, out_w = grad_out.shape
        _, C, H, W = self.last_input_shape
        
        # Reshape gradient
        grad_out_reshaped = grad_out.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)
        
        # Gradient w.r.t. bias
        grad_b = grad_out_reshaped.sum(axis=0)
        
        # Gradient w.r.t. weights
        grad_W = (grad_out_reshaped.T @ self.last_input).reshape(self.W.shape)
        
        # Gradient w.r.t. input
        W_col = self.W.reshape(self.out_ch, -1)
        grad_col = grad_out_reshaped @ W_col
        
        # Convert back to image format
        pad_h, pad_w = self._get_padding(H, W)
        grad_input = self._col2im(grad_col, self.last_input_shape, pad_h, pad_w)
        
        return grad_input, grad_W, grad_b


class MaxPool2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.last_input = None
        self.mask = None
        
    def forward(self, X):
        """
        Forward pass for max pooling
        X shape: (N, C, H, W)
        Returns: (N, C, out_h, out_w)
        """
        self.last_input = X
        N, C, H, W = X.shape
        
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        # Reshape for pooling
        X_reshaped = X.reshape(N * C, 1, H, W)
        
        # Create output
        out = np.zeros((N * C, out_h, out_w))
        self.mask = np.zeros_like(X)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                window = X_reshaped[:, :, h_start:h_end, w_start:w_end]
                out[:, i, j] = np.max(window, axis=(1, 2, 3))
                
                # Create mask for backward pass
                for n in range(N * C):
                    window_n = X_reshaped[n, 0, h_start:h_end, w_start:w_end]
                    max_val = out[n, i, j]
                    mask_window = (window_n == max_val)
                    # Distribute gradient equally among max values
                    mask_window = mask_window / mask_window.sum()
                    self.mask.reshape(N * C, H, W)[n, h_start:h_end, w_start:w_end] += mask_window
        
        out = out.reshape(N, C, out_h, out_w)
        return out
    
    def backward(self, grad_out):
        """
        Backward pass for max pooling
        grad_out shape: (N, C, out_h, out_w)
        Returns: grad_input, None, None
        """
        N, C, out_h, out_w = grad_out.shape
        _, _, H, W = self.last_input.shape
        
        grad_input = np.zeros_like(self.last_input)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                grad_input[:, :, h_start:h_end, w_start:w_end] += \
                    (self.mask[:, :, h_start:h_end, w_start:w_end] *
                     grad_out[:, :, i:i+1, j:j+1])
        
        return grad_input, None, None


class FlattenLayer:
    def __init__(self):
        self.input_shape = None
        
    def forward(self, X):
        """
        Flatten 4D tensor to 2D
        X shape: (N, C, H, W)
        Returns: (N, C*H*W)
        """
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad_out):
        """
        Reshape gradient back to original shape
        grad_out shape: (N, C*H*W)
        Returns: (N, C, H, W), None, None
        """
        return grad_out.reshape(self.input_shape), None, None


class ReLULayer:
    def __init__(self):
        self.last_input = None
        
    def forward(self, X):
        """ReLU activation"""
        self.last_input = X
        return np.maximum(0, X)
    
    def backward(self, grad_out):
        """ReLU gradient"""
        grad_input = grad_out * (self.last_input > 0)
        return grad_input, None, None


class ConvNetwork:
    """
    Convolutional Neural Network for image classification
    Supports Conv2D, MaxPool2D, Flatten, ReLU, BatchNorm, Dropout layers
    """
    def __init__(self, layers_config):
        """
        layers_config: list of layer instances
        Example for MNIST:
        [
            Conv2D(1, 32, 3),           # 32 filters, 3x3 kernel
            ReLULayer(),
            MaxPool2D(2),               # 2x2 pooling
            Conv2D(32, 64, 3),
            ReLULayer(),
            MaxPool2D(2),
            FlattenLayer(),
            Layer(3136, 128),           # fully connected
            DropoutLayer(0.5),
            SoftmaxOutputLayer(128, 10) # 10 classes
        ]
        """
        self.layers = layers_config
        
    def forward(self, X):
        """Forward pass through all layers"""
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad, optimizer):
        """Backward pass through all layers"""
        opt_i = 0
        for layer in reversed(self.layers):
            grad, grad_W, grad_b = layer.backward(grad)
            if grad_W is not None:
                optimizer.update(layer, grad_W, grad_b, layer_id=opt_i)
                opt_i += 1
    
    def train(self):
        """Set layers to training mode"""
        for layer in self.layers:
            if isinstance(layer, (DropoutLayer, BatchNormLayer)):
                layer.training = True
    
    def eval(self):
        """Set layers to evaluation mode"""
        for layer in self.layers:
            if isinstance(layer, (DropoutLayer, BatchNormLayer)):
                layer.training = False


# Data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0


# Helpers


def one_hot(y, n_classes=10):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

def get_batches(X, Y, batch_size):
    indices = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        idx = indices[i:i + batch_size]
        yield X[idx], Y[idx]

def accuracy(predicted, actual):
    pred_classes = np.argmax(predicted, axis=1)
    actual_classes = np.argmax(actual, axis=1)
    return np.mean(pred_classes == actual_classes)

###### Training Loop Examples ######

# Example 1: Fully Connected Network for MNIST
net       = Network([784, 128, 'batchnorm', 'dropout', 64, 'batchnorm','dropout', 10], output='softmax')
optimizer = Adam(learning_rate=0.001)
epochs    = 20
batch_size = 64

# Example 2: Convolutional Neural Network for MNIST
# First, reshape data for CNN: (N, C, H, W) format
# X_train_cnn = X_train.reshape(-1, 1, 28, 28)
# X_test_cnn = X_test.reshape(-1, 1, 28, 28)
#
# cnn = ConvNetwork([
#     Conv2D(1, 32, 3, padding='same'),    # Input: (N, 1, 28, 28) -> Output: (N, 32, 28, 28)
#     ReLULayer(),
#     MaxPool2D(2),                         # (N, 32, 14, 14)
#     Conv2D(32, 64, 3, padding='same'),   # (N, 64, 14, 14)
#     ReLULayer(),
#     MaxPool2D(2),                         # (N, 64, 7, 7)
#     FlattenLayer(),                       # (N, 3136)
#     Layer(3136, 128),                     # Fully connected layer
#     DropoutLayer(0.5),
#     SoftmaxOutputLayer(128, 10)           # Output: (N, 10)
# ])
# optimizer_cnn = Adam(learning_rate=0.001)
#
# # Training loop for CNN (same as fully connected)
# # for epoch in range(epochs):
# #     cnn.train()
# #     total_loss = 0
# #     batches = 0
# #     for X_batch, Y_batch in get_batches(X_train_cnn, Y_train, batch_size):
# #         predicted = cnn.forward(X_batch)
# #         total_loss += categorical_cross_entropy(predicted, Y_batch)
# #         grad = categorical_cross_entropy_grad(predicted, Y_batch)
# #         cnn.backward(grad, optimizer_cnn)
# #         batches += 1
# #     cnn.eval()
# #     test_pred = cnn.forward(X_test_cnn)
# #     test_acc = accuracy(test_pred, Y_test)
# #     avg_loss = total_loss / batches
# #     print(f"Epoch {epoch+1:2d} — Loss: {avg_loss:.4f}  Test accuracy: {test_acc*100:.1f}%")


for epoch in range(epochs):
    net.train()
    total_loss = 0
    batches    = 0

    for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size):
        predicted   = net.forward(X_batch)
        total_loss += categorical_cross_entropy(predicted, Y_batch)
        grad        = categorical_cross_entropy_grad(predicted, Y_batch)
        net.backward(grad, optimizer)
        batches += 1

    net.eval()
    test_pred = net.forward(X_test)
    test_acc  = accuracy(test_pred, Y_test)
    avg_loss  = total_loss / batches

    print(f"Epoch {epoch+1:2d} — Loss: {avg_loss:.4f}  Test accuracy: {test_acc*100:.1f}%") 