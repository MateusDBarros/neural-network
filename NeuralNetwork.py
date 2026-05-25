import numpy as np
from keras.datasets import mnist


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

###### Training Loop ######

net       = Network([784, 128, 'batchnorm', 'dropout', 64, 'batchnorm','dropout', 10], output='softmax')
optimizer = Adam(learning_rate=0.001)
epochs    = 20
batch_size = 64

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