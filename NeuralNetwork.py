import random
import numpy as np



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


# Network

class Network:

    def __init__(self, layer_size):

        self.layers = []
        for i in range(len(layer_size) - 1):
            is_output = (i == len(layer_size) - 2)
            LayerClass = OutputLayer if is_output else Layer
            self.layers.append(LayerClass(layer_size[i], layer_size[i + 1]))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad, optimizer):
        for i, layer in enumerate(reversed(self.layers)):
            grad, grad_W, grad_b = layer.backward(grad)
            optimizer.update(layer, grad_W, grad_b, layer_id=i)




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
            self.m[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
            self.v[layer_id] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

        # update biased moments
        self.m[layer_id]['W'] = self.beta1 * self.m[layer_id]['W'] + (1 - self.beta1) * grad_W
        self.m[layer_id]['b'] = self.beta1 * self.m[layer_id]['b'] + (1 - self.beta1) * grad_b
        self.v[layer_id]['W'] = self.beta2 * self.v[layer_id]['W'] + (1 - self.beta2) * grad_W ** 2
        self.v[layer_id]['b'] = self.beta2 * self.v[layer_id]['b'] + (1 - self.beta2) * grad_b ** 2

        # bias correction
        m_hat_W =self.m [layer_id]['W'] / (1 - self.beta1 ** self.t)
        m_hat_b =self.m [layer_id]['b'] / (1 - self.beta1 ** self.t)
        v_hat_W =self.v [layer_id]['W'] / (1 - self.beta2 ** self.t)
        v_hat_b =self.v [layer_id]['b'] / (1 - self.beta2 ** self.t)

        # Weight uppdate ( step size shinks as v grows
        layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)




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



############



X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

net = Network([2, 8, 1])
optimizer = Adam(learning_rate=0.01)

for epoch in range(1000):
    predicted = net.forward(X)
    loss = binary_cross_entropy(predicted, Y)
    grad = binary_cross_entropy_grad(predicted, Y)
    net.backward(grad, optimizer)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} — Loss: {loss:.4f}")

print("\nFinal predictions:")
for x, y, p in zip(X, Y, net.forward(X)):
    print(f"  {x.tolist()} → expected {int(y[0])}  got {p[0]:.3f}")