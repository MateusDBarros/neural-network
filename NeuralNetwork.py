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


class Network:

    def __init__(self, layer_size):

        self.layers = [Layer(layer_size[i], layer_size[i+1])
                       for i in range(len(layer_size) - 1)]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)


def mse_loss(predicted, actual):
    return np.mean((predicted - actual) ** 2)

def mse_grad(predicted, actual):
    return 2 * (predicted - actual) / predicted.shape[0]


Y = np.array([[0], [1], [1], [0]], dtype=float)

X = np.array([
    [0, 0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)

net = Network([2, 8, 1])

for epoch in range(1000):

    predicted = net.forward(X)
    loss = mse_loss(predicted, Y)
    grad = mse_grad(predicted, Y)
    net.backward(grad, learning_rate=0.1)


    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} - Loss: {loss:.4f}")

print("\nFinal predictions:")
for x, y, p in zip(X, Y, net.forward(X)):
    print(f" {x} -> expected {y[0]} got {p[0]:.3f}")
