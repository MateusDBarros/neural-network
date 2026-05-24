import random
import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros(n_neurons)
        self.last_input = None
        self.last_z = None


    def relu(self, z):
        return np.maximum(0,z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.last_input = X
        self.last_z = X @ self.W + self.b # batch, n_neurons
        return self.relu(self.last_z)


    def backward(self, grad_outputs, learning_rate):

        grad_z = grad_outputs * self.relu_derivative(self.last_z)

        grad_W = self.last_input.T @ grad_z
        grad_b = grad_z.sum(axis=0)
        grad_input = grad_z @ self.W.T

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

        return grad_input


class Network:

    def __init__(self, layer_size):
        self.layers = []

        for i in range(len(layer_size) - 1):
            self.layers.append(Layer(n_inputs=layer_size[i],
                                     n_neurons=layer_size[i+1]))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


def mse_loss(predicted, actual):
    return sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)


X = [[1.0, 0.5, -1.0], [0.2, 0.8, 0.3]]
Y = [[1.0, 0.0], [0.0, 1.0]]

net = Network([3, 4, 2])

for epoch in range(100):
    total_loss = 0

    for x, y in zip(X, Y):
        predicted = net.forward(x)
        loss = mse_loss(predicted, y)
        total_loss += loss

        grad = [(p - a) * 2 / len(predicted) for p, a in zip(predicted, y)]
        net.backward(grad, learning_rate=0.01)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {total_loss:.4f}")

