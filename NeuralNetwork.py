import random

class Neuron:

    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights
        self.last_input = None
        self.last_z = None

    def forward(self, inputs):
        self.last_input = inputs
        self.last_z = sum(w *  x for w, x in zip(self.weights, inputs)) + self.bias
        return self.relu(self.last_z)

    def relu(self, x):
        return max(0, x)

    def relu_derivative(self, x):
        return 1 if x > 0 else 0

    def backward(self, grad_output, learning_rate):

        grad_z = grad_output * self.relu_derivative(self.last_z)
        grad_weights = [grad_z * x for x in self.last_input]
        grad_input = [grad_z * w for w in self.weights]

        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, grad_weights)]
        self.bias -= learning_rate * grad_z

        return grad_input


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(weights=[random.uniform(-1, 1) for _ in range(n_inputs)],
                               bias=random.uniform(-1, 1))
                        for _ in range(n_neurons)]

    def backward(self, grad_outputs, learning_rate):
        grad_inputs = [0] * len(self.neurons[0].last_input)
        for neuron, grad in zip(self.neurons, grad_outputs):
            neuron_grads = neuron.backward(grad, learning_rate)
            grad_inputs = [g + ng for g, ng in zip(grad_inputs, neuron_grads)]
        return grad_inputs

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]



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

