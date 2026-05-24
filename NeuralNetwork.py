import math
import random


class Neuron:

    def __init__(self, bias, weigths, inputs):
        self.bias = bias
        self.weights = weigths
        self.inputs = inputs
        self.output = 0

    def forward(self, inputs):
        weighted_sum =  sum(w *  x for w, x in zip(self.weights, inputs))
        return self.relu(weighted_sum + self.bias)

    def relu(self, x):
        return max(0, x)

    def sigmod(self, x):
        return 1 / (1 + math.exp(-x))


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



class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(weights=[random.uniform(-1, 1) for _ in range(n_inputs)],
                               bias=random.uniform(-1, 1))
                        for _ in range(n_neurons)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]
