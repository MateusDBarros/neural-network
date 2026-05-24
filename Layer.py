import random

from Neuron import Neuron


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(weights=[random.uniform(-1, 1) for _ in range(n_inputs)],
                               bias=random.uniform(-1, 1))
                        for _ in range(n_neurons)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]
