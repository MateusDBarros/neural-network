import math

from openpyxl.styles.builtins import output


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


neuron = Neuron(weigths=[0.5, -0.3, 0.8], bias=0.1)
output = neuron.forward(inputs=[1.0, 2.0, 3.0])