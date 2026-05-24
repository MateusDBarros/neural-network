from Layer import Layer


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


