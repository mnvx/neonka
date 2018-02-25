import math
import numpy as np

from neuron.neuron import Neuron
from .network_abstract import NetworkAbstract


class LayeredNetwork(NetworkAbstract):
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        result = ''
        for l, layer in enumerate(self.neurons):
            result += "Layer %s:\n" % (l + 1)
            for k, neuron in enumerate(layer):
                result += "Neuron %s:\n" % (k + 1)
                result += str(layer[k])
                result += "\n"
            result += "\n"
        return result

    def __init__(self, layers):
        """
        :param layers: List (length is L) contains neuron counts in every layer
        """
        self.neurons = []
        prev_count = 1
        for k, count in enumerate(layers):
            layer = []
            min_weight = -math.sqrt(prev_count) / 2
            max_weight = -min_weight

            if k == 0:
                # Input layer. Only biases.
                layer_weights = np.zeros((prev_count, count))
            elif k == len(layers) - 1:
                # Output layer. Weights and biases (+1).
                layer_weights = np.zeros((prev_count + 1, count))
            else:
                # Hidden layers. Weights and biases (+1).
                layer_weights = np.random.random((prev_count + 1, count)) * (max_weight - min_weight) + min_weight

            for i in range(count):
                neuron_weights = np.array([layer_weights[:, i]]).T
                layer.append(Neuron(neuron_weights))
            self.neurons.append(layer)
            prev_count = count

    def set_neuron(self, layer: int, number: int, neuron: Neuron):
        self.neurons[layer, number] = neuron
