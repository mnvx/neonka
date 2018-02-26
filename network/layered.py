import math
from typing import List

import numpy

from neuron.neuron import Neuron
from .network_abstract import NetworkAbstract


class LayeredNetwork(NetworkAbstract):
    def __init__(self, layers: List[int]):
        """
        :param layers: List (length is L) contains neuron counts in every layer
        """
        super().__init__()
        prev_count = 1
        for k, count in enumerate(layers):
            layer = []
            min_weight = -math.sqrt(prev_count) / 2
            max_weight = -min_weight

            if k == 0:
                # Input layer. Biases = 0. Weights = 1.
                layer_weights = numpy.zeros((prev_count + 1, count))
                layer_weights[1:, :] = 1
            elif k == len(layers) - 1:
                # Output layer. Weights and biases (+1).
                layer_weights = numpy.zeros((prev_count + 1, count))
            else:
                # Hidden layers. Weights and biases (+1).
                layer_weights = numpy.random.random((prev_count + 1, count)) * (max_weight - min_weight) + min_weight

            for i in range(count):
                neuron_weights = numpy.array([layer_weights[:, i]]).T
                layer.append(Neuron(neuron_weights))
            self.neurons.append(layer)
            prev_count = count

    def set_neuron(self, layer: int, number: int, neuron: Neuron):
        self.neurons[layer][number] = neuron

    def activations(self, example: numpy.ndarray):
        """
        :param example: Vector with one example
        :return:
        """
        activations = []
        layer_inputs = numpy.ones((len(example), 2))
        layer_inputs[:, 1:] = example[numpy.newaxis, :].T
        for l, layer_neurons in enumerate(self.neurons):
            layer_activations = numpy.zeros((len(layer_neurons), 1))
            for k, neuron in enumerate(layer_neurons):
                neuron_inputs = numpy.array([layer_inputs[k]])
                layer_activations[k] = neuron.forward_pass(neuron_inputs)[0]
            activations.append(layer_activations)
            if l < len(self.neurons) - 1:
                layer_inputs = numpy.ones((len(self.neurons[l + 1]), len(layer_neurons) + 1))
                layer_inputs[:, 1:] = layer_activations.T
        return activations
