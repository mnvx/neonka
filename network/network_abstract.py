from abc import abstractmethod

import numpy


class NetworkAbstract:

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

    def __init__(self):
        self.neurons = []

    @abstractmethod
    def activations(self, example: numpy.ndarray):
        """
        Calculate activations according to network topology
        :param example:
        :return:
        """
        pass
