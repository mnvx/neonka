import numpy as np

from .utils import sigmoid, sigmoid_derivative


class Neuron:
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.w)

    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_derivative):
        """
        :param weights: Vertical vector of neuron weights (m, 1), weights[0][0] - bias.
        :param activation_function: Neuron activation function. Sigmoid by defaults.
        :param activation_function_derivative: Derivative of activation function.
        """

        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def summatory(self, input_matrix):
        """
        Calculate result of summatory function for every example from input_matrix.
        :param input_matrix: Matrix of examples with shape (n, m). Every row is example.
        n - examples count, m - variables count.
        :return: Vector of summatory function values with shape(n, 1).
        """
        return input_matrix.dot(self.w)

    def activation(self, summatory_result):
        """
        Calculate result of activation function for every example from summatory_result.
        :param summatory_result: Result of summatory function, vector with shape (n, 1),
        summatory_result[i] is value of summatory function for i-th example.
        :return: Vector of activation function values with shape (n, 1),
        i-th row contains value of activation function for i-th example.
        """
        return self.activation_function(summatory_result)

    def forward_pass(self, input_matrix):
        """
        Calculate outputs of neuron
        :param input_matrix: Matrix of examples with shape (n, m).
        n - examples count. m - variables count.
        :return: Vertical vector (n, 1) with output activations of neuron.
        """
        return self.activation(self.summatory(input_matrix))

