import numpy as np


def sigmoid(x):
    """
    Sigmoid function. Works with numbers and with vectors.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid function. Works with numbers and with vectors.
    """
    return sigmoid(x) * (1 - sigmoid(x))
