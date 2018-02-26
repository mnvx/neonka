import numpy

from network.network_abstract import NetworkAbstract
from neuron.utils import sigmoid_derivative


def get_error(deltas: numpy.ndarray, sums: numpy.ndarray, weights: numpy.ndarray):
    """
    Compute error on the previous layer of network
    :param deltas: ndarray of shape (n, n_{l+1})
    :param sums: ndarray of shape (n, n_l)
    :param weights: ndarray of shape (n_{l+1}, n_l)
    :return:
    """
    return (deltas.dot(weights) * sigmoid_derivative(sums)).mean(axis=0)


def forward_pass(net: NetworkAbstract, examples_batch: numpy.ndarray, expected_results_batch: numpy.ndarray):
    activations = []  # will be (len(examples_batch), len(net.neurons), len(net.neurons[i]))
    for i, example in enumerate(examples_batch):
        activations.append(net.activations(example))
        # @todo: calculate deltas


def back_propagation(net: NetworkAbstract, examples: numpy.ndarray, expected_results: numpy.ndarray,
                     batch_size: int = None):
    """
    Train neuron network with back propagation algorithm
    :param net: Neuron network
    :param examples: Inputs for network for every example (n, m)
    :param expected_results: Vertical vector of right answers for every example (n, 1)
    :param batch_size: Between 1 and m
    :return:
    """
    if batch_size is None:
        batch_size = min(max(1, int(len(examples) / 10)), 100)

    indexes = numpy.arange(len(examples))
    numpy.random.shuffle(indexes)

    shift = 0
    while shift * batch_size < len(indexes):
        batch_idx = indexes[(shift * batch_size):(shift * batch_size + batch_size)]
        examples_batch = examples[batch_idx]
        expected_results_batch = expected_results[batch_idx]
        forward_pass(net, examples_batch, expected_results_batch)
        shift += batch_size
