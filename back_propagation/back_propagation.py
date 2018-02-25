from neuron.utils import sigmoid_derivative


def get_error(deltas, sums, weights):
    """
    Compute error on the previous layer of network
    :param deltas: ndarray of shape (n, n_{l+1})
    :param sums: ndarray of shape (n, n_l)
    :param weights: ndarray of shape (n_{l+1}, n_l)
    :return:
    """
    return (deltas.dot(weights) * sigmoid_derivative(sums)).mean(axis=0)
