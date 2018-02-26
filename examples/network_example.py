import numpy

from back_propagation.back_propagation import back_propagation
from network.layered import LayeredNetwork
from neuron.neuron import Neuron


def activation(x):
    return max(x, 0)


def activation_derivative(x):
    return 1 if x > 0 else 0


net = LayeredNetwork([3, 2, 1])
# print(net)

# Custom weights
net.set_neuron(0, 0, Neuron(numpy.array([[0, 1]]).T))
net.set_neuron(0, 1, Neuron(numpy.array([[0, 1]]).T))
net.set_neuron(0, 2, Neuron(numpy.array([[0, 1]]).T))

net.set_neuron(1, 0, Neuron(
    numpy.array([[0, 0.7, 0.2, 0.7]]).T,
    activation_function=activation,
    activation_function_derivative=activation_derivative,
))
net.set_neuron(1, 1, Neuron(numpy.array([[0, 0.8, 0.3, 0.6]]).T))

net.set_neuron(2, 0, Neuron(numpy.array([[0, 0.2, 0.4]]).T))
# print(net)

examples = numpy.array([[0, 1, 1]])
print("examples")
print(examples)

expected_results = numpy.array([[1]]).T
print("expected_results")
print(expected_results)
print('')

back_propagation(net, examples, expected_results)
# print(net)
