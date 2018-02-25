import numpy as np

from back_propagation import back_propagation
from gradiant_descent.sgd import compute_grad_numerically, compute_grad_analytically, J_quadratic_derivative, \
    J_quadratic
from neuron.neuron import Neuron
from neuron.utils import sigmoid, sigmoid_derivative

data = np.loadtxt("data.csv", delimiter=",")
pears = data[:, 2] == 1
apples = np.logical_not(pears)

np.random.seed(1)

deltas = np.array(
    [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
    ]
)

sums = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)

weights = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [4, 5, 6],
    ]
)

print(back_propagation.get_error(deltas, sums, weights))



# Подготовим данные

X = data[:, :-1]
y = data[:, -1]

X = np.hstack((np.ones((len(y), 1)), X))
y = y.reshape((len(y), 1))  # Обратите внимание на эту очень противную и важную строчку


# Создадим нейрон

w = np.random.random((X.shape[1], 1))
neuron = Neuron(w, activation_function=sigmoid, activation_function_derivative=sigmoid_derivative)

# Посчитаем пример
num_grad = compute_grad_numerically(neuron, X, y, J=J_quadratic)
an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)

print("Численный градиент: \n", num_grad)
print("Аналитический градиент: \n", an_grad)

