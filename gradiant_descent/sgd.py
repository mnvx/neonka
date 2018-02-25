import numpy as np

from neuron.neuron import Neuron


def SGD(neuron: Neuron, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
    """
    Внешний цикл алгоритма градиентного спуска.
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)

    learning_rate - константа скорости обучения
    batch_size - размер батча, на основании которого
    рассчитывается градиент и совершается один шаг алгоритма

    eps - критерий остановки номер один: если разница между значением целевой функции
    до и после обновления весов меньше eps - алгоритм останавливается.
    Вторым вариантом была бы проверка размера градиента, а не изменение функции,
    что будет работать лучше - неочевидно. В заданиях используйте первый подход.

    max_steps - критерий остановки номер два: если количество обновлений весов
    достигло max_steps, то алгоритм останавливается

    Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся)
    и 0, если второй (спуск не достиг минимума за отведённое время).
    """
    steps = 0
    while True:
        indexes = np.arange(len(X))
        np.random.shuffle(indexes)

        success = True
        shift = 0
        while shift * batch_size < len(indexes):
            batch_idx = indexes[(shift * batch_size):(shift * batch_size + batch_size)]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            res = update_mini_batch(neuron, X_batch, y_batch, learning_rate, eps)
            if not res:
                success = False
            shift += batch_size
        if success:
            return 1

        steps += 1
        if steps > max_steps:
            return 0


def update_mini_batch(neuron: Neuron, X, y, learning_rate, eps):
    """
    X - матрица размера (batch_size, m)
    y - вектор правильных ответов размера (batch_size, 1)
    learning_rate - константа скорости обучения
    eps - критерий остановки номер один: если разница между значением целевой функции
    до и после обновления весов меньше eps - алгоритм останавливается.

    Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции)
    и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1,
    иначе возвращаем 0.
    """
    grad = compute_grad_analytically(neuron, X, y)
    J_old = J_quadratic(neuron, X, y)
    neuron.w -= learning_rate * grad
    J_new = J_quadratic(neuron, X, y)
    return abs(J_old - J_new) < eps


def J_quadratic(neuron: Neuron, X, y):
    """
    Оценивает значение квадратичной целевой функции.
    Всё как в лекции, никаких хитростей.

    neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)

    Возвращает значение J (число)
    """

    assert y.shape[1] == 1, 'Incorrect y shape'

    return 0.5 * np.mean((neuron.forward_pass(X) - y) ** 2)


def J_quadratic_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
    y_hat - вертикальный вектор предсказаний,
    y - вертикальный вектор правильных ответов,

    В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать
    с целевыми функциями - полезно вынести эти вычисления в отдельный этап.

    Возвращает вектор значений производной целевой функции для каждого примера отдельно.
    """

    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'

    return (y_hat - y) / len(y)


def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
    """
    Аналитическая производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам

    Возвращает вектор размера (m, 1)
    """

    # Вычисляем активации
    # z - вектор результатов сумматорной функции нейрона на разных примерах

    z = neuron.summatory(X)
    y_hat = neuron.activation(z)

    # Вычисляем нужные нам частные производные
    dy_dyhat = J_prime(y, y_hat)
    dyhat_dz = neuron.activation_function_derivative(z)

    # осознайте эту строчку:
    dz_dw = X

    # а главное, эту:
    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)

    # Сделаем из горизонтального вектора вертикальный
    grad = grad.T

    return grad


def compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для тестовой выборки X
    J - целевая функция, градиент которой мы хотим получить
    eps - размер $\delta w$ (малого изменения весов)
    """

    initial_cost = J(neuron, X, y)
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

    for i in range(len(w_0)):
        old_wi = neuron.w[i].copy()
        # Меняем вес
        neuron.w[i] += eps

        # Считаем новое значение целевой функции и вычисляем приближенное значение градиента
        num_grad[i] = (J(neuron, X, y) - initial_cost) / eps

        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi

    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad


def compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для тестовой выборки X
    J - целевая функция, градиент которой мы хотим получить
    eps - размер $\delta w$ (малого изменения весов)
    """
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

    for i in range(len(w_0)):
        old_wi = neuron.w[i].copy()

        neuron.w[i] -= eps
        prev_cost = J(neuron, X, y)
        neuron.w[i] = old_wi

        # Меняем вес
        neuron.w[i] += eps
        next_cost = J(neuron, X, y)

        # Считаем новое значение целевой функции и вычисляем приближенное значение градиента
        num_grad[i] = (next_cost - prev_cost) / (2 * eps)

        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi

    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad
