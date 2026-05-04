import numpy as np


def sphere_function(x):
    return np.sum(x ** 2)


def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin_function(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley_function(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) \
        - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e


