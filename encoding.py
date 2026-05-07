import numpy as np


class Encoding:
    def initialize(self, population_size):
        pass

    @staticmethod
    def repair(individual):
        return individual


class RealEncoding(Encoding):
    def __init__(self, dimension, bounds):
        self.dimensions = dimension
        self.bounds = bounds

    def initialize(self, population_size):
        low, high = self.bounds
        return np.random.uniform(low, high,
                                 (population_size, self.dimensions))


class PermutationEncoding(Encoding):
    def __init__(self, n):
        self.n = n

    def initialize(self, population_size):
        return np.array([
            np.random.permutation(self.n)
            for _ in range(population_size)
        ])

