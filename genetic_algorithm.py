import numpy as np
from abc import ABC, abstractmethod
from utils import Encoding, RealEncoding, PermutationEncoding


class BaseAG(ABC):
    def __init__(self, population_size, fittness_func, crossover_prob, mutation_prob):
        self.population_size = population_size
        self.fitness_func = fittness_func
        self.crossover_prov = crossover_prob
        self.mutation_prob = mutation_prob

    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def crossover(self, p1, p2):
        pass

    @abstractmethod
    def mutation(self, index):
        pass




