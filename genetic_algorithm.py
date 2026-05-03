import numpy as np
from abc import ABC, abstractmethod
from utils import Encoding, RealEncoding, PermutationEncoding


class BaseAG(ABC):
    def __init__(self, population_size, fittness_func, crossover_prob, mutation_prob, selection_method):
        self.population_size = population_size
        self.fitness_func = fittness_func
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection = selection_method

    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def crossover(self, p1, p2):
        pass

    @abstractmethod
    def mutation(self, index):
        pass

    def evaluate(self, population):
        return np.array([self.fitness_func(idx) for idx in population])

    def run(self, generations):
        population = self.initialize_population()
        history = []

        for _ in range(generations):
            fitness = self.evaluate(population)

            new_population = []

            while len(new_population) < self.population_size:
                p1 = self.selection(population, fitness)
                p2 = self.selection(population, fitness)

                if np.random.rand() < self.crossover_prob:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = self.mutation(c1)
                c2 = self.mutation(c2)

                new_population.extend([c1, c2])

            population = np.array(new_population[:self.population_size])
            history.append(np.min(fitness))

        return population, history


