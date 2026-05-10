import numpy as np


class BaseAG:

    def __init__(
            self,
            population_size,
            fitness_func,
            crossover_prob,
            mutation_prob,
            selection_method,
            crossover_method,
            mutation_method,
            encoding
    ):

        # parametry podstawowe
        self.population_size = population_size

        # funkcja fitness
        self.fitness_func = fitness_func

        # prawdopodobieństwa operatorów
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # operatory AG
        self.selection = selection_method
        self.crossover = crossover_method
        self.mutation = mutation_method

        # sposób enkodowania
        self.encoding = encoding

        # historia optymalizacji
        self.history = []

        # najlepsze rozwiązanie globalne
        self.best_fitness = np.inf
        self.best_individual = None

    # -----------------------------------------
    # inicjalizacja populacji
    # -----------------------------------------
    def initialize_population(self):

        return self.encoding.initialize(
            self.population_size
        )

    # -----------------------------------------
    # ocena populacji
    # -----------------------------------------
    def evaluate(self, population):

        return np.array([
            self.fitness_func(individual)
            for individual in population
        ])

    # -----------------------------------------
    # główna pętla algorytmu genetycznego
    # -----------------------------------------
    def run(
            self,
            generations,
            patience=20
    ):

        # inicjalizacja populacji
        population = self.initialize_population()

        # reset historii
        self.history = []

        # reset najlepszego rozwiązania
        self.best_fitness = np.inf
        self.best_individual = None

        # licznik stagnacji
        no_improvement = 0

        # -----------------------------------------
        # pętla ewolucji
        # -----------------------------------------
        for generation in range(generations):

            # ocena populacji
            fitness = self.evaluate(population)

            # najlepszy osobnik w aktualnej generacji
            current_best_idx = np.argmin(fitness)

            current_best_fitness = fitness[current_best_idx]

            current_best_individual = (
                population[current_best_idx].copy()
            )

            # zapis historii
            self.history.append(current_best_fitness)

            # -----------------------------------------
            # aktualizacja najlepszego rozwiązania
            # -----------------------------------------
            if current_best_fitness < self.best_fitness:

                self.best_fitness = current_best_fitness

                self.best_individual = (
                    current_best_individual
                )

                no_improvement = 0

            else:

                no_improvement += 1

            # -----------------------------------------
            # early stopping
            # -----------------------------------------
            if no_improvement >= patience:

                print(
                    f"Early stopping at generation {generation}"
                )

                break

            # -----------------------------------------
            # elityzm
            # -----------------------------------------
            elite = self.best_individual.copy()

            new_population = [elite]

            # -----------------------------------------
            # tworzenie nowej populacji
            # -----------------------------------------
            while len(new_population) < self.population_size:

                # selekcja rodziców
                p1 = self.selection(
                    population,
                    fitness
                )

                p2 = self.selection(
                    population,
                    fitness
                )

                # -----------------------------------------
                # crossover
                # -----------------------------------------
                if np.random.rand() < self.crossover_prob:

                    c1, c2 = self.crossover(
                        p1,
                        p2
                    )

                else:

                    c1 = p1.copy()
                    c2 = p2.copy()

                # -----------------------------------------
                # mutacja
                # -----------------------------------------
                c1 = self.mutation(c1)

                c2 = self.mutation(c2)

                # dodanie dzieci do nowej populacji
                new_population.extend([c1, c2])

            # ograniczenie populacji
            population = np.array(
                new_population[:self.population_size]
            )

        # -----------------------------------------
        # zwrot wyników
        # -----------------------------------------
        return (
            self.best_individual,
            self.best_fitness,
            self.history
        )