import numpy as np


# implementacja metody selekcji
def tournament_selection(population, fitness, tournament_size):

    indices = np.random.choice(
        len(population),
        tournament_size,
        replace=False
    )

    tournament_fitness = fitness[indices]

    winner_idx = indices[np.argmin(tournament_fitness)]

    return population[winner_idx]


# implementacja metody koła ruletki
def roulette_selection(population, fitness):

    probabilities = 1 / (fitness + 1e-10)

    probabilities /= probabilities.sum()

    idx = np.random.choice(
        len(population),
        p=probabilities
    )

    return population[idx]


# implementacja metody selekcji rankingowej
def ranking_selection(population, fitness):

    ranks = np.argsort(np.argsort(fitness))

    probabilities = len(fitness) - ranks
    probabilities = probabilities / probabilities.sum()

    idx = np.random.choice(
        len(population),
        p=probabilities
    )

    return population[idx]
