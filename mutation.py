import numpy as np


# -------------FUNKCJE CIĄGŁE---------------------#

# implementacja mutacji równomiernej
def uniform_mutation(individual, mutation_prob, bounds):
    low, high = bounds

    for i in range(len(individual)):

        if np.random.rand() < mutation_prob:
            individual[i] = np.random.uniform(low, high)

    return individual


# implementacja mutacji gaussowskiej
def gaussian_mutation(individual, mutation_prob, bounds, sigma=0.1):
    low, high = bounds

    for i in range(len(individual)):

        if np.random.rand() < mutation_prob:
            individual[i] += np.random.normal(0, sigma)

            individual[i] = np.clip(
                individual[i],
                low,
                high
            )

    return individual


# -------------MUTACJA TSP---------------------#

# implementacja mutacji typu swap
def swap_mutation(individual, mutation_prob):
    if np.random.rand() < mutation_prob:
        i, j = np.random.choice(
            len(individual),
            2,
            replace=False
        )

        individual[i], individual[j] = (
            individual[j],
            individual[i]
        )

    return individual


# implementacja mutacji odwrócenia
def inverse_mutation(individual, mutation_prob):

    if np.random.rand() < mutation_prob:

        i, j = sorted(
            np.random.choice(
                len(individual),
                2,
                replace=False
            )
        )

        individual[i:j] = individual[i:j][::-1]

    return individual
