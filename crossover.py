import numpy as np


# implementacja jednopunktowego krzyżowania
def one_point_crossover(p1, p2):
    point = np.random.randint(1, len(p1))

    child1 = np.concatenate([
        p1[:point],
        p2[point:]
    ])

    child2 = np.concatenate([
        p2[:point],
        p1[point:]
    ])

    return child1, child2


# dwupunktowe krzyżowanie chromosomów
def two_point_crossover(p1, p2):
    point1, point2 = sorted(
        np.random.choice(
            range(1, len(p1)),
            2,
            replace=False
        )
    )

    child1 = np.concatenate([
        p1[:point1],
        p2[point1:point2],
        p1[point2:]
    ])

    child2 = np.concatenate([
        p2[:point1],
        p1[point1:point2],
        p2[point2:]
    ])

    return child1, child2


# krzyżowanie arytmetyczne
def arithmetic_crossover(p1, p2):
    alpha = np.random.rand()

    child1 = alpha * p1 + (1 - alpha) * p2

    child2 = alpha * p2 + (1 - alpha) * p1

    return child1, child2


def order_crossover(p1, p2):

    size = len(p1)

    start, end = sorted(
        np.random.choice(
            range(size),
            2,
            replace=False
        )
    )

    child1 = [-1] * size
    child2 = [-1] * size

    # kopiowanie fragmentu
    child1[start:end] = p1[start:end]
    child2[start:end] = p2[start:end]

    # uzupełnianie
    fill1 = [x for x in p2 if x not in child1]
    fill2 = [x for x in p1 if x not in child2]

    ptr1 = ptr2 = 0

    for i in range(size):

        if child1[i] == -1:
            child1[i] = fill1[ptr1]
            ptr1 += 1

        if child2[i] == -1:
            child2[i] = fill2[ptr2]
            ptr2 += 1

    return np.array(child1), np.array(child2)



