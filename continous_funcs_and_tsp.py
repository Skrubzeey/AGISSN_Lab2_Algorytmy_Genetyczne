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


def get_tsp_distance(path, distance_matrix):
    # path: permutacja miast [0, 3, 1, 2]
    dist = 0
    for i in range(len(path) - 1):
        dist += distance_matrix[path[i], path[i + 1]]
    dist += distance_matrix[path[-1], path[0]]  # powrót
    return dist


def generate_tsp_data(n_cities, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100

    # Macierz odległości euklidesowych
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))
    return coords, dist_matrix


# zdefiniowanie funkcji przystosowania dla tsp
def create_tsp_fitness(distance_matrix):
    def fitness(route):
        dist = 0

        for i in range(len(route) - 1):
            dist += distance_matrix[route[i], route[i + 1]]

        dist += distance_matrix[route[-1], route[0]]

        return dist

    return fitness


def nearest_neighbor_route(distance_matrix):

    n = len(distance_matrix)

    start = np.random.randint(n)

    route = [start]

    unvisited = set(range(n))
    unvisited.remove(start)

    current = start

    while unvisited:

        next_city = min(
            unvisited,
            key=lambda city:
                distance_matrix[current][city]
        )

        route.append(next_city)

        unvisited.remove(next_city)

        current = next_city

    return np.array(route)
