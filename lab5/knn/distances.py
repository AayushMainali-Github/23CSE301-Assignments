import numpy as np


def check_vectors(a, b):
    # turn both inputs into numeric one dimensional arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # distance only makes sense for equal length vectors
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        raise ValueError("vectors must have the same length")

    return a, b


def minkowski_distance(a, b, p):
    # Minkowski distance can become Euclidean or Manhattan
    a, b = check_vectors(a, b)

    if p <= 0:
        raise ValueError("p must be positive")

    # calculate the distance feature by feature
    return float(np.sum(np.abs(a - b) ** p) ** (1 / p))


def euclidean_distance(a, b):
    # Euclidean distance is Minkowski distance with p equal to 2
    return minkowski_distance(a, b, 2)


def manhattan_distance(a, b):
    # Manhattan distance is Minkowski distance with p equal to 1
    return minkowski_distance(a, b, 1)


def cosine_distance(a, b):
    # cosine distance is one minus cosine similarity
    a, b = check_vectors(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    # handle zero vectors separately so there is no division by zero
    if denominator == 0:
        if np.array_equal(a, b):
            return 0.0
        return 1.0

    return float(1 - np.dot(a, b) / denominator)


def distance(a, b, metric, p):
    # choose the distance function from the configuration
    if metric == "euclidean":
        return euclidean_distance(a, b)

    if metric == "manhattan":
        return manhattan_distance(a, b)

    if metric == "minkowski":
        return minkowski_distance(a, b, p)

    if metric == "cosine":
        return cosine_distance(a, b)

    raise ValueError("unknown metric")
