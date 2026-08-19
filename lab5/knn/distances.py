def check_vectors(a, b):
    a = list(a)
    b = list(b)

    if len(a) != len(b):
        raise ValueError("vectors must have the same length")

    return a, b


def minkowski_distance(a, b, p):
    a, b = check_vectors(a, b)

    if p <= 0:
        raise ValueError("p must be positive")

    total = 0

    # calculate the total for every feature
    for i in range(len(a)):
        total = total + abs(a[i] - b[i]) ** p

    # return the pth root
    return total ** (1 / p)


def euclidean_distance(a, b):
    return minkowski_distance(a, b, 2)


def manhattan_distance(a, b):
    return minkowski_distance(a, b, 1)


def cosine_distance(a, b):
    a, b = check_vectors(a, b)
    dot = 0
    length_a = 0
    length_b = 0

    for i in range(len(a)):
        dot = dot + a[i] * b[i]
        length_a = length_a + a[i] ** 2
        length_b = length_b + b[i] ** 2

    denominator = length_a ** 0.5 * length_b ** 0.5

    if denominator == 0:
        if a == b:
            return 0
        return 1

    return 1 - dot / denominator


def distance(a, b, metric, p):
    if metric == "euclidean":
        return euclidean_distance(a, b)

    if metric == "manhattan":
        return manhattan_distance(a, b)

    if metric == "minkowski":
        return minkowski_distance(a, b, p)

    if metric == "cosine":
        return cosine_distance(a, b)

    raise ValueError("unknown metric")
