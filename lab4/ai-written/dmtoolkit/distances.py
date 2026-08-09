"""Distance functions. GenAI-assisted: ChatGPT (GPT-5.6 Sol)."""
def _prepare(a, b):
    a, b = list(a), list(b)
    if not a or not b:
        raise ValueError("Vectors cannot be empty.")
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")
    return [float(x) for x in a], [float(x) for x in b]

def minkowski_distance(a, b, p=2):
    if p <= 0:
        raise ValueError("p must be greater than zero.")
    a, b = _prepare(a, b)
    return sum(abs(x-y)**p for x, y in zip(a, b)) ** (1.0/p)

def manhattan_distance(a, b):
    return minkowski_distance(a, b, 1)

def euclidean_distance(a, b):
    return minkowski_distance(a, b, 2)
