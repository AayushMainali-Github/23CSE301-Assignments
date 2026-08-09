"""Vector operations. GenAI-assisted: ChatGPT (GPT-5.6 Sol)."""
def dot_product(a, b):
    a, b = list(a), list(b)
    if not a or not b:
        raise ValueError("Vectors cannot be empty.")
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")
    return sum(float(x)*float(y) for x, y in zip(a, b))

def euclidean_norm(vector):
    vector = list(vector)
    if not vector:
        raise ValueError("Vector cannot be empty.")
    return sum(float(x)**2 for x in vector) ** 0.5
