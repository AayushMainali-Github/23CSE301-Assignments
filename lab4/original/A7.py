import numpy as np


def my_dot(a, b):
    total = 0
    for i in range(len(a)):
        total = total + a[i] * b[i]
    return total


def my_norm(a):
    total = 0
    for i in range(len(a)):
        total = total + a[i] ** 2
    return total ** 0.5


def main():
    A = [2, 3, 4, 5]
    B = [1, 0, 2, 3]

    print("My dot product:", my_dot(A, B))
    print("numpy.dot:", np.dot(A, B))

    print("My length of A:", my_norm(A))
    print("numpy.linalg.norm A:", np.linalg.norm(A))

    print("My length of B:", my_norm(B))
    print("numpy.linalg.norm B:", np.linalg.norm(B))


if __name__ == "__main__":
    main()
