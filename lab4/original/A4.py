def minkowski_distance(a, b, p):
    n = len(a)
    total = 0
    for i in range(n):
        total = total + abs(a[i] - b[i]) ** p
    return total ** (1 / p)


if __name__ == "__main__":
    x = [1, 2, 3]
    y = [4, 5, 6]
    print("Manhattan (p=1):", minkowski_distance(x, y, 1))
    print("Euclidean (p=2):", minkowski_distance(x, y, 2))
