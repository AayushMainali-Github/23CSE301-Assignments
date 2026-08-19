def minkowski_distance(a, b, p):
    n = len(a)
    total = 0
    # Calculate total as Summation (ai - bi)^p
    for i in range(n):
        total = total + abs(a[i] - b[i]) ** p

    # At the end return the pth root
    return total ** (1 / p)
