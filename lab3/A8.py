from A3 import convert
import math


def my_mean(values):
    n = len(values)
    total = 0
    for x in values:
        total = total + x
    return total / n


def my_variance(values):
    m = my_mean(values)
    n = len(values)
    total = 0
    for x in values:
        total = total + (x - m) ** 2
    return total / n


def my_std(values):
    return math.sqrt(my_variance(values))


def dataset_stats(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])

    means = []
    stds = []

    for j in range(n_cols):
        col = []
        for i in range(n_rows):
            col.append(matrix[i][j])
        means.append(my_mean(col))
        stds.append(my_std(col))

    return means, stds


def main():
    data = convert()
    data = data.dropna()

    matrix = data.values.tolist()
    means, stds = dataset_stats(matrix)

    print("Feature means (first 5):", means[:5])
    print("Feature stds  (first 5):", stds[:5])
    return means, stds, data


if __name__ == "__main__":
    main()
