import math

import numpy as np

from .imputation import SimpleImputer


def my_mean(values):
    total = 0

    for x in values:
        total = total + x

    return total / len(values)


def my_variance(values):
    m = my_mean(values)
    total = 0

    for x in values:
        total = total + (x - m) ** 2

    return total / len(values)


def my_std(values):
    return math.sqrt(my_variance(values))


def dataset_stats(matrix):
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    means = []
    stds = []

    for j in range(n_cols):
        column = []
        for i in range(n_rows):
            column.append(matrix[i][j])

        means.append(my_mean(column))
        stds.append(my_std(column))

    return means, stds


class FeaturePreprocessor:
    def __init__(self, method):
        self.imputer = SimpleImputer(method)
        self.mean = None
        self.std = None

    def fit(self, X):
        # calculate these values from the training data only
        X = self.imputer.fit_transform(X)
        self.mean, self.std = dataset_stats(X.tolist())

        for i in range(len(self.std)):
            if self.std[i] == 0:
                self.std[i] = 1

        return self

    def transform(self, X):
        if self.mean is None:
            raise ValueError("fit first")

        X = self.imputer.transform(X)
        result = []

        # standardize every value using the training statistics
        for row in X:
            newrow = []
            for i in range(len(row)):
                newrow.append((row[i] - self.mean[i]) / self.std[i])
            result.append(newrow)

        return np.asarray(result, dtype=float)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
