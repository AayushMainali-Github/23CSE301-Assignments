import numpy as np


def missing(value):
    # None and nan both mean missing
    if value is None:
        return True

    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def mean(values):
    total = 0
    count = 0

    for x in values:
        if not missing(x):
            total = total + x
            count = count + 1

    if count == 0:
        raise ValueError("column is fully missing")

    return total / count


def median(values):
    validvalues = []

    for x in values:
        if not missing(x):
            validvalues.append(x)

    if len(validvalues) == 0:
        raise ValueError("column is fully missing")

    validvalues.sort()
    n = len(validvalues)
    middle = n // 2

    # for even length, use the average of the two middle values
    if n % 2 == 0:
        return (validvalues[middle - 1] + validvalues[middle]) / 2

    return validvalues[middle]


def mode(values):
    frequency = {}

    for x in values:
        if missing(x):
            continue

        if x not in frequency:
            frequency[x] = 0
        frequency[x] = frequency[x] + 1

    if len(frequency) == 0:
        raise ValueError("column is fully missing")

    best = list(frequency.keys())[0]
    for x in frequency:
        if frequency[x] > frequency[best]:
            best = x

    return best


def central_value(values, method):
    if method == "mean":
        return mean(values)

    if method == "median":
        return median(values)

    if method == "mode":
        return mode(values)

    raise ValueError("use mean, median, or mode")


class SimpleImputer:
    def __init__(self, method):
        self.method = method
        self.values = None

    def fit(self, X):
        # find one replacement value for every column
        X = np.asarray(X, dtype=object)
        self.values = []

        for column in range(X.shape[1]):
            self.values.append(
                central_value(X[:, column], self.method))

        return self

    def transform(self, X):
        if self.values is None:
            raise ValueError("fit first")

        X = np.asarray(X, dtype=object)
        result = []

        # replace missing values row by row
        for row in X:
            newrow = []
            for column in range(len(row)):
                value = row[column]
                if missing(value):
                    value = self.values[column]
                newrow.append(value)
            result.append(newrow)

        return np.asarray(result, dtype=float)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def impute_matrix(X, method):
    imputer = SimpleImputer(method)
    return imputer.fit_transform(X)
