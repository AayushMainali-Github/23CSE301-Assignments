from collections import Counter

import numpy as np


def missing(value):
    # None and nan both mean that a value is missing
    if value is None:
        return True

    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def central_value(values, method):
    # first remove all missing values from the column
    present = []

    for value in values:
        if not missing(value):
            present.append(value)

    if len(present) == 0:
        raise ValueError("column is fully missing")

    # choose the requested central tendency
    if method == "mean":
        return float(np.mean(present))

    if method == "median":
        return float(np.median(present))

    if method == "mode":
        counts = Counter(present)
        return max(counts, key=counts.get)

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
        # the replacement values must already be calculated
        if self.values is None:
            raise ValueError("fit first")

        X = np.asarray(X, dtype=object).copy()

        # fill missing values
        for row in range(X.shape[0]):
            for column in range(X.shape[1]):
                if missing(X[row, column]):
                    X[row, column] = self.values[column]

        # kNN needs a numeric matrix after imputation
        return X.astype(float)

    def fit_transform(self, X):
        # fit and transform are common preprocessing steps together
        self.fit(X)
        return self.transform(X)


def impute_matrix(X, method):
    # short helper for using the imputer on one matrix
    imputer = SimpleImputer(method)
    return imputer.fit_transform(X)
