import numpy as np

from .imputation import SimpleImputer


class FeaturePreprocessor:
    def __init__(self, method):
        # keep the imputer and scaling values inside the same object
        self.imputer = SimpleImputer(method)
        self.mean = None
        self.std = None

    def fit(self, X):
        # calculate missing value replacements using training data
        X = self.imputer.fit_transform(X)

        # calculate the center and spread for every feature
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # constant columns should not cause division by zero
        self.std[self.std == 0] = 1
        return self

    def transform(self, X):
        # do not transform data before fit has been called
        if self.mean is None:
            raise ValueError("fit first")

        # use the training mean and standard deviation on new data
        X = self.imputer.transform(X)
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        # useful shortcut for the training matrix
        self.fit(X)
        return self.transform(X)
