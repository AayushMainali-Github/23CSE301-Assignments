import numpy as np

from .neighbors import find_neighbors, vote


class KNNClassifier:
    def __init__(self, k, metric, p, sorting, weights):
        # check the main settings before storing them
        if k < 1:
            raise ValueError("k must be positive")

        if weights not in ["uniform", "distance"]:
            raise ValueError("invalid weights")

        self.k = k
        self.metric = metric
        self.p = p
        self.sorting = sorting
        self.weights = weights
        self.X = None
        self.y = None

    def fit(self, X, y):
        # kNN does not learn weights like a linear model
        # it mainly stores the training examples for later comparison
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # check that the input looks like a training matrix
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a matrix")

        if len(X) != len(y) or self.k > len(X):
            raise ValueError("invalid X, y, or k")

        self.X = X
        self.y = y
        return self

    def get_neighbors(self, row):
        # find the closest training rows for one test row
        if self.X is None:
            raise ValueError("fit first")

        return find_neighbors(
            self.X, self.y, row, self.k,
            self.metric, self.p, self.sorting)

    def predict(self, X):
        # make one prediction for every input row
        X = np.asarray(X, dtype=float)

        # allow a single vector as input too
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.X is None or X.shape[1] != self.X.shape[1]:
            raise ValueError("wrong input")

        weighted = self.weights == "distance"
        prediction = []

        # classify rows one at a time
        for row in X:
            neighbors = self.get_neighbors(row)
            prediction.append(vote(neighbors, weighted))

        return np.asarray(prediction)

    def score(self, X, y):
        # accuracy is the number of correct predictions divided by all predictions
        prediction = self.predict(X)
        return float(np.mean(prediction == np.asarray(y)))

    def explain(self, row, feature_names):
        # return the neighbors and the features that differ most from them
        row = np.asarray(row, dtype=float)
        result = []

        # explain every neighbor used for this prediction
        for neighbor in self.get_neighbors(row):
            # compare this row with the selected training row
            differences = np.abs(row - self.X[neighbor["index"]])
            order = np.argsort(differences)[::-1][:5]
            names = []

            # get the names of the five largest differences
            for i in order:
                names.append(feature_names[i])

            # store the difference beside its feature name
            largest = {}
            for name, i in zip(names, order):
                largest[name] = float(differences[i])

            result.append({
                "index": int(neighbor["index"]),
                "label": str(neighbor["label"]),
                "distance": float(neighbor["distance"]),
                "largest_feature_differences": largest,
            })

        return result


class WeightedKNNClassifier(KNNClassifier):
    def __init__(self, k, metric, p, sorting):
        # use the same classifier with distance-based voting
        super().__init__(k, metric, p, sorting, "distance")
