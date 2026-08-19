import numpy as np

from lab5.knn.classifier import KNNClassifier, WeightedKNNClassifier
from lab5.knn.imputation import impute_matrix
from lab5.knn.sorting import bubble_sort, insertion_sort, merge_sort


def test_three_sorters_have_the_same_tie_order():
    # equal distances should still use the smaller training index first
    values = [(2.0, 2), (1.0, 4), (1.0, 1), (3.0, 0)]
    expected = [(1.0, 1), (1.0, 4), (2.0, 2), (3.0, 0)]
    assert bubble_sort(values) == expected
    assert insertion_sort(values) == expected
    assert merge_sort(values) == expected


def test_imputation_strategies():
    # check that missing values are replaced correctly
    values = [[1.0, 10.0], [3.0, np.nan], [None, 30.0]]
    assert np.allclose(impute_matrix(values, "mean"), [[1, 10], [3, 20], [2, 30]])
    assert np.allclose(impute_matrix(values, "median"), [[1, 10], [3, 20], [2, 30]])


def test_custom_knn_fit_predict_score():
    # simple data where the two classes are far apart
    X = [[0], [1], [10], [11]]
    y = ["bonafide", "bonafide", "spoof", "spoof"]
    model = KNNClassifier(
        3, "euclidean", 2, "merge", "uniform").fit(X, y)
    assert model.predict([[0.2], [10.2]]).tolist() == ["bonafide", "spoof"]
    assert model.score(X, y) == 1.0


def test_weighted_knn_prefers_close_neighbor():
    # the closest point should have the strongest vote
    X = [[0.0], [1.0], [10.0]]
    y = ["spoof", "bonafide", "bonafide"]
    model = WeightedKNNClassifier(
        3, "euclidean", 2, "merge").fit(X, y)
    assert model.predict([[0.1]]).tolist() == ["spoof"]
