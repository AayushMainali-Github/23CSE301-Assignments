"""dmtoolkit: compact educational data-mining utilities for 23CSE301 Lab 04."""
from .preprocessing import label_encode, one_hot_encode, encode_dataframe
from .distances import minkowski_distance, manhattan_distance, euclidean_distance
from .vector_ops import dot_product, euclidean_norm
from .statistics import mean, variance, standard_deviation, dataset_statistics
from .visualization import plot_minkowski, plot_histogram, plot_clusters
from .kmeans import kmeans_ai, calculate_inertia

__version__ = "1.0.0"
__all__ = [
    "label_encode", "one_hot_encode", "encode_dataframe",
    "minkowski_distance", "manhattan_distance", "euclidean_distance",
    "dot_product", "euclidean_norm",
    "mean", "variance", "standard_deviation", "dataset_statistics",
    "plot_minkowski", "plot_histogram", "plot_clusters",
    "kmeans_ai", "calculate_inertia",
]
