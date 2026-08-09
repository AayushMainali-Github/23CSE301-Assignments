"""Visualization helpers. GenAI-assisted: ChatGPT (GPT-5.6 Sol)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _save(fig, save_path):
    if save_path:
        p = Path(save_path); p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, bbox_inches="tight")

def plot_minkowski(p_values, distances, save_path=None, show=False):
    fig, ax = plt.subplots()
    ax.plot(list(p_values), list(distances), marker="o")
    ax.set_xlabel("p (order)"); ax.set_ylabel("Minkowski distance"); ax.set_title("Minkowski distance for p = 1 to 10"); ax.grid(True)
    _save(fig, save_path)
    if show: plt.show()
    return fig, ax

def plot_histogram(values, bins=20, feature_name="Feature", save_path=None, show=False):
    fig, ax = plt.subplots()
    ax.hist(list(values), bins=bins)
    ax.set_xlabel(feature_name); ax.set_ylabel("Frequency"); ax.set_title(f"{feature_name} histogram")
    _save(fig, save_path)
    if show: plt.show()
    return fig, ax

def plot_clusters(points, labels, centroids=None, feature_names=None, save_path=None, show=False):
    points = np.asarray(points, dtype=float); labels = np.asarray(labels)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("At least two features are required.")
    names = feature_names or ("Feature 1", "Feature 2")
    fig, ax = plt.subplots()
    sc = ax.scatter(points[:,0], points[:,1], c=labels, s=14, alpha=0.75)
    if centroids is not None:
        c = np.asarray(centroids, dtype=float)
        ax.scatter(c[:,0], c[:,1], marker="X", s=150, label="Centroids")
        ax.legend()
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_title("AI-assisted K-means clusters")
    fig.colorbar(sc, ax=ax, label="Cluster")
    _save(fig, save_path)
    if show: plt.show()
    return fig, ax
