"""AI-assisted K-means. GenAI tool: ChatGPT (GPT-5.6 Sol)."""
import numpy as np

def calculate_inertia(points, labels, centroids):
    x = np.asarray(points, dtype=float); labels = np.asarray(labels, dtype=int); c = np.asarray(centroids, dtype=float)
    return float(np.sum((x - c[labels])**2))

def kmeans_ai(points, k, max_iter=100, tolerance=1e-6):
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("points must be a non-empty 2D matrix.")
    if k <= 0 or k > len(x):
        raise ValueError("Invalid k.")
    if max_iter <= 0 or tolerance < 0:
        raise ValueError("Invalid iteration/tolerance parameter.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Data contains NaN or infinite values.")
    centroids = x[:k].copy()
    labels = np.zeros(len(x), dtype=int)
    iterations = 0
    for iteration in range(1, max_iter+1):
        iterations = iteration
        distances = np.linalg.norm(x[:,None,:] - centroids[None,:,:], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)
        movement = np.max(np.linalg.norm(new_centroids-centroids, axis=1))
        centroids = new_centroids
        if movement <= tolerance:
            break
    return {"labels": labels.tolist(), "centroids": centroids.tolist(), "iterations": iterations,
            "inertia": calculate_inertia(x, labels, centroids)}
