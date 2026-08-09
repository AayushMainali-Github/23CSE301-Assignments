from A3 import convert
from A4 import minkowski_distance
from A8 import my_mean


def closest_centroid(point, centroids):
    best = 0
    best_dist = minkowski_distance(point, centroids[0], 2)
    for i in range(1, len(centroids)):
        d = minkowski_distance(point, centroids[i], 2)
        if d < best_dist:
            best_dist = d
            best = i
    return best


def recompute_centroids(points, labels, k):
    centroids = []
    for c in range(k):
        cluster = []
        for i in range(len(points)):
            if labels[i] == c:
                cluster.append(points[i])
        if len(cluster) == 0:
            centroids.append(points[c])
            continue
        n_feat = len(cluster[0])
        new_c = []
        for j in range(n_feat):
            col = [row[j] for row in cluster]
            new_c.append(my_mean(col))
        centroids.append(new_c)
    return centroids


def kmeans(points, k, max_iter=100):
    centroids = [points[i] for i in range(k)]

    for _ in range(max_iter):
        labels = []
        for p in points:
            labels.append(closest_centroid(p, centroids))

        new_centroids = recompute_centroids(points, labels, k)

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return labels, centroids


def main():
    data = convert()
    data = data.dropna()

    cols = ["Income", "Recency", "MntWines", "NumWebPurchases"]
    points = data[cols].values.tolist()

    labels, centroids = kmeans(points, k=3)

    print("final centroids:")
    for i in range(len(centroids)):
        print(i, centroids[i])

    counts = [0, 0, 0]
    for lab in labels:
        counts[lab] = counts[lab] + 1
    print("cluster sizes:", counts)


if __name__ == "__main__":
    main()
