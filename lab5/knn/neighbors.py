import math

from .distances import distance
from .sorting import sort_distances


def find_neighbors(X, y, row, k, metric, p, sorting):
    # k must be possible with the current training set
    if k < 1 or k > len(X):
        raise ValueError("invalid k")

    values = []

    # calculate all distances
    # keep the index so equal distances can be tied consistently
    for i in range(len(X)):
        value = distance(row, X[i], metric, p)
        values.append((value, i))

    # sort by distance and training index
    values = sort_distances(values, sorting)
    values = values[:k]

    # make the selected neighbors easier to use later
    neighbors = []
    for value, index in values:
        neighbors.append({
            "distance": value,
            "index": index,
            "label": y[index],
        })

    return neighbors


def vote(neighbors, weighted):
    # store the vote total, closest distance, and first position for every class
    scores = {}
    closest = {}
    first = {}

    for position in range(len(neighbors)):
        neighbor = neighbors[position]
        label = neighbor["label"]
        weight = 1.0

        # close neighbors get more influence in weighted kNN
        if weighted:
            weight = 1.0 / max(neighbor["distance"], 1e-12)

        scores[label] = scores.get(label, 0.0) + weight
        closest[label] = min(
            closest.get(label, math.inf), neighbor["distance"])

        if label not in first:
            first[label] = position

    # find the largest vote total
    best = max(scores.values())
    tied = []

    for label in scores:
        if math.isclose(scores[label], best):
            tied.append(label)

    # this is the first tie-breaking choice
    best_label = tied[0]

    # if the vote is tied, use the closest neighbor
    for label in tied[1:]:
        if closest[label] < closest[best_label]:
            best_label = label

        # if that is also tied, use the earlier neighbor
        elif closest[label] == closest[best_label]:
            if first[label] < first[best_label]:
                best_label = label

    return best_label
