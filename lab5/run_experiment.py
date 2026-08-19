from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .knn.classifier import KNNClassifier, WeightedKNNClassifier
from .knn.preprocessing import FeaturePreprocessor


# location of the prepared feature table
FEATURE_FILE = "lab5/features/features.csv"

# all plots and comparison values go here
RESULT_DIR = Path("lab5/results")


def package_model(k, weighted):
    # make the scikit-learn model with the same settings as our model
    weights = "distance" if weighted else "uniform"
    return KNeighborsClassifier(n_neighbors=k, weights=weights,
                                metric="euclidean")


def draw_confusion(matrix):
    # show the four values of the confusion matrix as an image
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["bonafide", "spoof"])
    plt.yticks([0, 1], ["bonafide", "spoof"])
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i][j], ha="center", va="center")
    plt.tight_layout()

    # save the figure for the report
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def main():
    # read data
    data = pd.read_csv(FEATURE_FILE)
    columns = []

    # do not use ids, labels, or file paths as model features
    for column in data:
        if column not in ["utt_id", "label", "audio_path"]:
            columns.append(column)

    # X contains the feature values and y contains the class names
    X = data[columns].values
    y = data["label"].values

    # split data
    # stratify keeps bonafide and spoof counts balanced in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # fill and scale values
    # fit preprocessing only on training data to avoid test data leakage
    process = FeaturePreprocessor("median")
    X_train = process.fit_transform(X_train)
    X_test = process.transform(X_test)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # compare different k values
    values = []
    for k in [1, 3, 5, 7, 9, 11]:
        # our normal kNN and the package version
        custom = KNNClassifier(
            k, "euclidean", 2, "merge", "uniform").fit(
                X_train, y_train)
        package = package_model(k, False).fit(X_train, y_train)

        # our weighted kNN and the package version
        custom_w = WeightedKNNClassifier(
            k, "euclidean", 2, "merge").fit(X_train, y_train)
        package_w = package_model(k, True).fit(X_train, y_train)

        # save train and test accuracy for all four models
        values.append([
            k,
            custom.score(X_train, y_train),
            custom.score(X_test, y_test),
            package.score(X_train, y_train),
            package.score(X_test, y_test),
            custom_w.score(X_train, y_train),
            custom_w.score(X_test, y_test),
            package_w.score(X_train, y_train),
            package_w.score(X_test, y_test),
        ])

    # names for the comparison table
    headings = [
        "k", "custom_train", "custom_test", "package_train", "package_test",
        "custom_weighted_train", "custom_weighted_test",
        "package_weighted_train", "package_weighted_test",
    ]

    # save the numbers so they can be used in the report
    pd.DataFrame(values, columns=headings).to_csv(
        RESULT_DIR / "knn_comparison.csv", index=False)

    # draw the test accuracy curves
    result = pd.DataFrame(values, columns=headings)
    plt.plot(result["k"], result["custom_test"], "o-", label="custom")
    plt.plot(result["k"], result["package_test"], "o--", label="scikit-learn")
    plt.plot(result["k"], result["custom_weighted_test"], "s-",
             label="custom weighted")
    plt.plot(result["k"], result["package_weighted_test"], "s--",
             label="scikit-learn weighted")
    plt.xlabel("number of neighbors (k)")
    plt.ylabel("accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "accuracy_comparison.png", dpi=160)
    plt.close()

    # required k = 3 model
    # this is the value required in the assignment
    custom = KNNClassifier(
        3, "euclidean", 2, "merge", "uniform").fit(
            X_train, y_train)
    package = package_model(3, False).fit(X_train, y_train)
    prediction = custom.predict(X_test)
    print("custom accuracy:", custom.score(X_test, y_test))
    print("package accuracy:", package.score(X_test, y_test))
    print("custom predictions:", prediction[:10])

    # confusion matrix
    # rows are the real labels and columns are the predicted labels
    matrix = confusion_matrix(y_test, prediction,
                              labels=["bonafide", "spoof"])
    draw_confusion(matrix)

    # simple separation plot
    # PCA reduces the feature table to two axes only for visualization
    points = PCA(n_components=2).fit_transform(np.vstack([X_train, X_test]))
    labels = np.concatenate([y_train, y_test])
    for label, color in [("bonafide", "blue"), ("spoof", "red")]:
        # plot one color for each class
        selected = labels == label
        plt.scatter(points[selected, 0], points[selected, 1],
                    label=label, alpha=0.7)
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "feature_separation.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    # run the experiment when this file is called directly
    main()
