from A8 import dataset_stats
from A3 import convert
import numpy as np


def main():
    data = convert()
    data = data.dropna()

    matrix = data.values.tolist()
    my_means, my_stds = dataset_stats(matrix)

    feat_vecs = data.to_numpy(dtype=float)
    np_means = feat_vecs.mean(axis=0)
    np_stds = feat_vecs.std(axis=0)

    print("Comparing own stats vs numpy:\n")
    print("Mean close?", np.allclose(my_means, np_means))
    print("Std  close?", np.allclose(my_stds, np_stds))

    print("\nFirst 5 features:")
    for i in range(5):
        print(
            data.columns[i],
            "  my_mean=", round(my_means[i], 4),
            " np_mean=", round(np_means[i], 4),
            "  my_std=", round(my_stds[i], 4),
            " np_std=", round(np_stds[i], 4),
        )


if __name__ == "__main__":
    main()
