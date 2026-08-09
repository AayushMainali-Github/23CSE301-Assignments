from A3 import convert
from A8 import my_mean, my_variance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = convert()
    data = data.dropna()

    feature = data["Income"].tolist()

    counts, bins = np.histogram(feature, bins=20)
    print("bin edges:", bins)
    print("counts:", counts)

    print("mean:", my_mean(feature))
    print("variance:", my_variance(feature))

    plt.hist(feature, bins=20)
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.title("Income histogram")
    plt.savefig("A10_histogram.png")
    print("Plot saved as A10_histogram.png")


if __name__ == "__main__":
    main()
