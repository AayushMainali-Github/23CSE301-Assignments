from A3 import convert
from A4 import minkowski_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    data = convert()
    data = data.dropna()

    v1 = data.iloc[0].tolist()
    v2 = data.iloc[1].tolist()

    p_values = list(range(1, 11))
    distances = []

    for p in p_values:
        d = minkowski_distance(v1, v2, p)
        distances.append(d)
        print("p =", p, "distance =", d)

    plt.plot(p_values, distances, marker="o")
    plt.xlabel("p (order)")
    plt.ylabel("Minkowski distance")
    plt.title("Minkowski distance for p = 1 to 10")
    plt.grid(True)
    plt.savefig("A5_minkowski_plot.png")
    print("Plot saved as A5_minkowski_plot.png")


if __name__ == "__main__":
    main()
