from A3 import convert
from A4 import minkowski_distance
from scipy.spatial.distance import minkowski


def main():
    data = convert()
    data = data.dropna()

    v1 = data.iloc[0].tolist()
    v2 = data.iloc[1].tolist()

    print("Comparing own function vs scipy minkowski:\n")
    for p in range(1, 11):
        mine = minkowski_distance(v1, v2, p)
        pkg = minkowski(v1, v2, p)
        print("p =", p, "  own =", mine, "  scipy =", pkg, "  same?", abs(mine - pkg) < 1e-9)


if __name__ == "__main__":
    main()
