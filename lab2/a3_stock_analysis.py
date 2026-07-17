"""A3: descriptive statistics, probabilities, timing and scatter plot."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter_ns
import matplotlib.pyplot as plt
import numpy as np
from common import load_stock


def custom_mean(values: np.ndarray) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total / len(values)


def custom_variance(values: np.ndarray) -> float:
    mean = custom_mean(values)
    total = 0.0
    for value in values:
        total += (float(value) - mean) ** 2
    return total / len(values)


def average_runtime_ns(function, values: np.ndarray, runs: int = 10) -> float:
    elapsed = []
    for _ in range(runs):
        start = perf_counter_ns()
        function(values)
        elapsed.append(perf_counter_ns() - start)
    return float(np.mean(elapsed))


def analyze_stock() -> dict:
    df = load_stock().dropna(subset=["Price", "Chg%", "Day", "Month"])
    price = df["Price"].to_numpy(dtype=float)
    wed = df.loc[df["Day"].str.lower().eq("wed"), "Price"]
    apr = df.loc[df["Month"].str.lower().eq("apr"), "Price"]
    loss = df["Chg%"].map(lambda x: x < 0)
    profit = df["Chg%"].map(lambda x: x > 0)
    is_wed = df["Day"].str.lower().eq("wed")
    result = {
        "observations": int(len(df)),
        "numpy_mean": float(np.mean(price)),
        "numpy_population_variance": float(np.var(price)),
        "custom_mean": custom_mean(price),
        "custom_population_variance": custom_variance(price),
        "wednesday_count": int(len(wed)),
        "wednesday_sample_mean": float(wed.mean()),
        "april_count": int(len(apr)),
        "april_sample_mean": float(apr.mean()),
        "loss_count": int(loss.sum()),
        "probability_loss": float(loss.mean()),
        "profit_and_wednesday_count": int((profit & is_wed).sum()),
        "probability_profit_and_wednesday": float((profit & is_wed).mean()),
        "conditional_probability_profit_given_wednesday": float(
            (profit & is_wed).sum() / is_wed.sum()
        ),
        "average_runtime_ns_10_runs": {
            "numpy_mean": average_runtime_ns(np.mean, price),
            "custom_mean": average_runtime_ns(custom_mean, price),
            "numpy_variance": average_runtime_ns(np.var, price),
            "custom_variance": average_runtime_ns(custom_variance, price),
        },
    }
    return result, df


def save_scatter(df, path: Path) -> None:
    order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    xpos = {day: i for i, day in enumerate(order)}
    x = df["Day"].map(xpos)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(x, df["Chg%"] * 100, alpha=0.65)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Change (%)")
    ax.set_title("IRCTC Change Percentage by Day of Week")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    result, df = analyze_stock()
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    save_scatter(df, out_dir / "a3_stock_day_scatter.png")
    (out_dir / "a3_stock_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
