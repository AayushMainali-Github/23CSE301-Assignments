"""A1: matrix dimensionality, rank and pseudoinverse regression."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import PURCHASE_FEATURES, load_purchase


def analyze_purchase() -> dict:
    df = load_purchase()
    x = df[PURCHASE_FEATURES].to_numpy(dtype=float)
    y = df["Payment (Rs)"].to_numpy(dtype=float)
    costs = np.linalg.pinv(x) @ y
    residual = x @ costs - y
    return {
        "vector_space_dimension": int(x.shape[1]),
        "number_of_vectors": int(x.shape[0]),
        "matrix_shape": list(x.shape),
        "rank": int(np.linalg.matrix_rank(x)),
        "unit_costs": dict(zip(PURCHASE_FEATURES, costs.tolist())),
        "max_absolute_residual": float(np.max(np.abs(residual))),
    }


def main() -> None:
    result = analyze_purchase()
    out = Path(__file__).resolve().parents[1] / "outputs" / "a1_purchase.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
