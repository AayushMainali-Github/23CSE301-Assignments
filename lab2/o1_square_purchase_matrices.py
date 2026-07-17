"""Optional O1: compare square purchase submatrices with the full solution."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from common import PURCHASE_FEATURES, load_purchase


def solve(x: np.ndarray, y: np.ndarray) -> dict:
    return {"rank": int(np.linalg.matrix_rank(x)), "costs": (np.linalg.pinv(x) @ y).tolist()}


def main() -> None:
    df = load_purchase()
    x = df[PURCHASE_FEATURES].to_numpy(float)
    y = df["Payment (Rs)"].to_numpy(float)
    result = {
        "full_matrix": solve(x, y),
        "square_rows_1_to_3": solve(x[:3], y[:3]),
        "square_rows_4_to_6": solve(x[3:6], y[3:6]),
    }
    result["interpretation"] = (
        "A full-rank square subset recovers the same product costs; a singular subset may not."
    )
    out = Path(__file__).resolve().parents[1] / "outputs" / "o1_square_matrices.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
