"""A7: pairwise Jaccard, SMC and cosine heatmaps for first 20 records."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from common import complete_thyroid_matrix, encode_binary, jaccard_smc, load_thyroid


def binary_similarity_matrices(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(binary)
    jc, smc = np.eye(n), np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            values = jaccard_smc(binary[i], binary[j])
            jc[i, j] = jc[j, i] = values["jaccard"]
            smc[i, j] = smc[j, i] = values["smc"]
    return jc, smc


def save_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", square=True, cbar=True, ax=ax,
                xticklabels=range(1, len(matrix) + 1),
                yticklabels=range(1, len(matrix) + 1))
    ax.set_title(title)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Observation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_heatmaps() -> dict:
    df = load_thyroid()
    binary = encode_binary(df)[:20]
    complete, _ = complete_thyroid_matrix(df)
    jc, smc = binary_similarity_matrices(binary)
    cos = cosine_similarity(complete[:20])
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    save_heatmap(jc, "Jaccard Similarity: First 20 Thyroid Records", out_dir / "a7_jaccard_heatmap.png")
    save_heatmap(smc, "Simple Matching Similarity: First 20 Thyroid Records", out_dir / "a7_smc_heatmap.png")
    save_heatmap(cos, "Cosine Similarity: First 20 Thyroid Records", out_dir / "a7_cosine_heatmap.png")
    off = np.triu_indices(20, 1)
    return {
        "mean_off_diagonal_jaccard": float(jc[off].mean()),
        "mean_off_diagonal_smc": float(smc[off].mean()),
        "mean_off_diagonal_cosine": float(cos[off].mean()),
    }


def main() -> None:
    result = run_heatmaps()
    out = Path(__file__).resolve().parents[1] / "outputs" / "a7_similarity_summary.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
