"""Optional O2: random 20-record thyroid similarity experiment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from common import complete_thyroid_matrix, encode_binary, load_thyroid
from a7_similarity_heatmaps import binary_similarity_matrices, save_heatmap


def main() -> None:
    rng = np.random.default_rng(42)
    df = load_thyroid()
    ids = np.sort(rng.choice(len(df), size=20, replace=False))
    binary = encode_binary(df)[ids]
    complete, _ = complete_thyroid_matrix(df)
    jc, smc = binary_similarity_matrices(binary)
    cos = cosine_similarity(complete[ids])
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    save_heatmap(jc, "Random-sample Jaccard Similarity", out_dir / "o2_random_jaccard.png")
    save_heatmap(smc, "Random-sample Simple Matching Similarity", out_dir / "o2_random_smc.png")
    save_heatmap(cos, "Random-sample Cosine Similarity", out_dir / "o2_random_cosine.png")
    result = {"seed": 42, "zero_based_row_indices": ids.tolist()}
    (out_dir / "o2_random_sample.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
