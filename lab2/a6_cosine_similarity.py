"""A6: cosine similarity using complete preprocessed thyroid vectors."""
from __future__ import annotations
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from common import complete_thyroid_matrix, load_thyroid


def first_pair_cosine() -> dict:
    matrix, feature_names = complete_thyroid_matrix(load_thyroid())
    value = float(cosine_similarity(matrix[:2])[0, 1])
    return {"cosine_similarity": value, "encoded_feature_count": len(feature_names)}


def main() -> None:
    result = first_pair_cosine()
    out = Path(__file__).resolve().parents[1] / "outputs" / "a6_cosine_similarity.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
