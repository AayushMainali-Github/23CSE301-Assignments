"""A5: Jaccard coefficient and Simple Matching Coefficient."""
from __future__ import annotations
import json
from pathlib import Path
from common import encode_binary, jaccard_smc, load_thyroid


def first_pair_similarity() -> dict:
    binary = encode_binary(load_thyroid())
    return jaccard_smc(binary[0], binary[1])


def main() -> None:
    result = first_pair_similarity()
    out = Path(__file__).resolve().parents[1] / "outputs" / "a5_binary_similarity.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("Jaccard ignores joint zeroes and is preferable for sparse positive clinical events.")
    print("SMC counts both joint presences and joint absences, so it is much larger here.")


if __name__ == "__main__":
    main()
