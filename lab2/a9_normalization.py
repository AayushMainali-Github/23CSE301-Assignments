"""A9: robust scaling of outlier-heavy numeric thyroid variables."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import RobustScaler
from common import THYROID_NUMERIC
from a8_imputation import impute_thyroid


def normalize_thyroid() -> tuple[pd.DataFrame, dict]:
    df, _ = impute_thyroid()
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[THYROID_NUMERIC])
    result_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in THYROID_NUMERIC])
    result = {
        "technique": "RobustScaler: (x - median) / IQR",
        "scaled_attributes": THYROID_NUMERIC,
        "centers": dict(zip(THYROID_NUMERIC, scaler.center_.tolist())),
        "scales": dict(zip(THYROID_NUMERIC, scaler.scale_.tolist())),
    }
    return result_df, result


def main() -> None:
    data, result = normalize_thyroid()
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    data.to_csv(out_dir / "a9_thyroid_normalized.csv", index=False)
    (out_dir / "a9_normalization_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
