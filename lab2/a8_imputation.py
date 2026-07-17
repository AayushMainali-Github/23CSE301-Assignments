"""A8: central-tendency-based missing-value imputation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import THYROID_NUMERIC, load_thyroid


def has_iqr_outliers(series: pd.Series) -> bool:
    clean = series.dropna()
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    return bool(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).any())


def impute_thyroid() -> tuple[pd.DataFrame, dict]:
    df = load_thyroid()
    df.loc[~df["age"].between(1, 120), "age"] = np.nan
    rules = {}
    for col in THYROID_NUMERIC:
        method = "median" if has_iqr_outliers(df[col]) else "mean"
        value = float(df[col].median() if method == "median" else df[col].mean())
        df[col] = df[col].fillna(value)
        rules[col] = {"method": method, "value": value}
    for col in df.select_dtypes(include="object").columns:
        mode = df[col].mode(dropna=True)
        value = mode.iloc[0] if not mode.empty else "UNKNOWN"
        df[col] = df[col].fillna(value)
        rules[col] = {"method": "mode", "value": str(value)}
    return df, {
        "remaining_missing_values": int(df.isna().sum().sum()),
        "imputation_rules": rules,
    }


def main() -> None:
    df, result = impute_thyroid()
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    df.to_csv(out_dir / "a8_thyroid_imputed.csv", index=False)
    (out_dir / "a8_imputation_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
