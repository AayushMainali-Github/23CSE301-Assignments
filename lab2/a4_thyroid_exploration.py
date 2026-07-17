"""A4: thyroid datatype, missingness, range, outlier and summary audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import THYROID_BINARY, THYROID_NUMERIC, load_thyroid


def numeric_audit(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce")
    clean = values.dropna()
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "missing": int(values.isna().sum()),
        "minimum": float(clean.min()), "maximum": float(clean.max()),
        "mean": float(clean.mean()), "variance": float(clean.var(ddof=0)),
        "standard_deviation": float(clean.std(ddof=0)), "median": float(clean.median()),
        "iqr_lower_fence": float(low), "iqr_upper_fence": float(high),
        "iqr_outliers": int(((clean < low) | (clean > high)).sum()),
    }


def explore_thyroid() -> tuple[dict, pd.DataFrame]:
    df = load_thyroid()
    rows = []
    numeric_cols = ["Record ID", *THYROID_NUMERIC]
    for col in df.columns:
        if col in numeric_cols:
            dtype, encoding = "numeric", "not required"
            stat = numeric_audit(df[col])
            domain = f"{stat['minimum']} to {stat['maximum']}"
            outliers = stat["iqr_outliers"]
        elif col in THYROID_BINARY:
            dtype, encoding = "binary nominal", "map f/F=0 and t/M=1"
            domain = ", ".join(map(str, df[col].dropna().unique()[:10]))
            outliers = 0
        elif col == "Condition":
            dtype, encoding = "nominal target", "label or one-hot depending on model"
            domain = f"{df[col].nunique(dropna=True)} classes"
            outliers = 0
        else:
            dtype, encoding = "nominal", "one-hot encoding"
            domain = ", ".join(map(str, df[col].dropna().unique()[:10]))
            outliers = 0
        rows.append({
            "attribute": col, "datatype": dtype, "suggested_encoding": encoding,
            "range_or_categories": domain, "missing_values": int(df[col].isna().sum()),
            "outlier_count": outliers,
        })
    audit = pd.DataFrame(rows)
    numeric = {c: numeric_audit(df[c]) for c in numeric_cols}
    result = {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "total_missing_values": int(df.isna().sum().sum()),
        "invalid_age_outside_1_to_120": int(((df["age"] < 1) | (df["age"] > 120)).sum()),
        "numeric_summary": numeric,
        "encoding_recommendation": {
            "binary_flags": "binary 0/1 mapping",
            "referral_source": "one-hot encoding",
            "condition": "label encoding for a classifier target",
        },
    }
    return result, audit


def main() -> None:
    result, audit = explore_thyroid()
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    audit.to_csv(out_dir / "a4_thyroid_attribute_audit.csv", index=False)
    (out_dir / "a4_thyroid_summary.json").write_text(json.dumps(result, indent=2))
    print(audit.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
