"""Shared data-loading and preprocessing utilities for Lab Session 02."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "data" / "Lab Session Data.xlsx"

PURCHASE_FEATURES = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
THYROID_NUMERIC = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
THYROID_BINARY = [
    "sex", "on thyroxine", "query on thyroxine",
    "on antithyroid medication", "sick", "pregnant",
    "thyroid surgery", "I131 treatment", "query hypothyroid",
    "query hyperthyroid", "lithium", "goitre", "tumor",
    "hypopituitary", "psych", "TSH measured", "T3 measured",
    "TT4 measured", "T4U measured", "FTI measured", "TBG measured",
]


def workbook_path(path: str | Path | None = None) -> Path:
    p = Path(path) if path else DEFAULT_WORKBOOK
    if not p.exists():
        raise FileNotFoundError(f"Workbook not found: {p}")
    return p


def load_purchase(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_excel(workbook_path(path), sheet_name="Purchase data").iloc[:, :5].copy()


def load_stock(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_excel(workbook_path(path), sheet_name="IRCTC Stock Price")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Price", "Open", "High", "Low", "Chg%"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_thyroid(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_excel(workbook_path(path), sheet_name="thyroid0387_UCI", dtype=str)
    df = df.replace("?", np.nan)
    for c in ["Record ID", *THYROID_NUMERIC]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def encode_binary(df: pd.DataFrame) -> np.ndarray:
    encoded = df[THYROID_BINARY].replace({"t": 1, "f": 0, "M": 1, "F": 0})
    encoded = encoded.apply(pd.to_numeric, errors="coerce")
    return SimpleImputer(strategy="most_frequent").fit_transform(encoded)


def complete_thyroid_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Create a fully numeric matrix for cosine similarity.

    Numeric values are median-imputed and min-max scaled. Binary clinical
    flags are mapped to 0/1, and referral source is one-hot encoded.
    The target and record identifier are intentionally excluded.
    """
    num = SimpleImputer(strategy="median").fit_transform(df[THYROID_NUMERIC])
    num = MinMaxScaler().fit_transform(num)
    binary = encode_binary(df)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    referral = encoder.fit_transform(df[["referral source"]])
    names = THYROID_NUMERIC + THYROID_BINARY + [
        f"referral_{x}" for x in encoder.categories_[0]
    ]
    return np.hstack([num, binary, referral]), names


def jaccard_smc(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    f11 = int(np.sum((a == 1) & (b == 1)))
    f10 = int(np.sum((a == 1) & (b == 0)))
    f01 = int(np.sum((a == 0) & (b == 1)))
    f00 = int(np.sum((a == 0) & (b == 0)))
    jc_denom = f11 + f10 + f01
    total = f11 + f10 + f01 + f00
    return {
        "f11": f11, "f10": f10, "f01": f01, "f00": f00,
        "jaccard": f11 / jc_denom if jc_denom else 1.0,
        "smc": (f11 + f00) / total if total else 1.0,
    }
