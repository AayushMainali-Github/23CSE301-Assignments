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


