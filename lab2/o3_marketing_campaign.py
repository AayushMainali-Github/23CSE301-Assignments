"""Optional O3: exploration, imputation, scaling and similarity on marketing data."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from common import workbook_path


def main() -> None:
    df = pd.read_excel(workbook_path(), sheet_name="marketing_campaign")
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    # Excel date/object columns can contain mixed Python types; normalize them to strings.
    for col in categorical:
        df[col] = df[col].astype("string")
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", RobustScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    matrix = pre.fit_transform(df)
    cos = cosine_similarity(matrix[:20])
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cos, annot=True, fmt=".2f", square=True, ax=ax)
    ax.set_title("Marketing Campaign: Cosine Similarity for First 20 Records")
    fig.tight_layout(); fig.savefig(out_dir / "o3_marketing_cosine.png", dpi=180); plt.close(fig)
    result = {
        "rows": len(df), "columns": df.shape[1],
        "numeric_columns": numeric, "categorical_columns": categorical,
        "missing_values": int(df.isna().sum().sum()),
        "encoded_shape": list(matrix.shape),
    }
    (out_dir / "o3_marketing_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
