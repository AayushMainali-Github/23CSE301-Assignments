"""A2: classify customers as RICH or POOR from purchase behaviour."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from common import PURCHASE_FEATURES, load_purchase


def classify_customers() -> tuple[dict, pd.DataFrame]:
    df = load_purchase()
    x = df[PURCHASE_FEATURES].to_numpy(dtype=float)
    y = (df["Payment (Rs)"].to_numpy(dtype=float) > 200).astype(int)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=3)),
    ])
    pred = cross_val_predict(model, x, y, cv=LeaveOneOut())
    table = df[["Customer", *PURCHASE_FEATURES, "Payment (Rs)"]].copy()
    table["Actual Class"] = ["RICH" if v else "POOR" for v in y]
    table["Predicted Class"] = ["RICH" if v else "POOR" for v in pred]
    result = {
        "threshold_rs": 200,
        "model": "StandardScaler + 3-nearest neighbours",
        "validation": "Leave-one-out cross-validation",
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix_labels": ["POOR", "RICH"],
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=["POOR", "RICH"], output_dict=True,
            zero_division=0,
        ),
    }
    return result, table


def main() -> None:
    result, table = classify_customers()
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    table.to_csv(out_dir / "a2_customer_predictions.csv", index=False)
    (out_dir / "a2_classifier.json").write_text(json.dumps(result, indent=2))
    print(table.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
