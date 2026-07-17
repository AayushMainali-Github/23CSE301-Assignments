"""Run all mandatory Lab 2 tasks and optional experiments."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "a1_purchase_regression.py", "a2_customer_classifier.py", "a3_stock_analysis.py",
    "a4_thyroid_exploration.py", "a5_binary_similarity.py", "a6_cosine_similarity.py",
    "a7_similarity_heatmaps.py", "a8_imputation.py", "a9_normalization.py",
    "segmentation_design.py", "o1_square_purchase_matrices.py",
    "o2_random_similarity_sample.py", "o3_marketing_campaign.py",
]


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n===== Running {script} =====")
        subprocess.run([sys.executable, str(code_dir / script)], check=True)
    print("\nAll tasks completed. See the outputs folder.")


if __name__ == "__main__":
    main()
