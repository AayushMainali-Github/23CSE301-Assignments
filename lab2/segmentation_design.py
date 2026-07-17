"""Methodology design: reusable customer/patient segmentation system diagrams."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_flow(path: Path) -> None:
    labels = ["Input handling", "Validation & cleaning", "Imputation & encoding",
              "Scaling & rank check", "Similarity / PCA", "Clustering",
              "Explanation & output"]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, len(labels) * 2.1)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, label in enumerate(labels):
        x = i * 2.1 + 0.1
        box = FancyBboxPatch((x, 0.65), 1.65, 0.7, boxstyle="round,pad=0.04")
        ax.add_patch(box)
        ax.text(x + 0.825, 1.0, label, ha="center", va="center", fontsize=9, wrap=True)
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.65, 1.0), (x + 2.05, 1.0), arrowstyle="->", mutation_scale=14))
    ax.set_title("Data Flow for Customer/Patient Segmentation")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_architecture(path: Path) -> None:
    layers = [
        ("Data layer", "Excel/CSV/EHR/transaction records"),
        ("Preparation layer", "schema checks, missingness, outliers, encoding, scaling"),
        ("Analytics layer", "rank analysis, similarity, PCA, clustering k=2...8"),
        ("Interpretability layer", "cluster profiles, feature contributions, stability"),
        ("Application layer", "segment dashboard, recommendations, monitoring"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    for i, (title, detail) in enumerate(layers):
        y = 8.4 - i * 1.75
        ax.add_patch(FancyBboxPatch((1, y), 8, 1.05, boxstyle="round,pad=0.05"))
        ax.text(1.35, y + 0.67, title, fontweight="bold", va="center")
        ax.text(3.25, y + 0.52, detail, va="center", fontsize=9)
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((5, y), (5, y - 0.6), arrowstyle="->", mutation_scale=15))
    ax.set_title("Proposed Segmentation System Architecture")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "outputs"
    draw_flow(out / "segmentation_data_flow.png")
    draw_architecture(out / "segmentation_architecture.png")
    print("Saved segmentation_data_flow.png and segmentation_architecture.png")


if __name__ == "__main__":
    main()
