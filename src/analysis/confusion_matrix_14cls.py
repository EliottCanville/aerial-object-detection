"""Redraw the 14-class confusion matrix with the reference class names.

The 14-class model was trained with a names list shifted by one position (class ids
were right, embedded names were not), so the matrix plotted by Ultralytics
(``experiments/03_oversampled_14classes/confusion_matrix_normalized.png``) carries
wrong axis labels. Its values are transcribed below and plotted again against
``configs/classes.txt``, together with the per-class rate of objects missed
(predicted as background).

Usage:
    python -m src.analysis.confusion_matrix_14cls
"""

import matplotlib.pyplot as plt
import numpy as np

from src.utils import REPO_ROOT, load_class_names

OUT_DIR = REPO_ROOT / "report" / "figures" / "04_oversampled"
CLASS_NAMES = load_class_names()

# Normalized confusion matrix: rows = predicted, columns = true, last row/column = background.
CONFUSION = np.array([
    # bask  build cross foot  grave large med   play  round ship  small swim  tenn  train bg
    [0.86, 0.00, 0.06, 0.00, 0.01, 0.00, 0.00, 0.11, 0.00, 0.04, 0.11, 0.00, 0.45, 0.00, 0.00],  # basketball field
    [0.00, 0.89, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.00, 0.00],  # building
    [0.00, 0.00, 0.53, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # crosswalk
    [0.00, 0.00, 0.00, 0.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # football field
    [0.00, 0.00, 0.00, 0.00, 0.74, 0.02, 0.00, 0.06, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00],  # graveyard
    [0.00, 0.00, 0.00, 0.00, 0.08, 0.81, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00],  # large vehicle
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.67, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # medium vehicle
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.64, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # playground
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.26, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # roundabout
    [0.00, 0.00, 0.00, 0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0.89, 0.00, 0.00, 0.44, 0.00, 0.00],  # ship
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.92, 0.00, 0.00, 0.00, 0.00],  # small vehicle
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.68, 0.00, 0.00, 0.00],  # swimming pool
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.58, 0.00, 0.00],  # tennis court
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # train (no val data)
    [0.14, 0.11, 0.41, 0.29, 0.17, 0.06, 0.33, 0.36, 0.57, 0.10, 0.04, 0.21, 0.42, 0.00, 0.00],  # background
])


def plot_confusion() -> None:
    matrix = CONFUSION[:14, :14]  # background row/column dropped for readability
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel("Proportion", rotation=-90, va="bottom")

    ticks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES)
    for i, j in zip(*np.nonzero(matrix > 0.01)):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if matrix[i, j] > 0.5 else "black")
    ax.set_xlabel("True class", fontsize=12)
    ax.set_ylabel("Predicted class", fontsize=12)
    ax.set_title("Normalized confusion matrix - 14-class model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix_14classes_corrected.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_missed_objects() -> None:
    missed = CONFUSION[14, :14]  # true objects predicted as background
    colors = ["#e74c3c" if r > 0.3 else "#f39c12" if r > 0.15 else "#2ecc71" for r in missed]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(CLASS_NAMES, missed, color=colors)
    for bar, rate in zip(bars, missed):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height() / 2, f"{rate:.0%}", va="center", fontsize=9)
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.7, label="Critical (30%)")
    ax.axvline(0.15, color="orange", linestyle="--", alpha=0.7, label="Acceptable (15%)")
    ax.set(xlim=(0, 0.7), xlabel="False negative rate (predicted as background)")
    ax.set_title("Missed objects per class", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "false_negative_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion()
    plot_missed_objects()
    print(f"Figures saved to {OUT_DIR}")
