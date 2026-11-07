"""Compare two YOLOv8 runs trained on different amounts of data.

Prints a final-metrics table and saves in ``--out``:
  * ``training_comparison.png``   validation metrics per epoch for both runs
  * ``metrics_comparison_bar.png`` final precision / recall / mAP
  * ``efficiency_comparison.png``  share of training images vs mAP50

Usage (full dataset vs active learning):
    python -m src.analysis.compare_runs \
        --runs experiments/01_baseline_full_dataset/results.csv experiments/02_active_learning_500/results.csv \
        --labels "Full dataset" "Active learning" --images 3234 500 --out report/figures/03_comparison
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "val/box_loss": "Box loss",
    "val/cls_loss": "Classification loss",
}


def label_bars(ax, bars, fmt: str) -> None:
    for bar in bars:
        ax.annotate(fmt.format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs=2, type=Path, required=True, help="two Ultralytics results.csv files")
    parser.add_argument("--labels", nargs=2, required=True)
    parser.add_argument("--images", nargs=2, type=int, required=True, help="number of training images of each run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = [pd.read_csv(path).rename(columns=str.strip) for path in args.runs]
    legends = [f"{label} ({n:,} imgs)" for label, n in zip(args.labels, args.images)]
    finals = [run.iloc[-1] for run in runs]

    table = pd.DataFrame({legend: [f"{final[col]:.4f}" for col in METRICS] for legend, final in zip(legends, finals)},
                         index=list(METRICS.values()))
    print(table.to_string())
    map50 = [final["metrics/mAP50(B)"] for final in finals]
    data_share = args.images[1] / args.images[0]
    print(f"\n{args.labels[1]} uses {data_share:.1%} of the images, "
          f"mAP50 change: {(map50[1] - map50[0]) / map50[0]:+.2%}")

    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training comparison: {args.labels[0]} vs {args.labels[1]}", fontsize=16, fontweight="bold")
    for ax, (column, title) in zip(axes.flat, METRICS.items()):
        for run, legend in zip(runs, legends):
            ax.plot(run["epoch"], run[column], label=legend, linewidth=2, alpha=0.8)
        ax.set(title=title, xlabel="Epoch", ylabel=title, ylim=(0, 1) if column.startswith("metrics") else None)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "training_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    x, width = np.arange(4), 0.35
    for offset, final, legend in zip((-width / 2, width / 2), finals, legends):
        bars = ax.bar(x + offset, [final[col] for col in list(METRICS)[:4]], width, label=legend, alpha=0.8)
        label_bars(ax, bars, "{:.3f}")
    ax.set(title="Final performance", ylabel="Score", ylim=(0, 1), xticks=x, xticklabels=list(METRICS.values())[:4])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.out / "metrics_comparison_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    for offset, share, score, label in zip((-width / 2, width / 2), (1.0, data_share), map50, args.labels):
        label_bars(ax, ax.bar(x + offset, [share * 100, score * 100], width, label=label, alpha=0.8), "{:.1f}%")
    ax.set(title="Data efficiency", ylabel="Percentage (%)", xticks=x, xticklabels=["Training images", "mAP50"])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.out / "efficiency_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to {args.out}")


if __name__ == "__main__":
    main()
