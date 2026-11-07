"""Plot the training curves of a YOLOv8 run and, optionally, sample detections.

Produces in ``--out``:
  * ``training_curves.png``         losses, precision, recall and mAP per epoch
  * ``detection_examples.png``      predictions on 6 images        (needs --weights and --images)
  * ``confidence_distribution.png`` confidence histograms per class (needs --weights and --images)

Usage:
    python -m src.analysis.plot_training --results experiments/01_baseline_full_dataset/results.csv \
        --weights experiments/01_baseline_full_dataset/weights/best.pt --images data/cadot/test/images \
        --out report/figures/01_baseline
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import list_images, load_class_names

PANELS = [  # (title, [(column, label)], y-limits)
    ("Box regression loss", [("train/box_loss", "Train"), ("val/box_loss", "Validation")], None),
    ("Classification loss", [("train/cls_loss", "Train"), ("val/cls_loss", "Validation")], None),
    ("Distribution focal loss", [("train/dfl_loss", "Train"), ("val/dfl_loss", "Validation")], None),
    ("Precision", [("metrics/precision(B)", "Precision")], (0, 1)),
    ("Recall", [("metrics/recall(B)", "Recall")], (0, 1)),
    ("Mean average precision", [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")], (0, 1)),
]


def plot_curves(results: pd.DataFrame, title: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for ax, (panel_title, series, ylim) in zip(axes.flat, PANELS):
        for column, label in series:
            ax.plot(results["epoch"], results[column], label=label, linewidth=2)
        ax.set(title=panel_title, xlabel="Epoch", ylim=ylim)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_detections(predictions, names: list[str], out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    for idx, (result, ax) in enumerate(zip(predictions, axes.flat)):
        ax.imshow(result.plot()[..., ::-1])  # BGR -> RGB
        ax.axis("off")
        classes = result.boxes.cls.cpu().numpy().astype(int)
        counts = ", ".join(f"{np.sum(classes == c)} {names[c]}" for c in np.unique(classes))
        ax.set_title(f"Image {idx + 1}: {counts or 'no detections'}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "detection_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confidences(predictions, names: list[str], out: Path) -> None:
    per_class = defaultdict(list)
    for result in predictions:
        for class_id, conf in zip(result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            per_class[names[class_id]].append(conf)
    groups = {"All detections": [c for confs in per_class.values() for c in confs]}
    groups |= dict(sorted(per_class.items(), key=lambda kv: -len(kv[1]))[:3])  # 3 most detected classes

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)
    for ax, (label, confs) in zip(axes[0], groups.items()):
        ax.hist(confs, bins=20, edgecolor="black", alpha=0.7)
        if confs:
            ax.axvline(np.mean(confs), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(confs):.3f}")
            ax.legend()
        ax.set(title=f"{label} (n={len(confs)})", xlabel="Confidence score", ylabel="Frequency")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Ultralytics results.csv")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None, help="folder of images for the detection examples")
    parser.add_argument("--names", type=Path, default=None, help="optional '<id> <name>' file overriding model names")
    parser.add_argument("--title", default="Training and validation metrics")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    results.columns = results.columns.str.strip()  # older Ultralytics versions pad column names
    final = results.iloc[-1]
    print(f"{len(results)} epochs | final mAP50 = {final['metrics/mAP50(B)']:.4f}, "
          f"mAP50-95 = {final['metrics/mAP50-95(B)']:.4f}")
    plot_curves(results, args.title, args.out)

    if args.weights and args.images:
        from ultralytics import YOLO  # only needed for the inference figures

        model = YOLO(str(args.weights))
        names = load_class_names(args.names) if args.names else [model.names[i] for i in sorted(model.names)]
        predictions = model.predict(source=list_images(args.images)[:6], conf=0.25, iou=0.45, verbose=False)
        plot_detections(predictions, names, args.out)
        plot_confidences(predictions, names, args.out)

    print(f"Figures saved to {args.out}")


if __name__ == "__main__":
    main()
