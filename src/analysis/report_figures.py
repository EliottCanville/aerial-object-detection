"""Regenerate the figures of the final report into ``report/figures/``.

Training curves and confusion matrices are rebuilt from ``experiments/``; detection
examples need the local weights (``experiments/*/weights/best.pt``) and the dataset.
Steps whose inputs are missing are skipped with a message.

Usage:
    python -m src.analysis.report_figures
"""

import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import REPO_ROOT, list_images, load_class_names, read_yolo_labels

DATA_DIR = REPO_ROOT / "data"
EXP_DIR = REPO_ROOT / "experiments"
FIG_DIR = REPO_ROOT / "report" / "figures"

CLASS_NAMES_14 = load_class_names()
CLASS_NAMES_2 = ["building", "small vehicle"]
COLORS_14 = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0),
    (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (255, 128, 0), (128, 255, 0),
]
COLORS_2 = [(0, 255, 0), (255, 0, 0)]  # building: green, small vehicle: red

# Headline (mAP50, mAP50-95) values of report.pdf, in %.
REPORTED_MAP = {
    "Full dataset (3234 images)": {"YOLOv8": (87.7, 62.6), "Faster R-CNN": (73.7, 49.2)},
    "Active learning (500 images)": {"YOLOv8": (87.7, 62.4), "Faster R-CNN": (69.4, 44.0)},
}
# Reference image shared with the Faster R-CNN qualitative comparison.
COMPARISON_IMAGE = "75-2021-0660-6865-LA93-0M20-E080-2467_jpeg_jpg.rf.5b64c98a8f3284d1fca04717c823d5ee"


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def training_curves(results_csv: Path, title: str, out: Path) -> None:
    if not results_csv.exists():
        print(f"Skipped {out.name}: {results_csv} not found")
        return
    df = pd.read_csv(results_csv).rename(columns=str.strip)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("mAP50 evolution", "mAP50 (%)", [("metrics/mAP50(B)", "mAP50", 100)]),
        ("mAP50-95 evolution", "mAP50-95 (%)", [("metrics/mAP50-95(B)", "mAP50-95", 100)]),
        ("Precision & recall", "Metric (%)", [("metrics/precision(B)", "Precision", 100),
                                              ("metrics/recall(B)", "Recall", 100)]),
        ("Training losses", "Loss", [("train/box_loss", "Box loss", 1), ("train/cls_loss", "Cls loss", 1),
                                     ("train/dfl_loss", "DFL loss", 1)]),
    ]
    for ax, (panel_title, ylabel, series) in zip(axes.flat, panels):
        for column, label, scale in series:
            ax.plot(df["epoch"], df[column] * scale, linewidth=2, label=label)
        ax.set(title=panel_title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, out)


def copy_confusion_matrices() -> None:
    for src, dst in [
        (EXP_DIR / "01_baseline_full_dataset/confusion_matrix_normalized.png", FIG_DIR / "01_baseline"),
        (EXP_DIR / "03_oversampled_14classes/confusion_matrix_normalized.png", FIG_DIR / "04_oversampled"),
    ]:
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst / src.name)
        print(f"Copied {src.relative_to(REPO_ROOT)}")


def performance_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x, width = np.arange(2), 0.35
    for ax, (setting, scores) in zip(axes, REPORTED_MAP.items()):
        models = list(scores)
        for offset, idx, label, color in [(-width / 2, 0, "mAP50", "#2ecc71"), (width / 2, 1, "mAP50-95", "#3498db")]:
            bars = ax.bar(x + offset, [scores[m][idx] for m in models], width, label=label, color=color)
            for bar in bars:
                ax.annotate(f"{bar.get_height():.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
        ax.set_title(setting, fontsize=14, fontweight="bold")
        ax.set(ylabel="Performance (%)", ylim=(0, 100), xticks=x, xticklabels=models)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, FIG_DIR / "05_comparison_yolo_frcnn/performance_comparison.png")


def draw_ground_truth(image: np.ndarray, labels_path: Path, names: list[str], colors) -> np.ndarray:
    h, w = image.shape[:2]
    for class_id, cx, cy, bw, bh in read_yolo_labels(labels_path):
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = colors[class_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, names[class_id][:15], (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return image


def draw_predictions(image: np.ndarray, result, names: list[str], colors, font_scale: float) -> np.ndarray:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id, conf = int(box.cls[0]), float(box.conf[0])
        color = colors[class_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{names[class_id][:10]} {conf:.2f}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    return image


def load_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def yolo_comparison_detections(YOLO) -> None:
    """Ground truth vs prediction on the reference image, for both 2-class models."""
    image_path = DATA_DIR / f"cadot_2cls/valid/images/{COMPARISON_IMAGE}.jpg"
    labels_path = DATA_DIR / f"cadot_2cls/valid/labels/{COMPARISON_IMAGE}.txt"
    if not image_path.exists():
        print(f"Skipped YOLO comparison: {image_path} not found")
        return

    for run, name, title in [
        ("01_baseline_full_dataset", "baseline", "YOLO - Full dataset"),
        ("02_active_learning_500", "active_learning", "YOLO - Active learning 500"),
    ]:
        weights = EXP_DIR / run / "weights/best.pt"
        if not weights.exists():
            print(f"Skipped {name}: {weights} not found")
            continue
        result = YOLO(str(weights)).predict(str(image_path), conf=0.5, verbose=False)[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(draw_ground_truth(load_rgb(image_path), labels_path, CLASS_NAMES_2, COLORS_2))
        axes[1].imshow(draw_predictions(load_rgb(image_path), result, CLASS_NAMES_2, COLORS_2, 0.4))
        for ax, subtitle in zip(axes, ["Ground truth", "YOLO prediction"]):
            ax.set_title(subtitle, fontsize=12, fontweight="bold")
            ax.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        save(fig, FIG_DIR / f"05_comparison_yolo_frcnn/yolo_{name}_detection.png")


def detection_examples_14cls(YOLO) -> None:
    """Six validation images with at least two classes: ground truth vs prediction."""
    weights = EXP_DIR / "03_oversampled_14classes/weights/best.pt"
    images_dir, labels_dir = DATA_DIR / "cadot/valid/images", DATA_DIR / "cadot/valid/labels"
    if not weights.exists() or not images_dir.exists():
        print("Skipped 14-class examples: weights or validation images not found")
        return

    candidates = list_images(images_dir)[:100]
    selected = [p for p in candidates if len({b[0] for b in read_yolo_labels(labels_dir / f"{p.stem}.txt")}) >= 2][:6]
    selected = selected if len(selected) == 6 else candidates[:6]

    model = YOLO(str(weights))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for idx, image_path in enumerate(selected):
        row, col = idx // 2, (idx % 2) * 2
        result = model.predict(str(image_path), conf=0.4, verbose=False)[0]
        gt = draw_ground_truth(load_rgb(image_path), labels_dir / f"{image_path.stem}.txt", CLASS_NAMES_14, COLORS_14)
        pred = draw_predictions(load_rgb(image_path), result, CLASS_NAMES_14, COLORS_14, 0.35)
        panels = [(axes[row, col], gt, "Ground truth"), (axes[row, col + 1], pred, "YOLO prediction")]
        for ax, image, subtitle in panels:
            ax.imshow(image)
            ax.set_title(subtitle, fontsize=10)
            ax.axis("off")
    fig.suptitle("Detection examples - 14-class model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, FIG_DIR / "04_oversampled/detection_examples_14classes.png")


def main() -> None:
    training_curves(EXP_DIR / "01_baseline_full_dataset/results.csv",
                    "YOLO training metrics - full dataset (2 classes)",
                    FIG_DIR / "01_baseline/yolo_training_curves.png")
    training_curves(EXP_DIR / "03_oversampled_14classes/results.csv",
                    "YOLO training metrics - 14 classes (oversampled)",
                    FIG_DIR / "04_oversampled/yolo_14class_training_curves.png")
    copy_confusion_matrices()
    performance_comparison()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed: detection figures skipped")
        return
    yolo_comparison_detections(YOLO)
    detection_examples_14cls(YOLO)


if __name__ == "__main__":
    main()
