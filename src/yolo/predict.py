"""Run a trained YOLOv8 model on images and save annotated predictions.

Each output image gets a footer summarizing the detections per class.
``--names`` relabels class ids with an external names file, which is needed for the
14-class checkpoint: it was trained with a shifted names list (ids are correct,
embedded names are not), see ``configs/classes.txt`` for the reference mapping.

Usage:
    python -m src.yolo.predict --weights experiments/03_oversampled_14classes/weights/best.pt \
        --source data/cadot/valid/images --out runs/predict/14cls --names configs/classes.txt --limit 20
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import list_images, load_class_names


def class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR color per class id."""
    hue = (class_id * 37) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def annotate(image: np.ndarray, boxes, names: list[str]) -> tuple[np.ndarray, Counter]:
    counts = Counter()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id, conf = int(box.cls[0]), float(box.conf[0])
        name = names[class_id] if class_id < len(names) else f"class_{class_id}"
        counts[name] += 1

        color = class_color(class_id)
        text_color = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)  # dark text on bright boxes
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    footer = np.zeros((30, image.shape[1], 3), dtype=np.uint8)
    summary = f"{sum(counts.values())} detections | " + ", ".join(f"{k} ({v})" for k, v in counts.most_common())
    cv2.putText(footer, summary[:90], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([image, footer]), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="image file or folder")
    parser.add_argument("--out", type=Path, default=Path("runs/predict"))
    parser.add_argument("--names", type=Path, default=None, help="optional '<id> <name>' file overriding model names")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N images")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    names = load_class_names(args.names) if args.names else [model.names[i] for i in sorted(model.names)]
    images = [args.source] if args.source.is_file() else list_images(args.source)[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    total = Counter()
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            continue
        result = model(image, conf=args.conf, iou=args.iou, verbose=False)[0]
        annotated, counts = annotate(image.copy(), result.boxes, names)
        cv2.imwrite(str(args.out / f"pred_{image_path.name}"), annotated)
        total += counts
        print(f"{image_path.name}: {sum(counts.values())} detections")

    print(f"\n{len(images)} images -> {args.out}")
    print("Detections per class: " + ", ".join(f"{k} ({v})" for k, v in total.most_common()))


if __name__ == "__main__":
    main()
