"""Run a trained Faster R-CNN checkpoint on images and save annotated predictions.

Usage:
    python -m src.faster_rcnn.predict --weights weights/fasterrcnn_r50fpn_cadot_2cls_full.pth \
        --source path/to/images --out runs/predict/frcnn
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import torch
from torchvision.transforms.functional import to_tensor

from src.faster_rcnn.dataset import CATEGORY_TO_LABEL
from src.faster_rcnn.train import build_model
from src.utils import list_images

LABEL_NAMES = {label: name for name, label in CATEGORY_TO_LABEL.items()}
COLORS = {1: (0, 200, 0), 2: (0, 0, 255)}  # BGR: building green, small vehicle red


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="checkpoint with a 'model_state_dict' entry")
    parser.add_argument("--source", type=Path, required=True, help="image file or folder")
    parser.add_argument("--out", type=Path, default=Path("runs/predict/frcnn"))
    parser.add_argument("--conf", type=float, default=0.5, help="minimum detection score")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu")["model_state_dict"])
    model.to(device).eval()

    images = [args.source] if args.source.is_file() else list_images(args.source)[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    total = Counter()
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            continue
        with torch.no_grad():
            output = model([to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device)])[0]

        keep = output["scores"] >= args.conf
        counts = Counter()
        for box, label, score in zip(output["boxes"][keep], output["labels"][keep], output["scores"][keep]):
            x1, y1, x2, y2 = map(int, box.tolist())
            name = LABEL_NAMES[int(label)]
            counts[name] += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[int(label)], 2)
            cv2.putText(image, f"{name} {score:.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[int(label)], 1)

        cv2.imwrite(str(args.out / f"pred_{image_path.name}"), image)
        total += counts
        print(f"{image_path.name}: {sum(counts.values())} detections")

    print(f"\n{len(images)} images -> {args.out}")
    print("Detections per class: " + ", ".join(f"{k} ({v})" for k, v in total.most_common()))


if __name__ == "__main__":
    main()
