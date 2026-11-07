"""Rebalance a YOLO dataset by duplicating images that contain minority classes.

Each minority class gets a factor ``min(target // count, max_factor)``. An image is
then copied ``f - 1`` extra times, where ``f`` is the largest factor among the
minority classes it contains. Majority classes are never used to drive duplication,
but their instances are duplicated along with the image (hence they still grow).

On the 500-image core-set subset this yields 2933 images (see
``experiments/03_oversampled_14classes/oversampling_summary.json``).

Usage:
    python -m src.data.oversample_minority_classes --src data/active500_14cls --dst data/oversampled_14cls
"""

import argparse
import json
import shutil
from pathlib import Path

from src.utils import count_instances, list_images, load_class_names, read_yolo_labels

# building, crosswalk, medium vehicle, small vehicle: already well represented
MAJORITY_CLASSES = {1, 2, 6, 10}
TARGET_INSTANCES = 500
MAX_FACTOR = 50  # caps duplication of extremely rare classes (a handful of instances)
DEFAULT_SUMMARY = Path("experiments/03_oversampled_14classes/oversampling_summary.json")


def oversampling_factors(instances: dict[int, int]) -> dict[int, int]:
    return {
        class_id: 1 if class_id in MAJORITY_CLASSES else min(max(1, TARGET_INSTANCES // count), MAX_FACTOR)
        for class_id, count in instances.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("data/active500_14cls"))
    parser.add_argument("--dst", type=Path, default=Path("data/oversampled_14cls"))
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    names = load_class_names()
    instances, _ = count_instances(args.src / "labels")
    factors = oversampling_factors(instances)

    print("Oversampling factors:")
    for class_id in sorted(factors):
        print(f"  {class_id:2d} {names[class_id]:<18} {instances[class_id]:>6} instances  x{factors[class_id]}")

    (args.dst / "images").mkdir(parents=True, exist_ok=True)
    (args.dst / "labels").mkdir(parents=True, exist_ok=True)

    n_images = 0
    for image in list_images(args.src / "images"):
        label = args.src / "labels" / f"{image.stem}.txt"
        copies = [image.stem]

        minority = {c for c, *_ in read_yolo_labels(label)} - MAJORITY_CLASSES
        if minority:
            factor = max(factors[c] for c in minority)
            copies += [f"{image.stem}_dup{i}" for i in range(1, factor)]

        for stem in copies:
            shutil.copy2(image, args.dst / "images" / f"{stem}{image.suffix}")
            if label.exists():
                shutil.copy2(label, args.dst / "labels" / f"{stem}.txt")
        n_images += len(copies)

    new_instances, _ = count_instances(args.dst / "labels")
    print(f"\n{len(list_images(args.src / 'images'))} -> {n_images} images")
    for class_id in sorted(new_instances):
        print(f"  {class_id:2d} {names[class_id]:<18} {instances[class_id]:>6} -> {new_instances[class_id]:>6}")

    summary = {
        "original_images": len(list_images(args.src / "images")),
        "total_images": n_images,
        "original_distribution": dict(instances),
        "new_distribution": dict(new_instances),
        "oversampling_factors": factors,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {args.summary}")


if __name__ == "__main__":
    main()
