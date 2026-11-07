"""Print per-class instance counts of one or more YOLO label folders.

The imbalance factor is the ratio between the most frequent class and each class:
it is the quantity the oversampling step (``oversample_minority_classes.py``) tries
to reduce for the 14-class experiment.

Usage:
    python -m src.data.class_distribution data/cadot/train/labels data/cadot/valid/labels
    python -m src.data.class_distribution data/active500_14cls/labels
"""

import argparse
from pathlib import Path

from src.utils import count_instances, load_class_names


def print_distribution(labels_dir: Path, class_names: list[str]) -> None:
    instances, images = count_instances(labels_dir)
    n_files = len(list(labels_dir.glob("*.txt")))
    majority = max(instances.values(), default=0)

    print(f"\n{labels_dir}  ({n_files:,} label files, {sum(instances.values()):,} instances)")
    print(f"{'id':>3}  {'class':<18} {'instances':>10} {'images':>8} {'imbalance':>10}")
    print("-" * 54)
    for class_id, count in instances.most_common():
        name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        print(f"{class_id:>3}  {name:<18} {count:>10,} {images[class_id]:>8,} {majority / count:>9.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels_dirs", nargs="+", type=Path, help="folders of YOLO .txt label files")
    parser.add_argument("--classes", type=Path, default=None, help="class names file (default: configs/classes.txt)")
    args = parser.parse_args()

    class_names = load_class_names(args.classes) if args.classes else load_class_names()
    for labels_dir in args.labels_dirs:
        print_distribution(labels_dir, class_names)


if __name__ == "__main__":
    main()
