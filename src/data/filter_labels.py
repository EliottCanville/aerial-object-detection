"""Build the 2-class version of CADOT (building, small vehicle).

Labels are remapped from the original 14-class ids (1 -> 0, 10 -> 1) and every other
class is dropped. Images are not copied: each split's ``images`` folder is a symlink
to the original one, so Ultralytics picks up the filtered ``labels`` folder next to it.

Usage:
    python -m src.data.filter_labels --src data/cadot --dst data/cadot_2cls
"""

import argparse
import os
from pathlib import Path

from tqdm import tqdm

# Original CADOT id -> new id
CLASS_MAPPING = {1: 0, 10: 1}  # building, small vehicle


def filter_split(src: Path, dst: Path) -> None:
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    images_link = dst / "images"
    if not images_link.exists():
        images_link.symlink_to(os.path.relpath(src / "images", dst), target_is_directory=True)

    src_labels = src / "labels"
    if not src_labels.exists():  # e.g. the test split has no public annotations
        print(f"{src.name}: no labels, images linked only")
        return

    kept_objects, empty_files = 0, 0
    label_files = sorted(src_labels.glob("*.txt"))
    for label_file in tqdm(label_files, desc=src.name):
        lines = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if parts and int(parts[0]) in CLASS_MAPPING:
                lines.append(" ".join([str(CLASS_MAPPING[int(parts[0])]), *parts[1:]]))
        # Images without any kept object stay in the split as background samples.
        (dst / "labels" / label_file.name).write_text("\n".join(lines) + ("\n" if lines else ""))
        kept_objects += len(lines)
        empty_files += not lines

    print(f"{src.name}: {len(label_files)} files, {kept_objects} objects kept, {empty_files} background images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("data/cadot"), help="original 14-class dataset root")
    parser.add_argument("--dst", type=Path, default=Path("data/cadot_2cls"), help="output dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    for split in args.splits:
        if (args.src / split).exists():
            filter_split(args.src / split, args.dst / split)
        else:
            print(f"{split}: not found in {args.src}, skipped")


if __name__ == "__main__":
    main()
