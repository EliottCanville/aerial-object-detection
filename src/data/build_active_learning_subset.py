"""Materialize the active-learning subset from a k-Center Greedy ranking.

The ranking (``coreset_ranking.json``, produced by ``src.faster_rcnn.coreset_selection``)
lists image indices from the most to the least informative. As in the original
experiment, index ``i`` refers to the i-th image of the source split sorted by file name.
The first ``--n`` images and their labels are copied to ``--dst``.

Usage (subset used in experiments 02 and 03):
    python -m src.data.build_active_learning_subset --src data/cadot_2cls/valid --dst data/active500_2cls
    python -m src.data.build_active_learning_subset --src data/cadot/valid --dst data/active500_14cls
"""

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from src.utils import count_instances, list_images

DEFAULT_RANKING = Path("experiments/02_active_learning_500/coreset_ranking.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="split folder containing images/ and labels/")
    parser.add_argument("--dst", type=Path, required=True, help="output folder")
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--n", type=int, default=500, help="number of images to keep")
    args = parser.parse_args()

    ranked_ids = json.loads(args.ranking.read_text())["sorted_image_ids"][: args.n]
    images = list_images(args.src / "images")
    selected = [images[i] for i in ranked_ids if i < len(images)]
    print(f"Selected {len(selected)} / {len(images)} images from {args.src}")

    (args.dst / "images").mkdir(parents=True, exist_ok=True)
    (args.dst / "labels").mkdir(parents=True, exist_ok=True)
    for image in tqdm(selected, desc="Copying"):
        shutil.copy2(image, args.dst / "images" / image.name)
        label = args.src / "labels" / f"{image.stem}.txt"
        target = args.dst / "labels" / label.name
        if label.exists():
            shutil.copy2(label, target)
        else:
            target.touch()  # background image

    instances, _ = count_instances(args.dst / "labels")
    print(f"Subset written to {args.dst}: {sum(instances.values()):,} objects, per class {dict(sorted(instances.items()))}")


if __name__ == "__main__":
    main()
