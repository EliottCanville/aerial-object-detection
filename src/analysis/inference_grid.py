"""Tile annotated prediction images (e.g. from ``src.yolo.predict``) into a single figure.

Usage:
    python -m src.analysis.inference_grid runs/predict/14cls/pred_*.jpg --cols 3 \
        --titles "Multi-class" "Swimming pool" "Football field" "Roundabout" "Tennis court" "Playground" \
        --out report/figures/04_oversampled/inference_examples.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--titles", nargs="*", default=None, help="one caption per image")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--tile-size", nargs=2, type=int, default=(400, 430), metavar=("W", "H"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tiles = []
    for idx, path in enumerate(args.images):
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Cannot read {path}")
        image = cv2.resize(image, tuple(args.tile_size))
        if args.titles and idx < len(args.titles):
            # White outline under black text keeps captions readable on any background.
            cv2.putText(image, args.titles[idx], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, args.titles[idx], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        tiles.append(image)

    # Pad the last row with blank tiles so every row has the same width.
    blank = np.full_like(tiles[0], 255)
    tiles += [blank] * (-len(tiles) % args.cols)
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"Grid of {len(args.images)} images saved to {args.out}")


if __name__ == "__main__":
    main()
