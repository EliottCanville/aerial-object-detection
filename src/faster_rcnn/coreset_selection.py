"""Rank images by visual diversity with k-Center Greedy (core-set active learning).

Every image is embedded with the backbone of a trained Faster R-CNN: ResNet-50 C5
feature map (2048 channels) followed by global average pooling. k-Center Greedy
(Sener & Savarese, ICLR 2018) then repeatedly picks the image farthest from the
already-selected set. The full ordering is saved, so the first N ids form the most
diverse subset of size N for any budget.

Output format (consumed by ``src.data.build_active_learning_subset``):
    {"sorted_image_ids": [...], "description": "..."}

Usage:
    python -m src.faster_rcnn.coreset_selection \
        --checkpoint weights/fasterrcnn_r50fpn_cadot_2cls_full.pth \
        --images-dir data/cadot_coco/valid --out experiments/02_active_learning_500/coreset_ranking.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.faster_rcnn.dataset import BinaryCocoDataset, collate_fn
from src.faster_rcnn.train import build_model


@torch.no_grad()
def extract_features(model, loader, device) -> tuple[np.ndarray, list[int]]:
    """Return one 2048-d embedding per image and the matching COCO image ids."""
    features, image_ids = [], []
    for images, targets in tqdm(loader, desc="Extracting features"):
        # Same resize + normalization as the detector applies internally.
        batch = model.transform([img.to(device) for img in images])[0].tensors
        c5 = model.backbone.body(batch)["3"]  # [B, 2048, H/32, W/32]
        features.append(F.adaptive_avg_pool2d(c5, 1).flatten(1).cpu().numpy())
        image_ids += [int(t["image_id"]) for t in targets]
    return np.concatenate(features), image_ids


def k_center_greedy(features: np.ndarray, k: int | None = None, seed: int = 0) -> list[int]:
    """Indices of ``k`` points chosen by farthest-point traversal (all points if ``k`` is None).

    ``min_dist[i]`` caches the distance from point i to its nearest selected point, so
    each iteration costs one O(n) distance computation instead of O(n * |selected|).
    """
    n = len(features)
    k = n if k is None else min(k, n)
    first = int(np.random.default_rng(seed).integers(n))

    selected = [first]
    min_dist = np.linalg.norm(features - features[first], axis=1)
    min_dist[first] = -np.inf
    for _ in tqdm(range(k - 1), desc="k-Center Greedy"):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(features - features[idx], axis=1))
        min_dist[idx] = -np.inf
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Faster R-CNN checkpoint (.pth)")
    parser.add_argument("--images-dir", type=Path, required=True, help="split folder with its COCO annotations")
    parser.add_argument("--annotations", type=Path, default=None,
                        help="COCO file (default: <images-dir>/_annotations.coco.json)")
    parser.add_argument("--out", type=Path, default=Path("coreset_ranking.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="only embed the first N images (debugging on CPU)")
    parser.add_argument("--seed", type=int, default=0, help="seed for the first, random pick")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    annotations = args.annotations or args.images_dir / "_annotations.coco.json"
    dataset = BinaryCocoDataset(json.loads(annotations.read_text()), args.images_dir, transform=transforms.ToTensor())
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    features, image_ids = extract_features(model, loader, device)
    print(f"Feature matrix: {features.shape}")
    order = k_center_greedy(features, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "sorted_image_ids": [image_ids[i] for i in order],
        "description": "Images sorted by diversity (k-center greedy)",
    }, indent=2))
    print(f"Ranking of {len(order)} images written to {args.out}")


if __name__ == "__main__":
    main()
