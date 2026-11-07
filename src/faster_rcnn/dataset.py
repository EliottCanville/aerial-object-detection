"""COCO-format CADOT dataset restricted to the building / small vehicle task.

Targets follow the torchvision detection convention: ``boxes`` in absolute
``[x1, y1, x2, y2]`` pixels and ``labels`` in ``1..N`` (0 is reserved for background).
"""

from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# COCO category name -> torchvision label
CATEGORY_TO_LABEL = {"building": 1, "small vehicle": 2}
NUM_CLASSES = len(CATEGORY_TO_LABEL) + 1  # + background


class BinaryCocoDataset(Dataset):
    """Images of a COCO annotation dict, keeping only building and small vehicle boxes.

    Images without any kept annotation are still returned (empty targets), so the
    detector also sees background-only tiles.
    """

    def __init__(self, coco: dict, images_dir: str | Path, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.images = coco["images"]

        category_map = {
            c["id"]: CATEGORY_TO_LABEL[c["name"]] for c in coco["categories"] if c["name"] in CATEGORY_TO_LABEL
        }
        self.annotations = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in category_map:
                self.annotations[ann["image_id"]].append((category_map[ann["category_id"]], ann["bbox"]))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")

        boxes, labels = [], []
        for label, (x, y, w, h) in self.annotations[info["id"]]:
            if w > 0 and h > 0:  # torchvision rejects degenerate boxes
                boxes.append([x, y, x + w, y + h])
                labels.append(label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(info["id"]),
        }
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def collate_fn(batch):
    """Keep images and targets as tuples: detection samples have variable sizes."""
    return tuple(zip(*batch))
