"""Fine-tune a COCO-pretrained Faster R-CNN (ResNet-50 + FPN) on building / small vehicle.

Expects the COCO export of CADOT: ``<data-root>/<split>/_annotations.coco.json``
next to the split images. ``--annotations`` points to another annotation file of the
same split (e.g. a core-set subset). A checkpoint is written at the end of every epoch.

Usage:
    python -m src.faster_rcnn.train --data-root data/cadot_coco --out-dir runs/faster_rcnn
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.faster_rcnn.dataset import NUM_CLASSES, BinaryCocoDataset, collate_fn


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    """Faster R-CNN ResNet-50 FPN with its box predictor resized to ``num_classes``.

    ``pretrained=False`` builds an uninitialized model, meant to receive a checkpoint.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/cadot_coco"))
    parser.add_argument("--annotations", type=Path, default=None,
                        help="training annotation file (default: <data-root>/train/_annotations.coco.json)")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/faster_rcnn"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2, help="limited by GPU memory at 800px")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = args.data_root / "train"
    annotations = args.annotations or train_dir / "_annotations.coco.json"
    dataset = BinaryCocoDataset(json.loads(annotations.read_text()), train_dir, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=collate_fn)
    print(f"{len(dataset)} training images | device: {device}")

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # Standard torchvision detection recipe: small LR with momentum, light L2 regularization.
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # In train mode the model returns its RPN and ROI-head losses.
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}: mean loss = {avg_loss:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "average_loss_per_epoch": avg_loss,
            },
            args.out_dir / f"fasterrcnn_binary_epoch{epoch}.pth",
        )


if __name__ == "__main__":
    main()
