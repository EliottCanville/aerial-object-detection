"""Train a YOLOv8 detector from an experiment config.

An experiment config (``configs/experiments/*.yaml``) holds the starting weights
(``model``) plus any Ultralytics training argument. After training, the best
checkpoint is evaluated on the validation split and the headline metrics are
written to ``<run_dir>/metrics.json``.

Usage (from the repository root):
    python -m src.yolo.train configs/experiments/03_oversampled_14classes.yaml
    python -m src.yolo.train configs/experiments/01_baseline_full_dataset.yaml --device cpu --epochs 1
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment config (YAML)")
    parser.add_argument("--device", default=None, help="override the device, e.g. 0, cpu, mps")
    parser.add_argument("--epochs", type=int, default=None, help="override the number of epochs (quick tests)")
    args = parser.parse_args()

    train_args = yaml.safe_load(args.config.read_text())
    weights = train_args.pop("model", "yolov8n.pt")
    if args.device is not None:
        train_args["device"] = args.device
    if args.epochs is not None:
        train_args["epochs"] = args.epochs

    print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(weights)
    model.train(**train_args)

    # After training, `model` holds the best checkpoint.
    run_dir = Path(model.trainer.save_dir)
    metrics = model.val(data=train_args["data"], split="val", project=str(run_dir), name="val", exist_ok=True)
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n".join(f"{k:>10}: {v:.4f}" for k, v in summary.items()))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
