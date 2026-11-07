"""Helpers shared by the data-preparation and analysis scripts."""

from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES_FILE = REPO_ROOT / "configs" / "classes.txt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_class_names(path: Path = CLASSES_FILE) -> list[str]:
    """Read a ``<id> <name>`` file into a list indexed by class id."""
    names = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            class_id, name = line.split(maxsplit=1)
            names[int(class_id)] = name
    return [names[i] for i in range(len(names))]


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into ``(class_id, cx, cy, w, h)`` tuples (normalized coords)."""
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes


def count_instances(labels_dir: Path) -> tuple[Counter, Counter]:
    """Count, per class, the number of instances and the number of images containing it."""
    instances, images = Counter(), Counter()
    for label_file in Path(labels_dir).glob("*.txt"):
        classes = [box[0] for box in read_yolo_labels(label_file)]
        instances.update(classes)
        images.update(set(classes))
    return instances, images


def list_images(directory: Path) -> list[Path]:
    """Return the images of a directory, sorted by file name."""
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
