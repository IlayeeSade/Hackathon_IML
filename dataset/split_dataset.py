"""
Splits dataset/train_original into a training subset and a validation subset.

Per-class images are shuffled (with a fixed seed for reproducibility) and
divided according to TRAIN_RATIO. The result is written as two new
directory trees (symlinks by default, so no images are duplicated on
disk):

    dataset/train_split/<class_name>/*.jpg
    dataset/val_split/<class_name>/*.jpg

This layout matches base_model.ImageNetSubset(root, split), so the splits
can be loaded directly, e.g.:

    train_ds = ImageNetSubset(Path("dataset"), "train_split")
    val_ds   = ImageNetSubset(Path("dataset"), "val_split")

Usage:
    python dataset/split_dataset.py
    python dataset/split_dataset.py --train-ratio 0.2 --seed 42
    python dataset/split_dataset.py --copy   # copy files instead of symlinking
"""

import argparse
import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.5  # fraction of each class's images assigned to the training split
SEED = 42

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.JPEG", "*.png")


def split_class(class_dir: Path, train_ratio: float, seed: int):
    images = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(class_dir.glob(pattern))
    images = sorted(images)

    rng = random.Random(seed)
    rng.shuffle(images)

    cutoff = int(len(images) * train_ratio)
    return images[:cutoff], images[cutoff:]


def populate_split(class_name: str, image_paths: list[Path], split_root: Path, copy: bool):
    out_dir = split_root / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        link_path = out_dir / img_path.name
        if link_path.exists() or link_path.is_symlink():
            continue
        if copy:
            shutil.copy2(img_path, link_path)
        else:
            link_path.symlink_to(img_path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / "train_original",
                         help="Directory containing per-class image folders (default: dataset/train_original)")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent,
                         help="Where to create train_split/ and val_split/ (default: dataset/)")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO,
                         help=f"Fraction of each class assigned to the training split (default: {TRAIN_RATIO})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Shuffle seed (default: {SEED})")
    parser.add_argument("--copy", action="store_true",
                         help="Copy image files instead of creating symlinks")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    train_root = args.output_root / "train_split"
    val_root = args.output_root / "val_split"

    class_dirs = sorted(p for p in args.source.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found in {args.source}")

    print(f"{'Class':<20} {'Train':>6} {'Val':>6} {'Total':>6}")
    print("-" * 44)

    total_train, total_val = 0, 0
    for class_dir in class_dirs:
        train_images, val_images = split_class(class_dir, args.train_ratio, args.seed)
        populate_split(class_dir.name, train_images, train_root, args.copy)
        populate_split(class_dir.name, val_images, val_root, args.copy)

        total_train += len(train_images)
        total_val += len(val_images)
        print(f"{class_dir.name:<20} {len(train_images):>6} {len(val_images):>6} {len(train_images) + len(val_images):>6}")

    print("-" * 44)
    print(f"{'TOTAL':<20} {total_train:>6} {total_val:>6} {total_train + total_val:>6}")
    print(f"\nWrote train split to {train_root}")
    print(f"Wrote val split to {val_root}")


if __name__ == "__main__":
    main()
