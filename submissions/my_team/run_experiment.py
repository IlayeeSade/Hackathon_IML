"""
Quick architecture comparison: train ConvNeXt-Tiny and/or ResNet-50 on a
fraction of the data for a given number of epochs.

Usage:
  python run_experiment.py --fraction 0.2 --epochs 5
  python run_experiment.py --fraction 0.5 --epochs 10 --arch resnet50
  python run_experiment.py --fraction 1.0 --epochs 20 --arch convnext --batch-size 64
"""
import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base_model import ImageNetSubset
from augmentations import build_train_transforms, build_eval_transforms
from train import run_epoch, plot_history, DEVICE  # reuse the existing training loop

DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = SCRIPT_DIR / "experiments"

TRAIN_SPLIT = "train_split"
VAL_SPLIT = "val_split"
NUM_CLASSES = 20


def build_model(arch: str) -> nn.Module:
    if arch == "convnext":
        backbone = models.convnext_tiny(weights=None)
        in_features = backbone.classifier[2].in_features
        backbone.classifier = nn.Sequential(
            backbone.classifier[0],  # LayerNorm
            backbone.classifier[1],  # Flatten
            nn.Dropout(p=0.3),
            nn.Linear(in_features, NUM_CLASSES),
        )
        return backbone

    if arch == "resnet50":
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, NUM_CLASSES),
        )
        return backbone

    raise ValueError(f"Unknown arch: {arch}")


def make_fraction_subset(dataset: ImageNetSubset, fraction: float, seed: int) -> Subset:
    """Stratified random subset: keeps `fraction` of each class's images."""
    if fraction >= 1.0:
        return Subset(dataset, range(len(dataset)))

    by_label = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        by_label[label].append(idx)

    rng = random.Random(seed)
    indices = []
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        k = max(1, round(len(label_indices) * fraction))
        indices.extend(label_indices[:k])

    return Subset(dataset, indices)


def train_one_arch(arch: str, args, train_loader, val_loader, out_dir: Path):
    print(f"\n=== Training {arch} | fraction={args.fraction} epochs={args.epochs} ===")
    model = build_model(arch).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=DEVICE.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, desc=f"[{arch}] Epoch {epoch}/{args.epochs} [train]"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, desc=f"[{arch}] Epoch {epoch}/{args.epochs} [val]"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(
            f"[{arch}] Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - start

    weights_path = out_dir / f"{arch}_weights.joblib"
    curves_path = out_dir / f"{arch}_curves.png"
    joblib.dump(best_state if best_state is not None else model.state_dict(), weights_path)
    plot_history(history, curves_path)

    print(f"[{arch}] done in {elapsed/60:.1f} min | best val_acc={best_val_acc:.4f}")
    print(f"[{arch}] weights -> {weights_path}")
    print(f"[{arch}] curves  -> {curves_path}")

    return best_val_acc, elapsed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=1.0,
                         help="Fraction of train_split to use, per class (0-1, default 1.0)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default 10)")
    parser.add_argument("--arch", choices=["convnext", "resnet50", "both"], default="both",
                         help="Which architecture(s) to train (default: both)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the fraction subsampling")
    parser.add_argument("--tag", default=None,
                         help="Subfolder name under experiments/ (default: auto from fraction/epochs)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0, 1.0]")

    print(f"Using device: {DEVICE}")

    train_ds_full = ImageNetSubset(DATA_ROOT, TRAIN_SPLIT, transform=build_train_transforms())
    val_ds = ImageNetSubset(DATA_ROOT, VAL_SPLIT, transform=build_eval_transforms())
    train_ds = make_fraction_subset(train_ds_full, args.fraction, seed=args.seed)
    print(f"Train subset: {len(train_ds)}/{len(train_ds_full)} images "
          f"({args.fraction:.0%}) | Val: {len(val_ds)} images")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    tag = args.tag or f"frac{args.fraction:g}_ep{args.epochs}"
    out_dir = OUTPUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    archs = ["convnext", "resnet50"] if args.arch == "both" else [args.arch]
    results = {}
    for arch in archs:
        best_val_acc, elapsed = train_one_arch(arch, args, train_loader, val_loader, out_dir)
        results[arch] = (best_val_acc, elapsed)

    print(f"\n=== Summary (fraction={args.fraction}, epochs={args.epochs}) ===")
    for arch, (best_val_acc, elapsed) in results.items():
        print(f"  {arch:10s} best_val_acc={best_val_acc:.4f}  time={elapsed/60:.1f} min")
    print(f"Results saved under {out_dir}")


if __name__ == "__main__":
    main()
