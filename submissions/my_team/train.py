import argparse
import os
import sys
from pathlib import Path
import numpy as np

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base_model import ImageNetSubset
from model import ModelArchitecture
from augmentations import build_train_transforms, build_eval_transforms, apply_cutmix

DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = SCRIPT_DIR / "weights.joblib"
PLOT_OUTPUT = SCRIPT_DIR / "training_curves.png"

TRAIN_SPLIT = "train_split"
VAL_SPLIT = "val_split"

BATCH_SIZE = 192
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = min(8, os.cpu_count() or 1)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, desc=""):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            if is_training and np.random.rand() < 0.5:
                images, target_a, target_b, lam = apply_cutmix(images, labels)
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    logits = model(images)
                    loss = lam * criterion(logits, target_a) + (1 - lam) * criterion(logits, target_b)
            else:
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total

def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                         help=f"Load existing weights from {OUTPUT} before training instead of starting fresh")
    parser.add_argument("--save-best", choices=["acc", "loss"], default="acc",
                         help="Checkpoint metric: keep the epoch with the highest val_acc (default) "
                              "or the lowest val_loss")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Using device: {DEVICE}")
    train_transform = build_train_transforms()
    eval_transform = build_eval_transforms()

    train_ds = ImageNetSubset(DATA_ROOT, TRAIN_SPLIT, transform=train_transform)
    val_ds = ImageNetSubset(DATA_ROOT, VAL_SPLIT, transform=eval_transform)

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    model = ModelArchitecture(num_classes=20).to(DEVICE)

    if args.resume:
        if not OUTPUT.exists():
            raise FileNotFoundError(f"--resume given but no checkpoint found at {OUTPUT}")
        model.load_state_dict(joblib.load(OUTPUT))
        print(f"Resumed weights from {OUTPUT}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=DEVICE.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    track_loss = args.save_best == "loss"
    best_metric = float("inf") if track_loss else 0.0
    best_state = None

    def is_better(val_loss, val_acc, best_metric):
        return val_loss < best_metric if track_loss else val_acc > best_metric

    if args.resume:
        resumed_val_loss, resumed_val_acc = run_epoch(model, val_loader, criterion, desc="Resumed checkpoint [val]")
        best_metric = resumed_val_loss if track_loss else resumed_val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Resumed checkpoint val_loss={resumed_val_loss:.4f} val_acc={resumed_val_acc:.4f} "
              f"(new epochs must beat val_{args.save_best}={best_metric:.4f} to be saved)")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, desc=f"Epoch {epoch}/{EPOCHS} [train]"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, desc=f"Epoch {epoch}/{EPOCHS} [val]"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]

        tqdm.write(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | lr {lr_after:.2e}"
        )
        if lr_after < lr_before:
            tqdm.write(f"  > val_loss plateaued, reducing LR {lr_before:.2e} -> {lr_after:.2e}")

        if is_better(val_loss, val_acc, best_metric):
            best_metric = val_loss if track_loss else val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    final_state = best_state if best_state is not None else model.state_dict()
    joblib.dump(final_state, OUTPUT)
    print(f"Saved weights to {OUTPUT} (best val_{args.save_best}={best_metric:.4f})")

    plot_history(history, PLOT_OUTPUT)
    print(f"Saved training curves to {PLOT_OUTPUT}")

if __name__ == "__main__":
    main()