import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base_model import ImageNetSubset
from model import ModelArchitecture

DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = SCRIPT_DIR / "weights.joblib"
PLOT_OUTPUT = SCRIPT_DIR / "training_curves.png"

TRAIN_SPLIT = "train_split"
VAL_SPLIT = "val_split"

IMAGE_SIZE = 224
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = min(8, os.cpu_count() or 1)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform

def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
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
# first model # working
def main():
    print(f"Using device: {DEVICE}")
    train_transform, eval_transform = build_transforms()

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, desc=f"Epoch {epoch}/{EPOCHS} [train]"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, desc=f"Epoch {epoch}/{EPOCHS} [val]"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        tqdm.write(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    final_state = best_state if best_state is not None else model.state_dict()
    joblib.dump(final_state, OUTPUT)
    print(f"Saved weights to {OUTPUT} (best val_acc={best_val_acc:.4f})")

    plot_history(history, PLOT_OUTPUT)
    print(f"Saved training curves to {PLOT_OUTPUT}")

if __name__ == "__main__":
    main()