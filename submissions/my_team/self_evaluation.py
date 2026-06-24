"""
Expected submissions layout:
  submissions/
    team_a/
      train.py
      model.py
      predict.py
      weights.joblib

Run:
  python evaluate.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# מייבאים את כל האוגמנטציות שבנית
from augmentations import (
    build_eval_transforms,
    build_geometric_stress_transforms,
    build_color_stress_transforms,
    build_noise_stress_transforms,
    build_stress_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD
)

from labels import (
    HF_INDEX_TO_NAME,
    HF_INDEX_TO_IDX,
    TARGET_HF_INDICES,
)

# ── editable ──────────────────────────────────────────────────────────────────
DATA_ROOT = PROJECT_ROOT / "dataset"   # contains val_split/
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
BATCH_SIZE = 64
WEIGHTS_FILENAME = "weights.joblib"
# ──────────────────────────────────────────────────────────────────────────────

class ImageNetSubset(Dataset):
    """Loads the 20 target classes from data/dataset/validation."""

    def __init__(self, root: Path, split: str = "val_split", transform=None):
        self.transform = transform
        self.samples = []

        split_root = root / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Validation folder not found: {split_root}\n"
                f"Expected structure: {root}/val_split/<class_name>/*.jpg"
            )

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

            local_idx = HF_INDEX_TO_IDX[hf_idx]

            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, local_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ── transforms & loaders ──────────────────────────────────────────────────────

def load_evaluation_suites():
    """
    בונה מילון של DataLoaders עבור כל סוגי מבחני הלחץ שהוגדרו.
    """
    print("Preparing evaluation suites...")
    suites = {
        "Base (Clean)  ": DataLoader(ImageNetSubset(DATA_ROOT, transform=build_eval_transforms()), batch_size=BATCH_SIZE, shuffle=False),
        "Geometric     ": DataLoader(ImageNetSubset(DATA_ROOT, transform=build_geometric_stress_transforms()), batch_size=BATCH_SIZE, shuffle=False),
        "Color/Photo   ": DataLoader(ImageNetSubset(DATA_ROOT, transform=build_color_stress_transforms()), batch_size=BATCH_SIZE, shuffle=False),
        "Noise/Occlude ": DataLoader(ImageNetSubset(DATA_ROOT, transform=build_noise_stress_transforms()), batch_size=BATCH_SIZE, shuffle=False),
        "Ultimate Combo": DataLoader(ImageNetSubset(DATA_ROOT, transform=build_stress_transforms()), batch_size=BATCH_SIZE, shuffle=True) # Shuffle for visuals
    }
    
    for name, loader in suites.items():
        print(f"  > Loaded '{name.strip()}' suite: {len(loader.dataset)} images.")
    print("")
    return suites


# ── submission loading ────────────────────────────────────────────────────────

def load_submission(team_dir: Path):
    predict_path = team_dir / "predict.py"
    model_path = team_dir / "model.py"
    weights_path = team_dir / WEIGHTS_FILENAME

    if not predict_path.exists():
        raise FileNotFoundError(f"Missing predict.py in {team_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.py in {team_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {WEIGHTS_FILENAME} in {team_dir}")

    sys.path.insert(0, str(team_dir))
    sys.modules.pop("model", None)

    try:
        spec = importlib.util.spec_from_file_location(
            f"{team_dir.name}_predict",
            predict_path,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import predict.py from {team_dir}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Model"):
            raise AttributeError(f"predict.py in {team_dir} must define a class named Model")

        model = module.Model()
        model.load(str(weights_path))

    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)

    return model


# ── evaluation & visualization ────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, return_preds=False):
    correct = 0
    total   = 0
    all_preds = []
    all_labels = []
    
    for x, y in loader:
        preds = model.predict(x)
        
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        y_np = y.cpu().numpy()
            
        correct += (preds == y_np).sum()
        total   += y.size(0)
        
        if return_preds:
            all_preds.extend(preds)
            all_labels.extend(y_np)
            
    acc = correct / total
    
    if return_preds:
        return acc, np.array(all_labels), np.array(all_preds)
    return acc

def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

@torch.no_grad()
def visualize_stress_test(model, loader, num_images=4, output_path=None):
    """Pulls a batch from the ultimate stress test and visualizes."""
    x, y = next(iter(loader))
    
    preds = model.predict(x)
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    
    y = y.cpu().numpy()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    fig.suptitle("Ultimate Combo Stress Test Predictions", fontsize=16)

    local_idx_to_name = {HF_INDEX_TO_IDX[k]: HF_INDEX_TO_NAME[k] for k in HF_INDEX_TO_IDX}

    for i in range(num_images):
        img_tensor = denormalize(x[i])
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        true_label = local_idx_to_name.get(y[i], "Unknown")
        pred_label = local_idx_to_name.get(preds[i], "Unknown")
        
        color = "green" if y[i] == preds[i] else "red"
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"True: {true_label[:10]}\nPred: {pred_label[:10]}", color=color)
        axes[i].axis("off")

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
        print(f"Saved stress-test visualization to {output_path}")
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", output_path=None):
    """מחשבת ומציירת מטריצת בלבול עם שמות המחלקות האמיתיים."""
    
    # שליפת השמות של המחלקות לפי הסדר הנכון מתוך המילונים שהוגדרו
    local_idx_to_name = {HF_INDEX_TO_IDX[k]: HF_INDEX_TO_NAME[k] for k in HF_INDEX_TO_IDX}
    class_names = [local_idx_to_name[i] for i in range(len(local_idx_to_name))]

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12)) # גודל שמתאים ל-20 מחלקות
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=18)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right') # סיבוב הטקסט כדי שלא יעלה אחד על השני
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
        print(f"Saved confusion matrix to {output_path}")
    plt.close()
# ── main ──────────────────────────────────────────────────────────────────────

def main():
    suites = load_evaluation_suites()

    team_dirs = sorted(d for d in SUBMISSIONS_DIR.iterdir() if d.is_dir())
    if not team_dirs:
        print(f"No submissions found in {SUBMISSIONS_DIR}/")
        sys.exit(1)

    results = {}
    best_model = None
    best_team = None

    for team_dir in team_dirs:
        print(f"Evaluating {team_dir.name}...")
        results[team_dir.name] = {}
        
        try:
            model = load_submission(team_dir)
            
            # מעבר על כל אחד מסטים של ה-Evaluation
            for suite_name, loader in suites.items():
                acc = evaluate(model, loader)
                results[team_dir.name][suite_name] = acc
                print(f"  > {suite_name}: {acc:.4f}")
                
            # הגדרת המודל לתצוגה הגרפית בסוף (הראשון שעובר בלי קריסה)
            if best_model is None:
                best_model = model
                best_team = team_dir.name
                
        except Exception as e:
            print(f"  > FAILED — {e}")
            results[team_dir.name] = None

    # הדפסת דוח מרכז יפה
    print("\n" + "="*70)
    print("--- Detailed Leaderboard ---")
    print("="*70)
    
    # מיון לפי הביצועים על סט הבסיס (הנקי)
    valid_teams = {k: v for k, v in results.items() if v is not None}
    ranked_teams = sorted(valid_teams.items(), key=lambda item: item[1]["Base (Clean)  "], reverse=True)
    
    for rank, (team, scores) in enumerate(ranked_teams, start=1):
        print(f"{rank}. Team: {team}")
        for suite_name, acc in scores.items():
            print(f"    - {suite_name}: {acc:.4f}")
        
        base_acc = scores["Base (Clean)  "]
        ultimate_acc = scores["Ultimate Combo"]
        print(f"    * Overall Drop (Base -> Ultimate): {base_acc - ultimate_acc:.4f}\n")

    for team, res in results.items():
        if res is None:
            print(f"--  Team: {team} (FAILED)")

    if best_model is not None:
            print(f"\nGenerating ultimate visual stress-test for {best_team}...")
            
            # קוראים לפונקציה המשודרגת ומבקשים לקבל גם את התחזיות
            ultimate_loader = suites["Ultimate Combo"]
            acc, y_true, y_pred = evaluate(best_model, ultimate_loader, return_preds=True)
            
            # 1. מציירים את 5 התמונות המעוותות (הפונקציה ממקודם)
            visualize_stress_test(
                best_model, ultimate_loader, num_images=5,
                output_path=SCRIPT_DIR / "stress_test_predictions.png",
            )

            # 2. מציירים את מטריצת הבלבול על כל הסט!
            print(f"Generating Confusion Matrix for {best_team}...")
            plot_confusion_matrix(
                y_true, y_pred, title=f"Confusion Matrix: Ultimate Combo ({best_team})",
                output_path=SCRIPT_DIR / f"confusion_matrix_{best_team}.png",
            )

if __name__ == "__main__":
    main()