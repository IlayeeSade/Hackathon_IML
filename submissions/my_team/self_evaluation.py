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
import os
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
from tqdm import tqdm
import seaborn as sns

NUM_WORKERS = os.cpu_count() // 2
PIN_MEMORY = torch.cuda.is_available() or torch.backends.mps.is_available()

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
    build_auto_transforms, # <- התוספת החדשה שלך
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
NUM_WORKERS = min(4, os.cpu_count() or 1)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# ──────────────────────────────────────────────────────────────────────────────

class ImageNetSubset(Dataset):
    """Loads a fraction of the 20 target classes from data/dataset/validation for faster evaluation."""

    def __init__(self, root: Path, split: str = "validation", transform=None, fraction=0.1):
        self.transform = transform
        self.samples = []

        split_root = root / split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Validation folder not found: {split_root}\n"
                f"Expected structure: {root}/validation/<class_name>/*.jpg"
            )

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

            local_idx = HF_INDEX_TO_IDX[hf_idx]

            # אוספים את כל התמונות במחלקה וממיינים
            all_images = sorted(class_dir.glob("*.jpg"))
            
            # מקטינים את כמות הדאטה: לוקחים רק 1 מכל 10 תמונות (אם fraction=0.1)
            step = max(1, int(1 / fraction))
            subset_images = all_images[::step]
            
            for img_path in subset_images:
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
    print(f"Preparing evaluation suites (Workers: {NUM_WORKERS}, Pin Memory: {PIN_MEMORY})...")
    
    # פונקציית עזר קטנה כדי לא לשכפל את כל הפרמטרים
    def make_loader(transform, shuffle=False):
        dataset = ImageNetSubset(DATA_ROOT, transform=transform)
        return DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=shuffle, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=True if NUM_WORKERS > 0 else False # שומר את התהליכים חיים בין באצ'ים
        )

    suites = {
        "Base (Clean)  ": make_loader(build_eval_transforms(), shuffle=False),
        "Geometric     ": make_loader(build_geometric_stress_transforms(), shuffle=False),
        "Color/Photo   ": make_loader(build_color_stress_transforms(), shuffle=False),
        "Noise/Occlude ": make_loader(build_noise_stress_transforms(), shuffle=False),
        "AutoAugment   ": make_loader(build_auto_transforms(), shuffle=False),
        "Ultimate Combo": make_loader(build_stress_transforms(), shuffle=True)
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
        if hasattr(model, "net"):
            model.net.to(DEVICE)
            model.net.eval()

    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)

    return model


# ── evaluation & visualization ────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, return_preds=False, desc=""):
    correct = 0
    total   = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for x, y in pbar:
        x = x.to(DEVICE, non_blocking=True)
        preds = model.predict(x)

        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        y_np = y.cpu().numpy()

        correct += (preds == y_np).sum()
        total   += y.size(0)

        if return_preds:
            all_preds.extend(preds)
            all_labels.extend(y_np)

        pbar.set_postfix(acc=correct / total)

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

    preds = model.predict(x.to(DEVICE, non_blocking=True))
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

def plot_accuracy_bar_chart(team_name, scores_dict):
    """מצייר גרף עמודות של אחוזי הדיוק לכל סוג של מבחן לחץ"""
    suites = list(scores_dict.keys())
    accuracies = list(scores_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(suites, accuracies, color=['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0'])

    plt.title(f'Robustness Evaluation Accuracy - {team_name}', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1.05) # מקבעים את ציר ה-Y בין 0 ל-100%
    plt.xticks(rotation=15, fontsize=12)

    # הוספת המספר המדויק מעל כל עמודה
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}")
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
                acc = evaluate(model, loader, desc=f"{team_dir.name} | {suite_name.strip()}")
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
    print("\n" + "="*80)
    print("--- Detailed Leaderboard (Ranked by Robustness Final Score) ---")
    print("="*80)
    
    valid_teams = {k: v for k, v in results.items() if v is not None}
    
    # חישוב הציון הסופי לכל קבוצה
    for team, scores in valid_teams.items():
        base_acc = scores.get("Base (Clean)  ", 0)
        
        # אוספים את כל הציונים שהם לא סט הבסיס
        stress_scores = [acc for name, acc in scores.items() if name != "Base (Clean)  "]
        avg_stress_acc = sum(stress_scores) / len(stress_scores) if stress_scores else 0
        
        # הציון המשוקלל: 50% בסיס, 50% ממוצע מבחני לחץ
        final_score = (0.5 * base_acc) + (0.5 * avg_stress_acc)
        scores["Avg Stress"] = avg_stress_acc
        scores["Final Score"] = final_score

    # מיון הקבוצות לפי הציון הסופי החדש!
    ranked_teams = sorted(valid_teams.items(), key=lambda item: item[1]["Final Score"], reverse=True)
    
    for rank, (team, scores) in enumerate(ranked_teams, start=1):
        print(f"{rank}. Team: {team}")
        
        # הדפסת כל התוצאות הבדידות
        for suite_name, acc in scores.items():
            if suite_name not in ["Avg Stress", "Final Score"]:
                print(f"    - {suite_name}: {acc:.4f}")
        
        print("-" * 40)
        print(f"    * Base Accuracy:    {scores['Base (Clean)  ']:.4f}")
        print(f"    * Avg Stress Acc:   {scores['Avg Stress']:.4f}")
        print(f"    ⭐ FINAL Score:      {scores['Final Score']:.4f}\n")

    for team, res in results.items():
        if res is None:
            print(f"--  Team: {team} (FAILED)")

    if best_model is not None:
        print(f"\nGenerating visual reports for {best_team}...")
        
        # --- הגרף החדש: עמודות דיוק ---
        print(f"Generating Bar Chart for {best_team}...")
        plot_accuracy_bar_chart(best_team, results[best_team])
        
        # (מכאן זה הקוד שכבר יש לך ממקודם)
        ultimate_loader = suites["Ultimate Combo"]
        acc, y_true, y_pred = evaluate(best_model, ultimate_loader, return_preds=True)
        
        visualize_stress_test(best_model, ultimate_loader, num_images=5)
        plot_confusion_matrix(y_true, y_pred, title=f"Confusion Matrix: Ultimate Combo ({best_team})")

if __name__ == "__main__":
    main()