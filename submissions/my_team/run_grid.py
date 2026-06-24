"""
Full grid runner: for each (fraction, arch) combination, train for a fixed
number of epochs, save weights + training curves, then run the official
self-evaluation stress-test suite against the trained model and save its log
plus a per-run summary doc.

Usage:
  python run_grid.py
  python run_grid.py --fractions 0.5 0.8 --epochs 100 --archs convnext resnet50
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from base_model import ImageNetSubset
from augmentations import build_train_transforms, build_eval_transforms
from train import DEVICE
import run_experiment as exp
import self_evaluation as se

DATA_ROOT = exp.DATA_ROOT
OUTPUT_ROOT = SCRIPT_DIR / "experiments"


class TrainedPredictor:
    """Wraps a trained nn.Module so it satisfies self_evaluation's `model.predict(x)` interface."""

    def __init__(self, net):
        self.net = net.to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, x):
        logits = self.net(x.to(DEVICE, non_blocking=True))
        return logits.argmax(dim=1).cpu().numpy()


def run_self_eval(net, suites, arch, fraction, epochs, log_path):
    predictor = TrainedPredictor(net)
    scores = {}
    for suite_name, loader in suites.items():
        acc = se.evaluate(predictor, loader, desc=f"{arch}-f{fraction} | {suite_name.strip()}")
        scores[suite_name] = acc
        print(f"  > {suite_name}: {acc:.4f}")

    base_acc = scores.get("Base (Clean)  ", 0.0)
    stress_scores = [acc for name, acc in scores.items() if name != "Base (Clean)  "]
    avg_stress = sum(stress_scores) / len(stress_scores) if stress_scores else 0.0
    final_score = 0.5 * base_acc + 0.5 * avg_stress

    print(f"  Base Accuracy:   {base_acc:.4f}")
    print(f"  Avg Stress Acc:  {avg_stress:.4f}")
    print(f"  Final Score:     {final_score:.4f}")

    lines = [
        f"Self-evaluation for arch={arch} fraction={fraction} epochs={epochs}",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    lines += [f"  > {name}: {acc:.4f}" for name, acc in scores.items()]
    lines += [
        "",
        f"Base Accuracy:   {base_acc:.4f}",
        f"Avg Stress Acc:  {avg_stress:.4f}",
        f"Final Score:     {final_score:.4f}",
    ]
    log_path.write_text("\n".join(lines) + "\n")

    scores["Avg Stress"] = avg_stress
    scores["Final Score"] = final_score
    return scores


def write_summary(out_dir, arch, fraction, epochs, args, best_val_acc, elapsed, eval_scores, weights_path, curves_path):
    doc_path = out_dir / f"{arch}_summary.md"
    lines = [
        f"# {arch} — fraction={fraction}, epochs={epochs}",
        "",
        "## Training config",
        f"- batch_size: {args.batch_size}",
        f"- lr: {args.lr}",
        f"- weight_decay: {args.weight_decay}",
        f"- seed: {args.seed}",
        f"- best val_acc (training): {best_val_acc:.4f}",
        f"- training time: {elapsed/60:.1f} min",
        f"- weights: {weights_path.name}",
        f"- training curves: {curves_path.name}",
        "",
        "## Self-evaluation (stress-test suites)",
    ]
    for name, acc in eval_scores.items():
        if name not in ("Avg Stress", "Final Score"):
            lines.append(f"- {name.strip()}: {acc:.4f}")
    lines += [
        f"- **Avg Stress Acc**: {eval_scores['Avg Stress']:.4f}",
        f"- **Final Score**: {eval_scores['Final Score']:.4f}",
    ]
    doc_path.write_text("\n".join(lines) + "\n")
    return doc_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--archs", nargs="+", choices=["convnext", "resnet50"], default=["convnext", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Using device: {DEVICE}")
    grid_start = time.time()

    train_ds_full = ImageNetSubset(DATA_ROOT, exp.TRAIN_SPLIT, transform=build_train_transforms())
    val_ds = ImageNetSubset(DATA_ROOT, exp.VAL_SPLIT, transform=build_eval_transforms())

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    print("Building self-evaluation stress-test suites (shared across all runs)...")
    suites = se.load_evaluation_suites()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    grid_rows = []

    for fraction in args.fractions:
        args.fraction = fraction  # train_one_arch reads this for its log header
        train_ds = exp.make_fraction_subset(train_ds_full, fraction, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
        print(f"\n##### fraction={fraction} -> {len(train_ds)}/{len(train_ds_full)} train images #####")

        out_dir = OUTPUT_ROOT / f"frac{fraction:g}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for arch in args.archs:
            best_val_acc, elapsed = exp.train_one_arch(arch, args, train_loader, val_loader, out_dir)

            weights_path = out_dir / f"{arch}_weights.joblib"
            curves_path = out_dir / f"{arch}_curves.png"

            net = exp.build_model(arch)
            net.load_state_dict(joblib.load(weights_path))

            print(f"\n--- Self-evaluating {arch} (fraction={fraction}) ---")
            eval_scores = run_self_eval(
                net, suites, arch, fraction, args.epochs, out_dir / f"{arch}_self_eval.log"
            )
            doc_path = write_summary(
                out_dir, arch, fraction, args.epochs, args,
                best_val_acc, elapsed, eval_scores, weights_path, curves_path,
            )
            print(f"[{arch} f={fraction}] summary -> {doc_path}")

            grid_rows.append({
                "fraction": fraction,
                "arch": arch,
                "train_best_val_acc": best_val_acc,
                "train_time_min": elapsed / 60,
                "self_eval_base_acc": eval_scores.get("Base (Clean)  ", 0.0),
                "self_eval_avg_stress": eval_scores["Avg Stress"],
                "self_eval_final_score": eval_scores["Final Score"],
            })

    summary_csv = OUTPUT_ROOT / "grid_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    total_elapsed = time.time() - grid_start
    print("\n" + "=" * 70)
    print(f"Grid results (total time {total_elapsed/60:.1f} min)")
    print("=" * 70)
    for row in grid_rows:
        print(
            f"  frac={row['fraction']:<4} {row['arch']:10s} "
            f"train_val_acc={row['train_best_val_acc']:.4f}  "
            f"final_score={row['self_eval_final_score']:.4f}  "
            f"time={row['train_time_min']:.1f}min"
        )
    print(f"\nFull grid summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
