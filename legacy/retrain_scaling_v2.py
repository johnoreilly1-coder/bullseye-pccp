# ============================================================
# Scaling Experiment Run 2 — Retraining
# ============================================================
# Retrains DenseNet-121 on poison sets from craft_poison_scaling_v2
# for two target patients across five training dataset sizes.
#
# Configuration:
#   Surrogate:    BB ResNet-50 (no EOT)
#   Patients:     Val index 4 and val index 15
#   Sizes:        10k, 25k, 50k, 100k, 191k
#   Condition:    Realistic (aug) only
#   Seed:         123
#
# 191k results for val index 4 carried from Stage 5
# (BB single target, aug: 0.8214).
# 191k for val index 15 needs to be run fresh.
# ============================================================

import torch
import json
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.amp import GradScaler

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms
    )
    from models.densenet_model import build_densenet121, get_device
    from attacks.bullseye_polytope import PLEURAL_EFFUSION_IDX
    from experiments.retrain_poisoned import (
        PoisonedCheXpertDataset,
        train_epoch, validate_epoch,
        measure_attack_success,
    )
except ImportError:
    pass


TRAINING_SIZES  = [10_000, 25_000, 50_000, 100_000, 191_027]
POISON_RATE     = 0.05
TARGET_PATIENTS = [4, 12]

# 191k result for val index 4 carried from Stage 5 BB single aug
STAGE5_191K_VAL4 = {
    "n_total":        191_027,
    "n_poisons":      9_551,
    "poison_rate":    0.05,
    "augment":        True,
    "best_mean_auc":  0.8507,
    "auc_delta":      +0.0126,
    "pe_probability": 0.8104,
    "attack_success": False,
    "source":         "Stage 5 BB single aug (carried forward)",
}

CONFIG = {
    "train_csv":   "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":   "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":   "/home/ubuntu/poison-storage/chexpert",
    "poison_dir_root":    "/home/ubuntu/poison-storage/poison_scaling2",
    "checkpoint_dir_root":"/home/ubuntu/poison-storage/checkpoints_scaling2",
    "results_path":       "/home/ubuntu/poison-storage/scaling2_results/results.json",
    "batch_size":    32,
    "num_workers":   4,
    "epochs":        10,
    "lr":            1e-4,
    "weight_decay":  1e-5,
    "seed":          123,
}


def retrain_one(target_image, target_idx, poison_dir,
                checkpoint_dir, n_total, config, device):
    """Retrain DenseNet-121 for one patient/size combination."""

    val_transform   = get_transforms(mode="val",   image_size=224)
    train_transform = get_transforms(mode="train", image_size=224)

    # Load poison set
    rate_dir      = Path(poison_dir) / "rate_05pct"
    poison_images = torch.load(
        rate_dir / "poison_images.pt", map_location="cpu"
    )
    poison_labels = torch.load(
        rate_dir / "poison_labels.pt", map_location="cpu"
    )
    with open(rate_dir / "metadata.json") as f:
        meta = json.load(f)
    base_indices = meta["base_indices"]

    # Build poisoned training dataset
    full_train = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
        subset           = n_total if n_total < 191_027 else None,
    )

    train_dataset = PoisonedCheXpertDataset(
        clean_dataset   = full_train,
        poison_images   = poison_images,
        poison_labels   = poison_labels,
        base_indices    = base_indices,
        augment_poisons = True,   # realistic only
        val_transform   = val_transform,
    )

    val_dataset = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config["batch_size"],
        shuffle     = True,
        num_workers = config["num_workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config["num_workers"],
        pin_memory  = True,
    )

    print(f"    Train: {len(train_dataset):,} images "
          f"({len(train_loader):,} batches)")

    # Train
    model     = build_densenet121(
        num_classes=14, pretrained=True
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )
    scaler   = GradScaler('cuda')
    best_auc = 0.0

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_auc, _, val_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_auc)
        attack = measure_attack_success(model, target_image, device)

        print(f"    Ep {epoch:>2}/{config['epochs']}  "
              f"AUC {val_auc:.4f}  "
              f"PE {attack['pe_probability']:.4f}  "
              f"{'✓' if attack['attack_success'] else '✗'}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_auc":          val_auc,
                "attack":           attack,
            }, ckpt_dir / "best_model.pt")

    best_ckpt = torch.load(
        ckpt_dir / "best_model.pt", map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    final = measure_attack_success(model, target_image, device)

    return {
        "best_epoch":     int(best_ckpt["epoch"]),
        "best_mean_auc":  round(best_auc, 4),
        "auc_delta":      round(best_auc - 0.8381, 4),
        "pe_probability": final["pe_probability"],
        "attack_success": final["attack_success"],
    }


def run_scaling_retraining(config):
    print("=" * 60)
    print("SCALING EXPERIMENT RUN 2 — RETRAINING")
    print("=" * 60)
    print(f"Target patients:  {TARGET_PATIENTS}")
    print(f"Training sizes:   "
          f"{[f'{n//1000}k' for n in TRAINING_SIZES]}")
    print(f"Condition:        Realistic (aug) only")
    print(f"Epochs:           {config['epochs']}")

    device = get_device()

    results_path = Path(config["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for safe resumption
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"\nResuming — {len(all_results)} runs already done")
    else:
        all_results = []

    completed = {
        (r["val_index"], r["n_total"]) for r in all_results
    }

    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )

    for target_idx in TARGET_PATIENTS:
        target_image, _ = val_dataset[target_idx]

        for n_total in TRAINING_SIZES:
            if (target_idx, n_total) in completed:
                print(f"\nSkipping val {target_idx} "
                      f"{n_total//1000}k (already done)")
                continue

            # Carry 191k val4 from Stage 5
            if target_idx == 4 and n_total == 191_027:
                print(f"\nCarrying 191k val4 from Stage 5...")
                result = {
                    "val_index": 4,
                    **STAGE5_191K_VAL4
                }
                all_results.append(result)
                completed.add((4, 191_027))
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                continue

            size_label = f"{n_total // 1000}k"
            print(f"\n{'='*60}")
            print(f"Val {target_idx} — {size_label} "
                  f"({round(n_total*POISON_RATE):,} poisons)")
            print(f"{'='*60}")

            poison_dir = (
                Path(config["poison_dir_root"])
                / f"patient_{target_idx}"
                / f"size_{size_label}"
            )
            checkpoint_dir = (
                Path(config["checkpoint_dir_root"])
                / f"patient_{target_idx}"
                / f"size_{size_label}"
            )

            t0 = time.time()
            retrain_result = retrain_one(
                target_image, target_idx,
                poison_dir, checkpoint_dir,
                n_total, config, device
            )
            elapsed = round((time.time() - t0) / 60, 1)

            result = {
                "val_index":   target_idx,
                "n_total":     n_total,
                "n_poisons":   round(n_total * POISON_RATE),
                "poison_rate": POISON_RATE,
                "augment":     True,
                "time_mins":   elapsed,
                **retrain_result
            }
            all_results.append(result)
            completed.add((target_idx, n_total))

            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"\n  Result: PE prob = "
                  f"{retrain_result['pe_probability']:.4f}  "
                  f"{'✓' if retrain_result['attack_success'] else '✗'}  "
                  f"AUC {retrain_result['best_mean_auc']:.4f}  "
                  f"{elapsed}min")

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SCALING EXPERIMENT RUN 2 COMPLETE")
    print(f"{'='*65}")
    print(f"{'Patient':<10}{'Size':<8}{'Mean AUC':<12}"
          f"{'PE prob':<12}{'Attack':<10}")
    print("-" * 65)
    print(f"{'Baseline':<10}{'—':<8}{'0.8381':<12}"
          f"{'~0.93':<12}{'—':<10}")
    print("-" * 65)

    for r in sorted(all_results,
                    key=lambda x: (x["val_index"], x["n_total"])):
        print(f"Val {r['val_index']:<6}"
              f"{r['n_total']//1000}k{'':<4}"
              f"{r['best_mean_auc']:<12}"
              f"{r['pe_probability']:<12.4f}"
              f"{'✓ YES' if r['attack_success'] else '✗ NO'}")

    return all_results


if __name__ == "__main__":
    results = run_scaling_retraining(CONFIG)
