# ============================================================
# Stage 5 / 5b — Poisoned Model Retraining (Single Target)
# ============================================================
# Retrains DenseNet-121 on poison sets crafted by
# craft_poison_single.py for a specified target patient.
#
# Usage:
#   PYTHONPATH=$(pwd) python experiments/retrain_poisoned_single.py [frontal_idx]
#
# Default target: frontal idx 37 (patient64577, PE prob 0.9147)
#
# Configuration:
#   Surrogates:   WB DenseNet-121 and BB ResNet-50
#   Rates:        1%, 2%, 5%
#   Condition:    Realistic (aug) only
#   Epochs:       20
#
# Baseline AUC and PE probability computed dynamically
# from the Stage 2 checkpoint — no hardcoded values.
# ============================================================

import torch
import json
import sys
import time
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
        train_epoch,
        validate_epoch,
        measure_attack_success,
    )
except ImportError:
    pass


CONFIG = {
    "train_csv":          "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":          "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":          "/home/ubuntu/poison-storage/chexpert",
    "target_checkpoint":  "/home/ubuntu/poison-storage/checkpoints/best_model.pt",
    "poison_dir_root":    "/home/ubuntu/poison-storage/poison_single",
    "checkpoint_dir_root":"/home/ubuntu/poison-storage/checkpoints_single",
    "results_dir_root":   "/home/ubuntu/poison-storage/results_single",
    "target_frontal_idx": 37,   # overridden by command line arg
    "poison_rates":       [0.01, 0.02, 0.05],
    #"poison_rates":       [0.02],
    "batch_size":         32,
    "num_workers":        4,
    "epochs":             20,
    "lr":                 1e-4,
    "weight_decay":       1e-5,
}

SURROGATE_TYPES = ["whitebox", "blackbox"]
#SURROGATE_TYPES = ["whitebox"]


def get_baseline_auc(checkpoint_path, device):
    """Read baseline AUC directly from Stage 2 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    return round(float(ckpt.get("val_auc", 0.8381)), 4)


def get_baseline_pe_prob(checkpoint_path, target_image, device):
    """Compute clean model PE probability dynamically."""
    model = build_densenet121(num_classes=14, pretrained=False)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device).eval()
    with torch.no_grad():
        logit = model(target_image.unsqueeze(0).to(device))
        prob  = torch.sigmoid(logit)[0, PLEURAL_EFFUSION_IDX].item()
    return round(prob, 4)


def retrain_one(target_image, target_idx, poison_dir,
                checkpoint_dir, config, device):
    """Retrain DenseNet-121 on one poison set."""

    val_transform   = get_transforms(mode="val",   image_size=224)
    train_transform = get_transforms(mode="train", image_size=224)

    # Load poison set
    poison_path = Path(poison_dir)
    poison_images = torch.load(
        poison_path / "poison_images.pt", map_location="cpu"
    )
    poison_labels = torch.load(
        poison_path / "poison_labels.pt", map_location="cpu"
    )
    with open(poison_path / "metadata.json") as f:
        meta = json.load(f)
    base_indices = meta["base_indices"]

    # Build datasets
    full_train = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )
    train_dataset = PoisonedCheXpertDataset(
        clean_dataset   = full_train,
        poison_images   = poison_images,
        poison_labels   = poison_labels,
        base_indices    = base_indices,
        augment_poisons = True,
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

    model     = build_densenet121(
        num_classes=14, pretrained=True
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )
    scaler   = GradScaler("cuda")
    best_auc = 0.0

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epoch_log = []  
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

        is_best = val_auc > best_auc

        if is_best:
            best_auc = val_auc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_auc":          val_auc,
                "attack":           attack,
            }, ckpt_dir / "best_model.pt")

        epoch_log.append({
            "epoch":          epoch,
            "val_auc":        round(val_auc, 4),
            "pe_probability": attack["pe_probability"],
            "attack_success": attack["attack_success"],
            "is_best":        is_best,
        })

    best_ckpt = torch.load(
        ckpt_dir / "best_model.pt", map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    final = measure_attack_success(model, target_image, device)

    return {
        "best_epoch":     int(best_ckpt["epoch"]),
        "best_mean_auc":  round(best_auc, 4),
        "pe_probability": final["pe_probability"],
        "attack_success": final["attack_success"],
        "epoch_log":      epoch_log,
    }


def run_retraining(config):
    device = get_device()

    # Load target image
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )
    target_idx      = config["target_frontal_idx"]
    target_image, _ = val_dataset[target_idx]
    patient = val_dataset.df.iloc[target_idx]["Path"].split("/")[1]

    # Compute baselines dynamically
    baseline_auc = get_baseline_auc(
        config["target_checkpoint"], device
    )
    baseline_pe = get_baseline_pe_prob(
        config["target_checkpoint"], target_image, device
    )

    print("=" * 65)
    print(f"STAGE 5/5b — RETRAINING (FRONTAL IDX {target_idx})")
    print("=" * 65)
    print(f"Target:           Frontal idx {target_idx} ({patient})")
    print(f"Baseline AUC:     {baseline_auc:.4f} (from Stage 2 checkpoint)")
    print(f"Baseline PE prob: {baseline_pe:.4f} (dynamically computed)")
    print(f"PE margin:        {baseline_pe-0.5:.4f}")
    print(f"Poison rates:     {[f'{r*100:.0f}%' for r in config['poison_rates']]}")
    print(f"Condition:        Realistic (aug) only")
    print(f"Epochs:           {config['epochs']}")

    results_dir = (
        Path(config["results_dir_root"]) / f"idx_{target_idx}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for surrogate_type in SURROGATE_TYPES:
        for rate in config["poison_rates"]:
            rate_label = f"{int(rate*100):02d}pct"

            print(f"\n{'='*65}")
            print(f"{surrogate_type.upper()} — {rate*100:.0f}% poisoning")
            print(f"{'='*65}")

            poison_dir = (
                Path(config["poison_dir_root"])
                / f"idx_{target_idx}"
                / surrogate_type
                / f"rate_{rate_label}"
            )
            checkpoint_dir = (
                Path(config["checkpoint_dir_root"])
                / f"idx_{target_idx}"
                / surrogate_type
                / f"rate_{rate_label}"
            )

            t0 = time.time()
            result = retrain_one(
                target_image, target_idx,
                poison_dir, checkpoint_dir,
                config, device
            )
            elapsed = round((time.time() - t0) / 60, 1)

            result.update({
                "surrogate":      surrogate_type,
                "poison_rate":    rate,
                "baseline_auc":   baseline_auc,
                "baseline_pe":    baseline_pe,
                "auc_delta":      round(
                    result["best_mean_auc"] - baseline_auc, 4
                ),
                "pe_drop":        round(
                    baseline_pe - result["pe_probability"], 4
                ),
                "time_mins":      elapsed,
            })
            all_results.append(result)

            print(f"\n  Result: PE {result['pe_probability']:.4f}  "
                  f"{'✓ SUCCESS' if result['attack_success'] else '✗ NO'}  "
                  f"AUC {result['best_mean_auc']:.4f}  "
                  f"ΔAUC {result['auc_delta']:+.4f}  "
                  f"PE drop {result['pe_drop']:+.4f}  "
                  f"{elapsed}min")

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"COMPLETE — FRONTAL IDX {target_idx} ({patient})")
    print(f"{'='*65}")
    print(f"{'Run':<22}{'AUC':>8}{'ΔAUC':>8}"
          f"{'PE prob':>10}{'PE drop':>10}{'Attack':>8}")
    print("-" * 65)
    print(f"{'Clean baseline':<22}{baseline_auc:>8.4f}{'—':>8}"
          f"{baseline_pe:>10.4f}{'—':>10}{'—':>8}")
    print("-" * 65)
    for r in all_results:
        label = f"{r['surrogate']} {r['poison_rate']*100:.0f}%"
        print(f"{label:<22}{r['best_mean_auc']:>8.4f}"
              f"{r['auc_delta']:>+8.4f}"
              f"{r['pe_probability']:>10.4f}"
              f"{r['pe_drop']:>+10.4f}"
              f"{'✓' if r['attack_success'] else '✗':>8}")

    # Save results
    results_path = results_dir / "summary.json"
    with open(results_path, "w") as f:
        json.dump({
            "target_frontal_idx": target_idx,
            "patient":            patient,
            "baseline_auc":       baseline_auc,
            "baseline_pe":        baseline_pe,
            "results":            all_results,
        }, f, indent=2)
    print(f"\nSaved: {results_path}")

    return all_results


if __name__ == "__main__":
    target_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 37
    CONFIG["target_frontal_idx"] = target_idx
    run_retraining(CONFIG)
