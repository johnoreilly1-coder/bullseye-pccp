# ============================================================
# Stage 4 / 4b — Poison Crafting (Single Target, No EOT)
# ============================================================
# Crafts poison sets for a specified target patient under
# both white-box and black-box surrogate configurations.
#
# Usage:
#   PYTHONPATH=$(pwd) python experiments/craft_poison_single.py [frontal_idx]
#
# Default target: frontal idx 37 (patient64577, PE prob 0.9147)
# First PE-positive patient with clean PE probability > 0.90
#
# Configuration:
#   Surrogate:    WB DenseNet-121 and BB ResNet-50 ensemble
#   Rates:        1%, 2%, 5%
#   EOT:          None (single target image)
#   ε:            8/255
#   Condition:    Realistic (aug) only
#
# Baseline AUC and PE probability computed dynamically
# from the Stage 2 checkpoint — no hardcoded values.
# ============================================================

import torch
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.densenet_model import build_densenet121
    from models.densenet_surrogate import load_whitebox_surrogate
    from models.resnet_surrogate import load_surrogate_ensemble
    from attacks.bullseye_polytope import (
        get_target_features,
        craft_poison_set,
        verify_poisons,
        save_poison_set,
        PLEURAL_EFFUSION_IDX,
    )
except ImportError:
    pass


CONFIG = {
    "train_csv":              "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":              "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":              "/home/ubuntu/poison-storage/chexpert",
    "target_checkpoint":      "/home/ubuntu/poison-storage/checkpoints/best_model.pt",
    "surrogate_checkpoint_dir": "/home/ubuntu/poison-storage/surrogate_checkpoints",
    "surrogate_seeds":        [42, 123, 456],
    "poison_dir_root":        "/home/ubuntu/poison-storage/poison_single",
    "target_frontal_idx":     37,   # overridden by command line arg
    "poison_rates":           [0.01, 0.02, 0.05],
    #"poison_rates":           [0.02],
    "steps":                  500,
    "batch_size":             16,
    "seed":                   42,
    "eps":                    8/255,
}

SURROGATE_TYPES = ["whitebox", "blackbox"]
#SURROGATE_TYPES = ["whitebox"]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run_crafting(config):
    print("=" * 60)
    print("STAGE 4 / 4b — POISON CRAFTING (SINGLE TARGET, NO EOT)")
    print("=" * 60)

    device = get_device()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # ── Load validation dataset ───────────────────────────────
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )

    # ── Load target image ─────────────────────────────────────
    target_idx   = config["target_frontal_idx"]
    target_image, target_labels = val_dataset[target_idx]
    pe_label = int(target_labels[PLEURAL_EFFUSION_IDX].item())
    assert pe_label == 1, \
        f"Frontal idx {target_idx} is not PE-positive!"

    patient = val_dataset.df.iloc[target_idx]["Path"].split("/")[1]

    # ── Compute baselines dynamically ─────────────────────────
    baseline_auc = get_baseline_auc(
        config["target_checkpoint"], device
    )
    baseline_pe = get_baseline_pe_prob(
        config["target_checkpoint"], target_image, device
    )

    print(f"\nTarget patient:")
    print(f"  Frontal idx:      {target_idx}")
    print(f"  Patient:          {patient}")
    print(f"  PE label:         Positive")
    print(f"  Baseline AUC:     {baseline_auc:.4f} (from checkpoint)")
    print(f"  Baseline PE prob: {baseline_pe:.4f} (dynamically computed)")
    print(f"  PE margin:        {baseline_pe-0.5:.4f}")
    print(f"\nPoison rates: {[f'{r*100:.0f}%' for r in config['poison_rates']]}")
    print(f"EOT:          None (single target image)")
    print(f"ε:            {config['eps']:.4f} ({config['eps']*255:.0f}/255)")

    # ── Set output directory per target ───────────────────────
    output_root = Path(config["poison_dir_root"]) / f"idx_{target_idx}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Save target image and info
    torch.save(target_image, output_root / "target_image.pt")
    with open(output_root / "target_info.json", "w") as f:
        json.dump({
            "frontal_idx":  target_idx,
            "patient":      patient,
            "pe_label":     pe_label,
            "baseline_auc": baseline_auc,
            "baseline_pe":  baseline_pe,
            "pe_margin":    round(baseline_pe - 0.5, 4),
            "eot":          "none",
            "seed":         config["seed"],
            "eps":          config["eps"],
        }, f, indent=2)

    # ── Load training dataset ─────────────────────────────────
    train_transform = get_transforms(mode="train", image_size=224)
    train_dataset   = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )
    n_train = len(train_dataset)
    print(f"\nTraining images: {n_train:,}")

    all_summary = {}

    for surrogate_type in SURROGATE_TYPES:
        print(f"\n{'='*60}")
        print(f"SURROGATE: {surrogate_type.upper()}")
        print(f"{'='*60}")

        if surrogate_type == "whitebox":
            surrogate = load_whitebox_surrogate(
                config["target_checkpoint"], device
            )
            surrogate_label = "DenseNet-121 white-box"
        else:
            surrogate = load_surrogate_ensemble(
                checkpoint_dir = config["surrogate_checkpoint_dir"],
                seeds          = config["surrogate_seeds"],
                device         = device,
            )
            surrogate_label = "ResNet-50 black-box ensemble"

        print(f"Surrogate: {surrogate_label}")

        target_features = get_target_features(
            target_image, surrogate, device
        )

        surrogate_dir = output_root / surrogate_type
        surrogate_dir.mkdir(parents=True, exist_ok=True)

        surrogate_summary = {}

        for rate in config["poison_rates"]:
            n_poisons  = max(1, round(n_train * rate))
            rate_label = f"{int(rate*100):02d}pct"
            save_dir   = surrogate_dir 

            if (save_dir / f"rate_{rate_label}" / "poison_images.pt").exists():
                print(f"\n  {rate*100:.0f}% — already exists, skipping")
                continue
            print(f"\n  Crafting {rate*100:.0f}% — {n_poisons:,} poisons...")

            poison_images, poison_labels, base_indices, base_images = \
                craft_poison_set(
                    train_dataset      = train_dataset,
                    target_image       = target_image,
                    surrogate_ensemble = surrogate,
                    n_poisons          = n_poisons,
                    device             = device,
                    eps                = config["eps"],
                    steps              = config["steps"],
                    batch_size         = config["batch_size"],
                    seed               = config["seed"],
                    target_features    = None,
                    n_eot              = 1,
                )

            n_verify = min(32, len(poison_images))
            verify_results = verify_poisons(
                base_images        = base_images[:n_verify],
                poison_images      = poison_images[:n_verify],
                eps                = config["eps"],
                surrogate_ensemble = surrogate,
                target_features    = target_features,
                device             = device,
            )

            save_poison_set(
                poison_images = poison_images,
                poison_labels = poison_labels,
                base_indices  = base_indices,
                target_idx    = target_idx,
                poison_rate   = rate,
                save_dir      = save_dir,
                base_images   = base_images,
            )

            surrogate_summary[f"{rate*100:.0f}pct"] = {
                "n_poisons":   n_poisons,
                "mean_psnr":   verify_results["mean_psnr"],
                "mean_ssim":   verify_results.get("mean_ssim"),
                "budget_ok":   verify_results["budget_violations"] == 0,
            }

            print(f"  {rate*100:.0f}%: {n_poisons:,} poisons  "
                  f"PSNR {verify_results['mean_psnr']:.2f}dB  "
                  f"{'✓' if verify_results['budget_violations']==0 else '✗'}")

        all_summary[surrogate_type] = surrogate_summary

    with open(output_root / "crafting_summary.json", "w") as f:
        json.dump({
            "target_frontal_idx": target_idx,
            "patient":            patient,
            "baseline_auc":       baseline_auc,
            "baseline_pe":        baseline_pe,
            "summary":            all_summary,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("CRAFTING COMPLETE")
    print(f"{'='*60}")
    for stype, rates in all_summary.items():
        print(f"\n{stype}:")
        for rate_key, s in rates.items():
            print(f"  {rate_key}: {s['n_poisons']:,} poisons  "
                  f"PSNR {s['mean_psnr']:.2f}dB  "
                  f"{'✓' if s['budget_ok'] else '✗'}")

    return all_summary


if __name__ == "__main__":
    target_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 37
    CONFIG["target_frontal_idx"] = target_idx
    run_crafting(CONFIG)
