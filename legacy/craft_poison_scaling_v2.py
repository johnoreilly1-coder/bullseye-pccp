# ============================================================
# Scaling Experiment Run 2 — Poison Crafting
# ============================================================
# Crafts poison sets for two target patients across five
# training dataset sizes at a fixed 5% poisoning rate.
#
# Configuration:
#   Surrogate:    Black-box ResNet-50 ensemble (seeds 42/123/456)
#   Target:       Single image (no EOT)
#   Patients:     Val index 4 (clean PE 0.93) and
#                 Val index 15 (clean PE 0.736)
#   Sizes:        10k, 25k, 50k, 100k, 191k
#   Rate:         5% fixed
#   Seed:         123 (different from Stage 5e seed=42)
#   ε:            8/255
#
# Saves each poison set to a patient-specific and size-specific
# subdirectory for clean organisation and safe resumption.
# ============================================================

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.resnet_surrogate import load_surrogate_ensemble
    from attacks.bullseye_polytope import (
        get_target_features,
        craft_poison_set,
        verify_poisons,
        save_poison_set,
        PLEURAL_EFFUSION_IDX,
        DEFAULT_EPS,
    )
except ImportError:
    pass


TRAINING_SIZES = [10_000, 25_000, 50_000, 100_000, 191_027]
POISON_RATE    = 0.05
TARGET_PATIENTS = [4, 12]   # val index 4 and val index 15


CONFIG = {
    "train_csv":   "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":   "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":   "/home/ubuntu/poison-storage/chexpert",
    "surrogate_checkpoint_dir": "/home/ubuntu/poison-storage/surrogate_checkpoints",
    "surrogate_seeds":          [42, 123, 456],
    "poison_dir_root": "/home/ubuntu/poison-storage/poison_scaling2",
    "steps":       500,
    "batch_size":  16,
    "seed":        123,
    "eps":         8/255,
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_scaling_crafting(config):
    print("=" * 60)
    print("SCALING EXPERIMENT RUN 2 — POISON CRAFTING")
    print("=" * 60)
    print(f"Target patients:  {TARGET_PATIENTS}")
    print(f"Training sizes:   {[f'{n//1000}k' for n in TRAINING_SIZES]}")
    print(f"Poisoning rate:   {POISON_RATE*100:.0f}% (fixed)")
    print(f"Surrogate:        BB ResNet-50 ensemble (no EOT)")
    print(f"Seed:             {config['seed']}")
    print(f"ε:                {config['eps']:.4f} ({config['eps']*255:.0f}/255)")

    device = get_device()

    # ── Load surrogate ensemble ───────────────────────────────
    print("\nLoading BB surrogate ensemble...")
    surrogate_ensemble = load_surrogate_ensemble(
        checkpoint_dir = config["surrogate_checkpoint_dir"],
        seeds          = config["surrogate_seeds"],
        device         = device,
    )

    # ── Load validation dataset ───────────────────────────────
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )

    # ── Process each target patient ───────────────────────────
    all_summary = {}

    for target_idx in TARGET_PATIENTS:
        print(f"\n{'='*60}")
        print(f"TARGET PATIENT: val index {target_idx}")
        print(f"{'='*60}")

        # Load target image
        target_image, target_labels = val_dataset[target_idx]
        pe_label = int(target_labels[PLEURAL_EFFUSION_IDX].item())
        assert pe_label == 1, \
            f"Val index {target_idx} is not PE-positive!"

        # Get target features (single image, no EOT)
        target_features = get_target_features(
            target_image, surrogate_ensemble, device
        )

        patient_dir = Path(config["poison_dir_root"]) / f"patient_{target_idx}"
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Save target image
        torch.save(target_image, patient_dir / "target_image.pt")
        with open(patient_dir / "target_info.json", "w") as f:
            json.dump({
                "val_index":   target_idx,
                "pe_label":    pe_label,
                "surrogate":   "BB ResNet-50 ensemble",
                "eot":         "none",
                "seed":        config["seed"],
            }, f, indent=2)

        patient_summary = {}

        # ── Process each training size ─────────────────────────
        for n_total in TRAINING_SIZES:
            n_poisons  = round(n_total * POISON_RATE)
            size_label = f"{n_total // 1000}k"
            save_dir   = patient_dir / f"size_{size_label}"

            # Skip if already done
            rate_dir = save_dir / "rate_05pct"
            if (rate_dir / "poison_images.pt").exists():
                print(f"\n  Skipping {size_label} "
                      f"(already exists)")
                continue

            print(f"\n  Size {size_label}: crafting "
                  f"{n_poisons:,} poisons...")

            # Load training subset for this size
            train_transform = get_transforms(
                mode="train", image_size=224
            )
            train_dataset = CheXpertDataset(
                csv_path         = config["train_csv"],
                image_dir        = config["image_dir"],
                transform        = train_transform,
                uncertain_policy = "zeros",
                frontal_only     = True,
                subset           = n_total if n_total < 191_027 else None,
            )

            # Craft poisons
            poison_images, poison_labels, base_indices, base_images = \
                craft_poison_set(
                    train_dataset      = train_dataset,
                    target_image       = target_image,
                    surrogate_ensemble = surrogate_ensemble,
                    n_poisons          = n_poisons,
                    device             = device,
                    eps                = config["eps"],
                    steps              = config["steps"],
                    batch_size         = config["batch_size"],
                    seed               = config["seed"],
                    target_features    = None,
                    n_eot              = 1,
                )

            # Verify
            n_verify = min(16, len(poison_images))
            verify_results = verify_poisons(
                base_images        = base_images[:n_verify],
                poison_images      = poison_images[:n_verify],
                eps                = config["eps"],
                surrogate_ensemble = surrogate_ensemble,
                target_features    = target_features,
                device             = device,
            )

            # Save
            save_poison_set(
                poison_images = poison_images,
                poison_labels = poison_labels,
                base_indices  = base_indices,
                target_idx    = target_idx,
                poison_rate   = POISON_RATE,
                save_dir      = save_dir,
                base_images   = base_images,
            )

            patient_summary[size_label] = {
                "n_total":          n_total,
                "n_poisons":        n_poisons,
                "mean_psnr":        verify_results["mean_psnr"],
                "mean_ssim":        verify_results.get("mean_ssim"),
                "budget_ok":        verify_results["budget_violations"] == 0,
            }

            print(f"  {size_label}: {n_poisons:,} poisons  "
                  f"PSNR {verify_results['mean_psnr']:.2f}dB  "
                  f"{'✓' if verify_results['budget_violations']==0 else '✗'}")

        all_summary[f"patient_{target_idx}"] = patient_summary

    # Save summary
    with open(Path(config["poison_dir_root"]) / "crafting_summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SCALING CRAFTING COMPLETE")
    print(f"{'='*60}")
    for patient_key, sizes in all_summary.items():
        print(f"\n{patient_key}:")
        for size, s in sizes.items():
            print(f"  {size}: {s['n_poisons']:,} poisons  "
                  f"PSNR {s['mean_psnr']:.2f}dB  "
                  f"{'✓' if s['budget_ok'] else '✗'}")

    return all_summary


if __name__ == "__main__":
    run_scaling_crafting(CONFIG)
