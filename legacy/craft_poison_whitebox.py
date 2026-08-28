# ============================================================
# Craft White-Box Poison Samples — Stage 4b
# ============================================================
# Uses the trained DenseNet-121 target model directly as the
# surrogate for Bullseye Polytope poison crafting.
#
# This is the white-box upper bound experiment. By using the
# actual target model rather than ResNet-50 surrogates, the
# cross-architecture transfer gap is eliminated entirely.
# If the attack succeeds here but not in the black-box setting
# (Stages 3-5), this confirms that the surrogate transfer gap
# is the limiting factor — not the attack algorithm itself.
#
# Relationship to Stage 4 (black-box):
#   Stage 4:  3 × ResNet-50 surrogates (black-box)
#   Stage 4b: 1 × DenseNet-121 target  (white-box)
#   Same: Bullseye Polytope algorithm, ε=8/255, 500 steps
#   Same: target image (val index 4), poisoning rates 1/2/5%
#
# Outputs saved to poison_dir_whitebox (separate from Stage 4
# black-box poisons to allow direct comparison).
# ============================================================

import torch
import json
import numpy as np
from pathlib import Path

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.densenet_surrogate import (
        load_whitebox_surrogate, verify_whitebox_surrogate
    )
    from attacks.bullseye_polytope import (
        select_target_image,
        get_target_features,
        craft_poison_set,
        verify_poisons,
        save_poison_set,
        PLEURAL_EFFUSION_IDX,
        DEFAULT_EPS,
    )
except ImportError:
    pass


POISON_RATES = [0.01, 0.02, 0.05]


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def run_stage4b(config):
    """
    White-box poison crafting pipeline.

    Identical to run_stage4() in craft_poison.py except the
    surrogate ensemble is replaced by the trained DenseNet-121
    target model loaded from checkpoint.
    """
    print("=" * 55)
    print("STAGE 4b — WHITE-BOX POISON CRAFTING")
    print("=" * 55)
    print(f"Surrogate:       DenseNet-121 (white-box target model)")
    print(f"Checkpoint:      {config['target_checkpoint']}")
    print(f"Attack target:   Pleural Effusion")
    print(f"Poisoning rates: "
          f"{[f'{r*100:.0f}%' for r in config['poison_rates']]}")
    print(f"Steps per batch: {config['steps']}")
    print(f"ε (pixel space): {DEFAULT_EPS:.4f} "
          f"({DEFAULT_EPS*255:.1f}/255)")

    device = get_device()

    # ── Load validation dataset ───────────────────────────────
    print("\n── Loading validation dataset ───────────────────")
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
        subset           = None,
    )

    # ── Load training dataset ─────────────────────────────────
    print("\n── Loading training dataset ─────────────────────")
    train_transform = get_transforms(mode="train", image_size=224)
    train_dataset   = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
        subset           = config["train_subset"],
    )
    n_train = len(train_dataset)
    print(f"Training images: {n_train:,}")

    # ── Select target image ───────────────────────────────────
    print("\n── Selecting target image ───────────────────────")
    target_image, target_labels, target_idx = select_target_image(
        val_dataset,
        target_label_idx = PLEURAL_EFFUSION_IDX,
        sample_idx       = config["target_sample_idx"],
    )

    print("\n" + "=" * 50)
    print("TARGET IMAGE")
    print("=" * 50)
    print(f"Dataset index:  {target_idx}")
    print(f"Image shape:    {list(target_image.shape)}")
    print(f"\nLabels (1=positive, 0=negative):")
    for i, label in enumerate(LABEL_COLS):
        val = int(target_labels[i].item())
        marker = " ← ATTACK TARGET" if label == "Pleural Effusion" \
                 else ""
        if val == 1:
            print(f"  {label:<35} {val}{marker}")
    print("=" * 50)

    # Confirm same target as Stage 4 black-box
    assert target_idx == config["expected_target_idx"], (
        f"Target index mismatch: expected "
        f"{config['expected_target_idx']}, got {target_idx}. "
        f"Adjust target_sample_idx or expected_target_idx."
    )
    print(f"\n✓ Target confirmed: val index {target_idx} "
          f"(same as Stage 4 black-box)")

    # Save target image
    poison_dir = Path(config["poison_dir_whitebox"])
    poison_dir.mkdir(parents=True, exist_ok=True)
    torch.save(target_image, poison_dir / "target_image.pt")
    with open(poison_dir / "target_info.json", "w") as f:
        json.dump({
            "dataset_idx":  target_idx,
            "target_label": "Pleural Effusion",
            "label_idx":    PLEURAL_EFFUSION_IDX,
            "surrogate":    "DenseNet-121 white-box",
            "all_labels":   {
                LABEL_COLS[i]: int(target_labels[i].item())
                for i in range(len(LABEL_COLS))
            }
        }, f, indent=2)

    # ── Load white-box surrogate ──────────────────────────────
    print("\n── Loading white-box surrogate ──────────────────")
    surrogate_ensemble = load_whitebox_surrogate(
        checkpoint_path = config["target_checkpoint"],
        device          = device,
    )
    verify_whitebox_surrogate(surrogate_ensemble[0], device)

    # ── Craft poisons at each rate ────────────────────────────
    summary = {}

    for rate in config["poison_rates"]:
        n_poisons = max(1, round(n_train * rate))
        pct       = f"{rate*100:.0f}%"

        print(f"\n{'='*55}")
        print(f"CRAFTING POISONS — rate {pct} ({n_poisons:,} images)")
        print(f"{'='*55}")

        poison_images, poison_labels, base_indices, base_images = \
            craft_poison_set(
                train_dataset      = train_dataset,
                target_image       = target_image,
                surrogate_ensemble = surrogate_ensemble,
                n_poisons          = n_poisons,
                device             = device,
                eps                = DEFAULT_EPS,
                steps              = config["steps"],
                batch_size         = config["batch_size"],
                seed               = config["seed"],
            )

        # ── Verify ────────────────────────────────────────────
        print(f"\nVerifying poison quality ({pct})...")
        target_features = get_target_features(
            target_image, surrogate_ensemble, device
        )
        n_verify = min(32, len(poison_images))
        verify_results = verify_poisons(
            base_images        = base_images[:n_verify],
            poison_images      = poison_images[:n_verify],
            eps                = DEFAULT_EPS,
            surrogate_ensemble = surrogate_ensemble,
            target_features    = target_features,
            device             = device,
        )

        # ── Save ──────────────────────────────────────────────
        save_dir = save_poison_set(
            poison_images = poison_images,
            poison_labels = poison_labels,
            base_indices  = base_indices,
            target_idx    = target_idx,
            poison_rate   = rate,
            save_dir      = config["poison_dir_whitebox"],
            base_images   = base_images,        # 
        )

        summary[pct] = {
            "n_poisons":        n_poisons,
            "max_perturbation": verify_results["max_perturbation"],
            "mean_psnr":        verify_results["mean_psnr"],
            "mean_ssim":        verify_results.get("mean_ssim"),
            "budget_ok":        verify_results["budget_violations"] == 0,
            "feature_similarity": verify_results.get(
                "feature_similarity"
            ),
            "save_dir":         str(save_dir),
        }

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STAGE 4b COMPLETE — WHITE-BOX POISON CRAFTING")
    print("=" * 55)
    print(f"Surrogate:     DenseNet-121 (white-box)")
    print(f"Target image:  val index {target_idx} "
          f"(Pleural Effusion positive)")
    print(f"\n{'Rate':<8} {'Poisons':>10} {'Max pert':>12} "
          f"{'PSNR':>10} {'Budget':>10}")
    print("-" * 55)
    for pct, s in summary.items():
        print(f"{pct:<8} {s['n_poisons']:>10,} "
              f"{s['max_perturbation']:>12.4f} "
              f"{s['mean_psnr']:>9.2f}dB "
              f"{'✓' if s['budget_ok'] else '✗':>10}")
    print("=" * 55)

    with open(poison_dir / "stage4b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {poison_dir / 'stage4b_summary.json'}")

    return summary


# ── Configs ───────────────────────────────────────────────────

KAGGLE_CONFIG = {
    "train_csv":   "/kaggle/input/datasets/ashery/chexpert/train.csv",
    "valid_csv":   "/kaggle/input/datasets/ashery/chexpert/valid.csv",
    "image_dir":   "/kaggle/input/datasets/ashery/chexpert",
    "poison_dir_whitebox": "/kaggle/working/poison_samples_whitebox",
    "target_checkpoint":   None,   # not available on Kaggle
    "poison_rates":        [0.01],
    "target_sample_idx":   0,
    "expected_target_idx": 4,      # val index 4 confirmed in Stage 4
    "steps":               50,
    "batch_size":          8,
    "seed":                42,
    "train_subset":        500,
}

LAMBDA_CONFIG = {
    "train_csv":   "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":   "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":   "/home/ubuntu/poison-storage/chexpert",
    "poison_dir_whitebox": "/home/ubuntu/poison-storage/poison_samples_whitebox",
    "target_checkpoint":   "/home/ubuntu/poison-storage/checkpoints/best_model.pt",
    "poison_rates":        [0.01, 0.02, 0.05],
    "target_sample_idx":   0,
    "expected_target_idx": 4,      # val index 4 confirmed in Stage 4
    "steps":               500,
    "batch_size":          16,
    "seed":                42,
    "train_subset":        None,
}


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_stage4b(LAMBDA_CONFIG)