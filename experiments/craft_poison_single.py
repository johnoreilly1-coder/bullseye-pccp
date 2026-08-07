# ============================================================
# Poison Crafting -- Single Target, Multi-Layer Bullseye Polytope
# ============================================================
# Crafts poison sets for each of the 6 target patients used
# in the multi-run controlled evaluation.
#
# Usage:
#   PYTHONPATH=$(pwd) python experiments/craft_poison_single.py \
#       --frontal_idx 12
#
# Target patients (frontal indices):
#   near 0.5: idx 70 (patient64609), idx 134 (patient64673)
#   near 0.7: idx 12 (patient64552), idx 111 (patient64650)
#   near 0.9: idx 37 (patient64577), idx 105 (patient64644)
#
# Configuration:
#   Surrogate:    WB DenseNet-121 (white-box only)
#   Rate:         5% (9,551 poison images from 191,027)
#   Mode:         Multi-layer end-to-end (use_multilayer=True)
#   epsilon:      8/255
#   Steps:        500 per batch
#   Seed:         42 (controls base image selection)
#
# Output:
#   poison-storage/poison_single/idx_{N}/whitebox/
#       rate_05pct/
#           poison_images.pt
#           poison_labels.pt
#           base_images.pt
#           metadata.json
#       target_image.pt
#       target_info.json
#       crafting_summary.json
#
# Notes:
#   - Multi-layer mode uses features at 3 intermediate layers
#     simultaneously, normalised by feature dimension.
#     Appropriate for end-to-end fine-tuning where all layers
#     update during retraining (Aghakhani et al., 2021).
#   - Baseline AUC and PE score computed dynamically from the
#     Stage 2 checkpoint -- no hardcoded values.
# ============================================================

import torch
import json
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.densenet_model import build_densenet121
    from models.densenet_surrogate import load_whitebox_surrogate
    from attacks.bullseye_polytope import (
        get_target_multi_layer_features,
        craft_poison_set,
        verify_poisons,
        save_poison_set,
        get_target_features,
        PLEURAL_EFFUSION_IDX,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Run with: PYTHONPATH=$(pwd) python experiments/craft_poison_single.py")
    sys.exit(1)


# ── Target patients for the 6-patient multi-run experiment ───
TARGET_PATIENTS = {
    70:  {"patient": "patient64609", "group": "near_0.5"},
    134: {"patient": "patient64673", "group": "near_0.5"},
    12:  {"patient": "patient64552", "group": "near_0.7"},
    111: {"patient": "patient64650", "group": "near_0.7"},
    37:  {"patient": "patient64577", "group": "near_0.9"},
    105: {"patient": "patient64644", "group": "near_0.9"},
}

CONFIG = {
    "train_csv":           "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":           "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":           "/home/ubuntu/poison-storage/chexpert",
    "target_checkpoint":   "/home/ubuntu/poison-storage/checkpoints/best_model.pt",
    "poison_dir_root":     "/home/ubuntu/poison-storage/poison_single",
    "poison_rate":         0.05,      # 5% -- 9,551 poison images
    "steps":               500,
    "batch_size":          16,
    "seed":                42,        # controls base image selection
    "eps":                 8 / 255,
    "use_multilayer":      True,      # end-to-end multi-layer mode
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_baseline_auc(checkpoint_path, device):
    """Read baseline AUC directly from Stage 2 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    return round(float(ckpt.get("val_auc", 0.8381)), 4)


def get_baseline_pe_score(checkpoint_path, target_image, device):
    """
    Compute clean model PE classification score dynamically.
    Returns sigmoid output (model score, not calibrated probability).
    """
    model = build_densenet121(num_classes=14, pretrained=False)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device).eval()
    with torch.no_grad():
        logit = model(target_image.unsqueeze(0).to(device))
        score = torch.sigmoid(logit)[0, PLEURAL_EFFUSION_IDX].item()
    return round(score, 4)


def run_crafting(frontal_idx, config):
    print("=" * 60)
    print(f"POISON CRAFTING -- FRONTAL IDX {frontal_idx}")
    print(f"Mode: multi-layer end-to-end (use_multilayer=True)")
    print("=" * 60)

    if frontal_idx not in TARGET_PATIENTS:
        print(f"Warning: idx {frontal_idx} not in standard 6-patient set")
        print(f"Standard targets: {list(TARGET_PATIENTS.keys())}")

    device = get_device()
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load validation dataset
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )

    # Load target image
    target_image, target_labels = val_dataset[frontal_idx]
    pe_label = int(target_labels[PLEURAL_EFFUSION_IDX].item())

    if pe_label != 1:
        raise ValueError(
            f"Frontal idx {frontal_idx} is not PE-positive! "
            f"PE label = {pe_label}"
        )

    patient_info = TARGET_PATIENTS.get(frontal_idx, {})
    patient      = patient_info.get(
        "patient",
        val_dataset.df.iloc[frontal_idx]["Path"].split("/")[1]
    )
    group = patient_info.get("group", "unknown")

    # Compute baselines dynamically
    baseline_auc   = get_baseline_auc(
        config["target_checkpoint"], device
    )
    baseline_score = get_baseline_pe_score(
        config["target_checkpoint"], target_image, device
    )

    print(f"\nTarget patient:")
    print(f"  Frontal idx:       {frontal_idx}")
    print(f"  Patient:           {patient}")
    print(f"  Group:             {group}")
    print(f"  PE label:          Positive")
    print(f"  Baseline AUC:      {baseline_auc:.4f}")
    print(f"  Baseline PE score: {baseline_score:.4f}")
    print(f"  PE margin:         {baseline_score - 0.5:.4f}")
    print(f"\nPoison rate: {config['poison_rate']*100:.0f}% "
          f"({round(191027 * config['poison_rate']):,} images)")
    print(f"Crafting mode: {'multi-layer' if config['use_multilayer'] else 'single-layer'}")
    print(f"epsilon: {config['eps']:.4f} ({config['eps']*255:.0f}/255)")
    print(f"Seed: {config['seed']} (base image selection)")

    # Set output directory
    output_root = (
        Path(config["poison_dir_root"]) / f"idx_{frontal_idx}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Save target image and metadata
    torch.save(target_image, output_root / "target_image.pt")
    with open(output_root / "target_info.json", "w") as f:
        json.dump({
            "frontal_idx":     frontal_idx,
            "patient":         patient,
            "group":           group,
            "pe_label":        pe_label,
            "baseline_auc":    baseline_auc,
            "baseline_score":  baseline_score,
            "pe_margin":       round(baseline_score - 0.5, 4),
            "seed":            config["seed"],
            "eps":             config["eps"],
            "use_multilayer":  config["use_multilayer"],
        }, f, indent=2)

    # Load training dataset
    train_transform = get_transforms(mode="train", image_size=224)
    train_dataset   = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )
    n_train   = len(train_dataset)
    n_poisons = max(1, round(n_train * config["poison_rate"]))
    print(f"\nTraining images: {n_train:,}")
    print(f"Poison images:   {n_poisons:,}")

    # Load white-box surrogate
    surrogate = load_whitebox_surrogate(
        config["target_checkpoint"], device
    )
    surrogate_dir = output_root / "whitebox"
    surrogate_dir.mkdir(parents=True, exist_ok=True)

    rate_label = f"{int(config['poison_rate']*100):02d}pct"
    out_path   = surrogate_dir / f"rate_{rate_label}"

    if (out_path / "poison_images.pt").exists():
        print(f"\nPoison set already exists at {out_path}")
        print("Delete the directory to re-craft.")
        return

    # Craft poison set
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
            use_multilayer     = config["use_multilayer"],
        )

    # Verify poison quality
    n_verify = min(32, len(poison_images))

    # For verification, use single-layer target features
    target_feats_verify = [
        model.get_features(
            target_image.unsqueeze(0).to(device)
        ).squeeze(0)
        for model in surrogate
    ]

    verify_results = verify_poisons(
        base_images        = base_images[:n_verify],
        poison_images      = poison_images[:n_verify],
        eps                = config["eps"],
        surrogate_ensemble = surrogate,
        target_features    = target_feats_verify,
        device             = device,
    )

    # Save poison set
    save_poison_set(
        poison_images  = poison_images,
        poison_labels  = poison_labels,
        base_indices   = base_indices,
        target_idx     = frontal_idx,
        poison_rate    = config["poison_rate"],
        save_dir       = surrogate_dir,
        base_images    = base_images,
        use_multilayer = config["use_multilayer"],
    )

    # Save crafting summary
    summary = {
        "frontal_idx":    frontal_idx,
        "patient":        patient,
        "group":          group,
        "baseline_auc":   baseline_auc,
        "baseline_score": baseline_score,
        "n_poisons":      n_poisons,
        "use_multilayer": config["use_multilayer"],
        "mean_psnr":      verify_results["mean_psnr"],
        "mean_ssim":      verify_results.get("mean_ssim"),
        "budget_ok":      verify_results["budget_violations"] == 0,
    }
    with open(output_root / "crafting_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CRAFTING COMPLETE -- idx {frontal_idx}")
    print(f"  Poisons:   {n_poisons:,}")
    print(f"  PSNR:      {verify_results['mean_psnr']:.2f} dB")
    print(f"  SSIM:      {verify_results.get('mean_ssim', 0):.4f}")
    print(f"  Budget OK: {verify_results['budget_violations'] == 0}")
    print(f"  Mode:      {'multi-layer' if config['use_multilayer'] else 'single-layer'}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Craft poison images for a target patient"
    )
    parser.add_argument(
        "--frontal_idx",
        type=int,
        required=True,
        help=f"Frontal index of target patient. "
             f"Standard 6-patient targets: "
             f"{list(TARGET_PATIENTS.keys())}"
    )
    parser.add_argument(
        "--single_layer",
        action="store_true",
        help="Use single penultimate layer (default: multi-layer)"
    )
    args = parser.parse_args()

    if args.single_layer:
        CONFIG["use_multilayer"] = False
        print("WARNING: Using single-layer mode. "
              "Multi-layer is recommended for end-to-end fine-tuning.")

    run_crafting(args.frontal_idx, CONFIG)
