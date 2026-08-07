# ============================================================
# Control Retraining -- Matched Seed Design
# ============================================================
# Retrains the DenseNet-121 target model on CLEAN data
# (no poison images) across 5 independent runs using the
# same seeds as the poisoned retraining (retrain_multirun.py).
#
# Matched seed design:
#   Each control run uses the same seed as its corresponding
#   poisoned run. This means the two runs share identical:
#     - Weight initialisation (classifier layer)
#     - Data ordering (batch shuffle sequence)
#     - Augmentation sequence (random flips and crops)
#   The only difference between a matched pair is the presence
#   or absence of poison images in the training set.
#   This eliminates seed-to-seed variation as a confounding
#   factor and enables paired statistical analysis.
#
# Usage:
#   PYTHONPATH=$(pwd) python experiments/retrain_control.py
#
# Output:
#   results/control/
#       run_{seed}/
#           scores.json    -- PE scores for all 202 patients
#                             + mean AUC for this run
#       summary.json       -- all 5 runs aggregated, including
#                             PE ground truth labels and
#                             threshold-straddling pre-analysis
#
# Matched seeds (same as retrain_multirun.py):
#   [100, 200, 300, 400, 500]
#
# Notes:
#   - PE scores recorded for ALL 202 frontal validation patients
#     after each run, not just the 6 target patients.
#     This supports the threshold-straddling analysis across
#     the full validation set.
#   - Best-AUC checkpoint selected per run for score recording.
#   - Same validation set used for checkpoint selection and
#     evaluation -- acknowledged as a limitation (selection bias).
#   - PE ground truth labels recorded in summary.json to support
#     threshold_straddling.py without reloading the dataset.
# ============================================================

import torch
import torch.nn as nn
import json
import sys
import numpy as np
import random
from pathlib import Path
from torch.utils.data import DataLoader

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.densenet_model import build_densenet121
    from attacks.bullseye_polytope import PLEURAL_EFFUSION_IDX
except ImportError as e:
    print(f"Import error: {e}")
    print("Run with: PYTHONPATH=$(pwd) python experiments/retrain_control.py")
    sys.exit(1)

from sklearn.metrics import roc_auc_score


# Matched seeds -- identical to retrain_multirun.py
MATCHED_SEEDS = [100, 200, 300, 400, 500]

CONFIG = {
    "train_csv":         "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":         "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":         "/home/ubuntu/poison-storage/chexpert",
    "target_checkpoint": "/home/ubuntu/poison-storage/checkpoints/best_model.pt",
    "results_dir":       "/home/ubuntu/poison-storage/results/control",
    "epochs":            10,
    "lr":                1e-4,
    "weight_decay":      1e-5,
    "batch_size":        32,
    "num_workers":       4,
    "seeds":             MATCHED_SEEDS,
}


def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    Must be called before model construction and DataLoader
    creation to fully control weight initialisation, data
    ordering and augmentation sequence.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_all_patients(model, val_loader, device):
    """
    Evaluate model on the full validation set.

    Returns PE classification score for every patient and mean
    AUC across all 14 labels. Records scores for all 202
    patients to support threshold-straddling analysis.

    Parameters
    ----------
    model      : nn.Module  (eval mode)
    val_loader : DataLoader  (shuffle=False, fixed order)
    device     : torch.device

    Returns
    -------
    mean_auc   : float
    all_scores : list of float  PE score for each patient [N=202]
    all_preds  : np.ndarray     [N, 14] all label predictions
    all_labels : np.ndarray     [N, 14] ground truth
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds  = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)  # [N, 14]
    all_labels = np.concatenate(all_labels, axis=0)  # [N, 14]

    # Mean AUC across 14 labels
    aucs = []
    for i in range(14):
        if len(np.unique(all_labels[:, i])) > 1:
            aucs.append(
                roc_auc_score(all_labels[:, i], all_preds[:, i])
            )
    mean_auc = float(np.mean(aucs)) if aucs else 0.0

    # PE score for every patient
    all_scores = [
        round(float(all_preds[i, PLEURAL_EFFUSION_IDX]), 4)
        for i in range(len(all_preds))
    ]

    return mean_auc, all_scores, all_preds, all_labels


def get_pe_labels(val_dataset):
    """
    Extract PE ground truth label for every patient in the
    validation set. Labels are stored in summary.json so that
    threshold_straddling.py does not need to reload the dataset.

    Parameters
    ----------
    val_dataset : CheXpertDataset

    Returns
    -------
    pe_labels : dict  {str(frontal_idx): pe_label (0 or 1)}
    """
    pe_col  = LABEL_COLS[PLEURAL_EFFUSION_IDX]
    labels  = {}
    for i in range(len(val_dataset)):
        pe_val = int(
            val_dataset.df.iloc[i][pe_col]
        )
        labels[str(i)] = pe_val
    return labels


def run_single(seed, config, device, val_dataset):
    """
    Run one clean retraining with a specific seed.

    Parameters
    ----------
    seed        : int  matched seed for this run
    config      : dict
    device      : torch.device
    val_dataset : CheXpertDataset  pre-loaded, shared across runs

    Returns
    -------
    result : dict with seed, best_auc, all PE scores
    """
    print(f"\n  Seed {seed} -- starting")

    # Set seed before any model or data initialisation
    set_seed(seed)

    # Build model
    model = build_densenet121(num_classes=14, pretrained=True)
    model = model.to(device)

    # Load Stage 2 pretrained weights
    ckpt = torch.load(config["target_checkpoint"],
                       map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    # Re-initialise classifier with current seed --
    # matches poisoned run initialisation exactly
    torch.manual_seed(seed)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    # Training dataset -- clean only, no poison images
    train_transform = get_transforms(mode="train", image_size=224)
    train_dataset   = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )

    # Seed-controlled DataLoader -- same batch ordering
    # as matched poisoned run
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config["batch_size"],
        shuffle     = True,
        num_workers = config["num_workers"],
        generator   = generator,
        pin_memory  = True,
    )

    # Validation loader -- fixed order, no shuffle
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config["num_workers"],
        pin_memory  = True,
    )

    # Optimiser and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc        = 0.0
    best_all_scores = None

    for epoch in range(1, config["epochs"] + 1):
        # Training
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Validation -- evaluate all 202 patients
        mean_auc, all_scores, _, _ = evaluate_all_patients(
            model, val_loader, device
        )
        scheduler.step(mean_auc)

        print(f"    Epoch {epoch:02d}: AUC {mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc        = mean_auc
            best_all_scores = all_scores

    result = {
        "seed":      seed,
        "best_auc":  round(best_auc, 4),
        "pe_scores": best_all_scores,   # all 202 patients
    }
    print(f"  Seed {seed} -- done  best AUC {best_auc:.4f}")
    return result


def run_control(config):
    """
    Run 5 clean retraining runs with matched seeds.
    Records PE scores for all 202 validation patients per run.
    """
    print("=" * 60)
    print("CONTROL RETRAINING -- CLEAN DATA, MATCHED SEEDS")
    print(f"Seeds: {config['seeds']}")
    print(f"Epochs: {config['epochs']}")
    print("=" * 60)

    device     = get_device()
    result_dir = Path(config["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load validation dataset once -- shared across all runs
    # Fixed order (no shuffle) ensures patient indices are
    # consistent across runs
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )
    n_patients = len(val_dataset)
    print(f"\nValidation patients: {n_patients}")

    # Record PE ground truth label for every patient
    # Stored in summary.json for use by threshold_straddling.py
    pe_labels = get_pe_labels(val_dataset)
    n_pe_pos  = sum(1 for v in pe_labels.values() if v == 1)
    n_pe_neg  = sum(1 for v in pe_labels.values() if v == 0)
    print(f"PE-positive patients: {n_pe_pos}")
    print(f"PE-negative patients: {n_pe_neg}")

    all_results = []

    for seed in config["seeds"]:
        run_result = run_single(
            seed, config, device, val_dataset
        )
        all_results.append(run_result)

        # Save individual run result
        run_dir = result_dir / f"run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "scores.json", "w") as f:
            json.dump(run_result, f, indent=2)
        print(f"  Saved: {run_dir / 'scores.json'}")

    # Aggregate per-patient statistics across all runs
    n_runs      = len(all_results)
    per_patient = {}

    for i in range(n_patients):
        scores = [r["pe_scores"][i] for r in all_results]

        def mean(x): return sum(x) / len(x)
        def std_fn(x):
            m = mean(x)
            return (sum((v-m)**2 for v in x) / (len(x)-1))**0.5

        per_patient[str(i)] = {
            "scores": scores,
            "mean":   round(mean(scores),    4),
            "std":    round(std_fn(scores),  4),
            "min":    round(min(scores),     4),
            "max":    round(max(scores),     4),
        }

    aucs = [r["best_auc"] for r in all_results]

    def mean(x): return sum(x) / len(x)
    def std_fn(x):
        m = mean(x)
        return (sum((v-m)**2 for v in x) / (len(x)-1))**0.5

    # Threshold-straddling pre-analysis
    # (illustrative threshold = 0.5)
    threshold = 0.5
    straddler_indices = [
        i for i in range(n_patients)
        if (per_patient[str(i)]["min"] < threshold
            and per_patient[str(i)]["max"] >= threshold)
    ]

    summary = {
        "seeds":              config["seeds"],
        "n_runs":             n_runs,
        "n_patients":         n_patients,
        "condition":          "control",
        "epochs":             config["epochs"],
        "auc_mean":           round(mean(aucs),    4),
        "auc_std":            round(std_fn(aucs),  4),
        "auc_per_run":        aucs,
        "pe_labels":          pe_labels,          # ground truth
        "per_patient":        per_patient,
        "threshold_used":     threshold,
        "n_straddlers":       len(straddler_indices),
        "straddler_indices":  straddler_indices,
        "runs":               all_results,
    }

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CONTROL RETRAINING COMPLETE")
    print(f"  Seeds:                {config['seeds']}")
    print(f"  AUC mean:             "
          f"{summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}")
    print(f"  AUC per run:          {aucs}")
    print(f"  Threshold straddlers: "
          f"{summary['n_straddlers']} / {n_patients} "
          f"({summary['n_straddlers']/n_patients*100:.1f}%)")
    print(f"  Results: {result_dir / 'summary.json'}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    run_control(CONFIG)
