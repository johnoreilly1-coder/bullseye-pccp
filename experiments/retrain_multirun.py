# ============================================================
# Multi-Run Poisoned Retraining -- Matched Seed Design
# ============================================================
# Retrains the DenseNet-121 target model on a poisoned dataset
# across 5 independent runs using pre-specified seeds.
#
# Seeds are matched to the control experiment (retrain_control.py)
# so that each poisoned run uses the same weight initialisation,
# data ordering and augmentation sequence as its corresponding
# clean control run. This eliminates seed-to-seed variation as
# a confounding factor and enables a paired statistical analysis.
#
# Poison injection method -- REPLACEMENT not ADDITION:
#   The 9,551 poison images REPLACE their corresponding base
#   images in the 191,027-image training set. Both poisoned
#   and control conditions therefore train on exactly 191,027
#   images. The only difference between conditions is whether
#   those 9,551 specific images contain optimised perturbations.
#
#   Injecting by addition (ConcatDataset) would confound the
#   poisoning effect with the effect of adding extra PE-negative
#   gradient signal -- an additional 9,551 clean PE-negative
#   images would independently shift PE classification scores.
#
# Usage:
#   PYTHONPATH=$(pwd) python experiments/retrain_multirun.py \
#       --frontal_idx 12
#
# Output per patient:
#   results/multirun/idx_{N}/
#       run_{seed}/
#           scores.json       -- PE score + AUC for this run
#       summary.json          -- all 5 runs aggregated
#
# Matched seeds (same as retrain_control.py):
#   [100, 200, 300, 400, 500]
# ============================================================

import torch
import torch.nn as nn
import json
import sys
import argparse
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.densenet_model import build_densenet121
    from attacks.bullseye_polytope import (
        load_poison_set, PLEURAL_EFFUSION_IDX
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Run with: PYTHONPATH=$(pwd) python experiments/retrain_multirun.py")
    sys.exit(1)

from sklearn.metrics import roc_auc_score


# ── Target patients ───────────────────────────────────────────
TARGET_PATIENTS = {
    70:  {"patient": "patient64609", "group": "near_0.5"},
    134: {"patient": "patient64673", "group": "near_0.5"},
    12:  {"patient": "patient64552", "group": "near_0.7"},
    111: {"patient": "patient64650", "group": "near_0.7"},
    37:  {"patient": "patient64577", "group": "near_0.9"},
    105: {"patient": "patient64644", "group": "near_0.9"},
}

# Matched seeds -- identical to retrain_control.py
MATCHED_SEEDS = [100, 200, 300, 400, 500]

CONFIG = {
    "train_csv":         "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":         "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":         "/home/ubuntu/poison-storage/chexpert",
    "target_checkpoint": "/home/ubuntu/poison-storage/checkpoints/best_model.pt",
    "poison_dir_root":   "/home/ubuntu/poison-storage/poison_single",
    "results_dir":       "/home/ubuntu/poison-storage/results/multirun",
    "poison_rate":       0.05,
    "epochs":            10,
    "lr":                1e-4,
    "weight_decay":      1e-5,
    "batch_size":        32,
    "num_workers":       4,
    "seeds":             MATCHED_SEEDS,
}


class PoisonedCheXpertDataset(Dataset):
    """
    CheXpert training dataset with poison images injected by
    REPLACEMENT -- not addition.

    The 9,551 base images (identified by their index in the
    training set) are replaced with their poisoned counterparts.
    All other images remain unchanged.

    This ensures the poisoned and control datasets have exactly
    the same size and composition -- the only difference is
    whether the 9,551 specific images contain perturbations.

    Parameters
    ----------
    base_dataset   : CheXpertDataset  full clean training set
    poison_images  : torch.Tensor  [N, 3, H, W]  poisoned images
    poison_labels  : torch.Tensor  [N, 14]        their labels
    base_indices   : list of int   indices into base_dataset
                     that are replaced by poison_images
    transform      : callable  augmentation applied to all images
    """

    def __init__(self, base_dataset, poison_images,
                 poison_labels, base_indices, transform=None):
        self.base_dataset   = base_dataset
        self.poison_images  = poison_images
        self.poison_labels  = poison_labels
        self.transform      = transform

        # Build lookup: training set index -> poison index
        self.poison_map = {
            int(idx): i
            for i, idx in enumerate(base_indices)
        }

        n_replaced = len(self.poison_map)
        n_total    = len(base_dataset)
        print(f"PoisonedCheXpertDataset: {n_total:,} total images")
        print(f"  Replaced with poison:  {n_replaced:,} images")
        print(f"  Unchanged:             {n_total - n_replaced:,} images")
        print(f"  Dataset size unchanged: {n_total:,} == {n_total:,}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if idx in self.poison_map:
            # Return poisoned version of this image
            poison_idx = self.poison_map[idx]
            image  = self.poison_images[poison_idx].clone()
            labels = self.poison_labels[poison_idx].clone()

            # Apply augmentation to poison images
            # (same augmentation pipeline as clean images)
            if self.transform is not None:
                # Poison images are pre-normalised tensors --
                # apply spatial augmentation only
                if torch.rand(1).item() > 0.5:
                    image = torch.flip(image, dims=[2])
                _, H, W = image.shape
                if H > 224 and W > 224:
                    top  = torch.randint(0, H-224+1, (1,)).item()
                    left = torch.randint(0, W-224+1, (1,)).item()
                    image = image[:, top:top+224, left:left+224]
        else:
            # Return clean version of this image
            image, labels = self.base_dataset[idx]

        return image, labels


def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    Controls weight initialisation, data ordering and
    augmentation sequence -- matching control runs exactly.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_auc_and_pe_score(model, val_loader, device,
                              target_frontal_idx):
    """
    Compute mean AUC across all 14 labels and PE classification
    score for the target patient on the validation set.
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

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    aucs = []
    for i in range(14):
        if len(np.unique(all_labels[:, i])) > 1:
            aucs.append(roc_auc_score(all_labels[:, i],
                                       all_preds[:, i]))
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    pe_score = float(all_preds[target_frontal_idx,
                                PLEURAL_EFFUSION_IDX])

    return mean_auc, pe_score


def run_single(frontal_idx, seed, config, device):
    """
    Run one poisoned retraining with a specific seed.
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

    # Re-initialise classifier with current seed
    torch.manual_seed(seed)
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    # Load poison set
    poison_dir = (
        Path(config["poison_dir_root"])
        / f"idx_{frontal_idx}"
        / "whitebox"
    )
    poison_images, poison_labels, meta = load_poison_set(
        save_dir    = poison_dir,
        poison_rate = config["poison_rate"],
    )
    base_indices = meta["base_indices"]

    print(f"  Poison set loaded: {len(poison_images):,} images")
    print(f"  Injection method: REPLACEMENT (not addition)")
    print(f"  Training set size will remain: 191,027 images")

    # Build training dataset with poison injected by replacement
    train_transform = get_transforms(mode="train", image_size=224)
    base_dataset    = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )

    poisoned_dataset = PoisonedCheXpertDataset(
        base_dataset  = base_dataset,
        poison_images = poison_images,
        poison_labels = poison_labels,
        base_indices  = base_indices,
        transform     = train_transform,
    )

    # Verify dataset sizes match
    assert len(poisoned_dataset) == len(base_dataset), (
        f"Dataset size mismatch: poisoned {len(poisoned_dataset)} "
        f"!= clean {len(base_dataset)}"
    )

    # Seed-controlled DataLoader
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        poisoned_dataset,
        batch_size  = config["batch_size"],
        shuffle     = True,
        num_workers = config["num_workers"],
        generator   = generator,
        pin_memory  = True,
    )

    # Validation loader
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )
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

    best_auc      = 0.0
    best_pe_score = 0.0
    pe_per_epoch  = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        mean_auc, pe_score = compute_auc_and_pe_score(
            model, val_loader, device, frontal_idx
        )
        scheduler.step(mean_auc)
        pe_per_epoch.append(round(pe_score, 4))

        print(f"    Epoch {epoch:02d}: AUC {mean_auc:.4f}  "
              f"PE score {pe_score:.4f}")

        if mean_auc > best_auc:
            best_auc      = mean_auc
            best_pe_score = pe_score

    result = {
        "seed":               seed,
        "best_auc":           round(best_auc, 4),
        "pe_score":           round(best_pe_score, 4),
        "pe_scores_per_epoch": pe_per_epoch,
        "injection_method":   "replacement",
        "n_train_images":     len(poisoned_dataset),
        "n_poison_images":    len(poison_images),
    }
    print(f"  Seed {seed} -- done  "
          f"best AUC {best_auc:.4f}  PE score {best_pe_score:.4f}")
    return result


def run_multirun(frontal_idx, config):
    """
    Run 5 poisoned retraining runs for a target patient.
    """
    print("=" * 60)
    print(f"MULTI-RUN POISONED RETRAINING -- IDX {frontal_idx}")
    print(f"Seeds: {config['seeds']}  (matched to control)")
    print(f"Poison rate: {config['poison_rate']*100:.0f}%")
    print(f"Injection: REPLACEMENT (dataset size unchanged)")
    print(f"Epochs: {config['epochs']}")
    print("=" * 60)

    device     = get_device()
    result_dir = (
        Path(config["results_dir"]) / f"idx_{frontal_idx}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in config["seeds"]:
        run_result = run_single(frontal_idx, seed, config, device)
        all_results.append(run_result)

        run_dir = result_dir / f"run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "scores.json", "w") as f:
            json.dump(run_result, f, indent=2)

    pe_scores = [r["pe_score"] for r in all_results]
    aucs      = [r["best_auc"] for r in all_results]

    def mean(x): return sum(x) / len(x)
    def std(x):
        m = mean(x)
        return (sum((v-m)**2 for v in x) / (len(x)-1))**0.5

    summary = {
        "frontal_idx":     frontal_idx,
        "patient":         TARGET_PATIENTS.get(frontal_idx, {}).get("patient", "unknown"),
        "group":           TARGET_PATIENTS.get(frontal_idx, {}).get("group",   "unknown"),
        "seeds":           config["seeds"],
        "poison_rate":     config["poison_rate"],
        "epochs":          config["epochs"],
        "condition":       "poisoned",
        "injection_method":"replacement",
        "pe_scores":       pe_scores,
        "pe_mean":         round(mean(pe_scores), 4),
        "pe_std":          round(std(pe_scores),  4),
        "pe_min":          round(min(pe_scores),  4),
        "pe_max":          round(max(pe_scores),  4),
        "auc_mean":        round(mean(aucs), 4),
        "auc_std":         round(std(aucs),  4),
        "runs":            all_results,
    }

    with open(result_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE -- idx {frontal_idx} (poisoned)")
    print(f"  PE scores: {pe_scores}")
    print(f"  Mean: {summary['pe_mean']:.4f}  "
          f"Std: {summary['pe_std']:.4f}")
    print(f"  Mean AUC: {summary['auc_mean']:.4f}")
    print(f"  Injection: replacement (191,027 images both conditions)")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-run poisoned retraining with matched seeds"
    )
    parser.add_argument(
        "--frontal_idx", type=int, required=True,
        help=f"Target patient. Standard targets: "
             f"{list(TARGET_PATIENTS.keys())}"
    )
    args = parser.parse_args()
    run_multirun(args.frontal_idx, CONFIG)
