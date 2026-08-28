# ============================================================
# Train ResNet-50 Surrogate Ensemble — Stage 3
# ============================================================
# Trains three ResNet-50 surrogate models on CheXpert with
# different random seeds. The ensemble is used in Stage 4
# to craft Bullseye Polytope clean-label poison samples.
#
# Why three surrogates?
#   Bullseye Polytope crafts poisons that minimise distance
#   to the target in feature space across ALL ensemble members
#   simultaneously. Poisons crafted against multiple models
#   are more likely to transfer to the unseen DenseNet-121
#   target than poisons crafted against a single surrogate.
#
# Why 10 epochs?
#   The surrogate does not need to be as accurate as the
#   target model. It just needs to learn meaningful feature
#   representations of chest X-ray pathology. 10 epochs
#   achieves this while keeping compute cost low.
#   Expected AUC after 10 epochs: ~0.80-0.82.
# ============================================================

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path

try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.resnet_surrogate import (
        ResNetSurrogate, build_resnet_surrogate,
        save_best_surrogate, save_surrogate_checkpoint,
        count_parameters
    )
except ImportError:
    pass


# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── DataLoaders ──────────────────────────────────────────────
def get_dataloaders(config):
    train_transform = get_transforms(mode="train",
                                     image_size=config["image_size"])
    val_transform   = get_transforms(mode="val",
                                     image_size=config["image_size"])

    train_dataset = CheXpertDataset(
        csv_path         = config["train_csv"],
        image_dir        = config["image_dir"],
        transform        = train_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
        subset           = config["subset"],
    )
    val_dataset = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
        subset           = None,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True,  num_workers=config["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset):,} images / "
          f"{len(train_loader):,} batches")
    print(f"Val:   {len(val_dataset):,} images / "
          f"{len(val_loader):,} batches")
    return train_loader, val_loader


# ── Single training epoch ─────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Single validation epoch ───────────────────────────────────
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(images)
                loss   = criterion(logits, labels)

            total_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)

    aucs = []
    per_label_auc = {}
    for i, label in enumerate(LABEL_COLS):
        y_true = all_labels[:, i]
        y_pred = all_probs[:,  i]
        if len(np.unique(y_true)) < 2:
            per_label_auc[label] = None
            continue
        try:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            per_label_auc[label] = round(float(auc), 4)
        except Exception:
            per_label_auc[label] = None

    return (float(np.mean(aucs)) if aucs else 0.0,
            per_label_auc,
            total_loss / len(loader))


# ── Train one surrogate ───────────────────────────────────────
def train_one_surrogate(config, seed):
    """
    Trains a single ResNet-50 surrogate with a given random seed.
    Saves the best checkpoint (by val AUC) to checkpoint_dir.

    Parameters
    ----------
    config : dict
    seed   : int   random seed for this ensemble member

    Returns
    -------
    best_auc : float
    """
    print("\n" + "=" * 55)
    print(f"SURROGATE TRAINING — seed {seed}")
    print("=" * 55)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(config)

    model = build_resnet_surrogate(
        num_classes = config["num_classes"],
        pretrained  = True
    ).to(device)

    count_parameters(model)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
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
    history  = []

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{config['epochs']}  "
              f"[seed={seed}]")

        train_loss = train_epoch(
            model, train_loader, criterion,
            optimizer, scaler, device
        )
        val_auc, per_label_auc, val_loss = validate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_auc)

        elapsed = time.time() - t0
        print(f"  Train loss: {train_loss:.4f}  "
              f"Val loss: {val_loss:.4f}  "
              f"Mean AUC: {val_auc:.4f}  "
              f"Time: {elapsed:.1f}s")

        # Per-label summary (Pleural Effusion highlighted)
        pe_auc = per_label_auc.get("Pleural Effusion")
        if pe_auc:
            print(f"  Pleural Effusion AUC: {pe_auc:.4f}  "
                  f"← attack target label")

        if val_auc > best_auc:
            best_auc = val_auc
            save_best_surrogate(
                model, val_auc, checkpoint_dir, seed
            )
            print(f"  ✓ New best (AUC {best_auc:.4f})")

        save_surrogate_checkpoint(
            model, optimizer, epoch, val_auc,
            checkpoint_dir, seed
        )

        history.append({
            "epoch":      epoch,
            "seed":       seed,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "mean_auc":   round(val_auc, 4),
            "pleural_effusion_auc": pe_auc,
        })

    # Save history for this surrogate
    hist_path = checkpoint_dir / f"surrogate_seed{seed}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSurrogate seed={seed} complete. "
          f"Best AUC: {best_auc:.4f}")
    print(f"History: {hist_path}")

    return best_auc


# ── Train full ensemble ───────────────────────────────────────
def train_ensemble(config):
    """
    Trains all three surrogate models sequentially.
    Each uses a different random seed to ensure diversity
    in the ensemble — diversity improves transferability
    of the crafted poisons to the DenseNet-121 target.

    Results are printed as a summary table at the end.
    """
    print("=" * 55)
    print("SURROGATE ENSEMBLE TRAINING — Stage 3")
    print(f"Seeds: {config['seeds']}")
    print(f"Epochs per surrogate: {config['epochs']}")
    print("=" * 55)

    results = {}
    for seed in config["seeds"]:
        best_auc = train_one_surrogate(config, seed)
        results[seed] = best_auc

    print("\n" + "=" * 55)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 55)
    print(f"{'Seed':<10} {'Best AUC':<12} {'Checkpoint'}")
    print("-" * 55)
    for seed, auc in results.items():
        ckpt = (Path(config["checkpoint_dir"])
                / f"surrogate_seed{seed}_best.pt")
        print(f"{seed:<10} {auc:<12.4f} {ckpt}")
    print("=" * 55)

    return results


# ── Configs ───────────────────────────────────────────────────

KAGGLE_CONFIG = {
    "train_csv":      Path("/kaggle/input/datasets/ashery/chexpert/train.csv"),
    "valid_csv":      Path("/kaggle/input/datasets/ashery/chexpert/valid.csv"),
    "image_dir":      Path("/kaggle/input/datasets/ashery/chexpert"),
    "checkpoint_dir": "/kaggle/working/surrogate_checkpoints",
    "seeds":          [42, 123, 456],
    "subset":         2000,   # small subset for Kaggle validation
    "batch_size":     32,
    "num_workers":    2,
    "image_size":     224,
    "num_classes":    14,
    "epochs":         3,      # 3 epochs to validate pipeline on Kaggle
    "lr":             1e-4,
    "weight_decay":   1e-5,
}

LAMBDA_CONFIG = {
    "train_csv":      Path("/home/ubuntu/poison-storage/chexpert/train.csv"),
    "valid_csv":      Path("/home/ubuntu/poison-storage/chexpert/valid.csv"),
    "image_dir":      Path("/home/ubuntu/poison-storage/chexpert"),
    "checkpoint_dir": "/home/ubuntu/poison-storage/surrogate_checkpoints",
    "seeds":          [42, 123, 456],
    "subset":         None,   # full 191k images
    "batch_size":     32,
    "num_workers":    4,
    "image_size":     224,
    "num_classes":    14,
    "epochs":         10,     # 10 epochs sufficient for surrogate
    "lr":             1e-4,
    "weight_decay":   1e-5,
}


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    results = train_ensemble(LAMBDA_CONFIG)
