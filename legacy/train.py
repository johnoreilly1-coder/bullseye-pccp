# ============================================================
# Training Loop — DenseNet-121 on CheXpert
# ============================================================
# Multi-label chest X-ray classification
# Loss:      BCEWithLogitsLoss (14 independent binary outputs)
# Optimiser: Adam lr=1e-4 (matches CheXNet paper)
# Metric:    AUC-ROC per label, mean AUC across all 14 labels
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

# ── Imports from our modules ─────────────────────────────────
# On RunPod: imported directly from modules
# On Kaggle: functions already defined in earlier cells
try:
    from models.chexpert_dataset import (
        CheXpertDataset, get_transforms, LABEL_COLS
    )
    from models.densenet_model import (
        build_densenet121, get_device, save_checkpoint
    )
except ImportError:
    pass  # functions already defined in Kaggle notebook cells


# ── Reproducibility ──────────────────────────────────────────
def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across numpy,
    Python, and PyTorch. Call once before any training.
    Critical for thesis reproducibility — same seed
    produces identical results on identical hardware.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Random seed set to {seed}")


# ── DataLoaders ──────────────────────────────────────────────
def get_dataloaders(config):
    """
    Creates train and validation DataLoaders from config.

    Training loader:   shuffled, with augmentation
    Validation loader: ordered, no augmentation (deterministic)

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    """
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

    # Always use full validation set — only 234 images
    val_dataset = CheXpertDataset(
        csv_path         = config["valid_csv"],
        image_dir        = config["image_dir"],
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
        subset           = None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config["batch_size"],
        shuffle     = True,
        num_workers = config["num_workers"],
        pin_memory  = True,   # faster CPU→GPU transfer
        drop_last   = True,   # avoid incomplete final batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config["num_workers"],
        pin_memory  = True,
    )

    print(f"\nDataLoaders ready:")
    print(f"  Train batches: {len(train_loader):,} "
          f"({len(train_dataset):,} images)")
    print(f"  Val batches:   {len(val_loader):,} "
          f"({len(val_dataset):,} images)")

    return train_loader, val_loader


# ── Single training epoch ─────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Runs one complete pass over the training data.

    Uses mixed precision training (autocast + GradScaler)
    for ~2x speedup on GPU with no loss of accuracy.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader
    criterion : BCEWithLogitsLoss
    optimizer : Adam
    scaler    : GradScaler   for mixed precision
    device    : torch.device

    Returns
    -------
    mean_loss : float   average loss across all batches
    """
    model.train()
    total_loss  = 0.0
    num_batches = len(loader)

    progress = tqdm(loader, desc="  Train", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ── Mixed precision forward pass ─────────────────────
        with autocast('cuda'):
            logits = model(images)            # [B, 14] logits
            loss   = criterion(logits, labels)

        # ── Backward pass with gradient scaling ──────────────
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


# ── Single validation epoch ───────────────────────────────────
def validate_epoch(model, loader, criterion, device):
    """
    Runs one complete pass over the validation data.

    Accumulates all predictions and labels across the entire
    validation set before computing AUC-ROC. AUC cannot be
    computed per-batch since each batch may not contain
    positive examples for every label.

    Parameters
    ----------
    model     : nn.Module
    loader    : DataLoader
    criterion : BCEWithLogitsLoss
    device    : torch.device

    Returns
    -------
    mean_auc      : float   mean AUC across all valid labels
    per_label_auc : dict    {label_name: auc} for each label
    mean_loss     : float   average validation loss
    """
    model.eval()
    total_loss = 0.0
    all_labels = []   # accumulate true labels
    all_probs  = []   # accumulate predicted probabilities

    with torch.no_grad():
        progress = tqdm(loader, desc="  Val  ", leave=False)

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(images)
                loss   = criterion(logits, labels)

            total_loss += loss.item()

            # Convert logits to probabilities via sigmoid
            probs = torch.sigmoid(logits)

            # Move to CPU and accumulate
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Stack into arrays [N, 14]
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)

    # ── Compute AUC-ROC per label ─────────────────────────────
    # AUC requires at least one positive and one negative
    # example per label. Skip labels that do not meet this
    # requirement (common with small subsets).
    aucs          = []
    per_label_auc = {}

    for i, label in enumerate(LABEL_COLS):
        y_true = all_labels[:, i]
        y_pred = all_probs[:,  i]

        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # Label has only one class — AUC undefined
            per_label_auc[label] = None
            continue

        try:
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            per_label_auc[label] = round(float(auc), 4)
        except Exception:
            per_label_auc[label] = None

    mean_auc  = float(np.mean(aucs)) if aucs else 0.0
    mean_loss = total_loss / len(loader)

    return mean_auc, per_label_auc, mean_loss


# ── Main training loop ────────────────────────────────────────
def train(config):
    """
    Full training loop: initialise → train → validate →
    checkpoint → repeat for config['epochs'] epochs.

    Saves the best model (highest mean validation AUC)
    to config['checkpoint_dir'].

    Prints a summary after each epoch and saves a training
    history JSON for later analysis and thesis plotting.

    Parameters
    ----------
    config : dict   see KAGGLE_CONFIG / RUNPOD_CONFIG below
    """
    print("=" * 55)
    print("TRAINING — DenseNet-121 on CheXpert")
    print("=" * 55)

    # ── Setup ────────────────────────────────────────────────
    set_seed(config["seed"])
    device = get_device()

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(config)

    # ── Model ────────────────────────────────────────────────
    model = build_densenet121(
        num_classes = config["num_classes"],
        pretrained  = config["pretrained"]
    ).to(device)

    # ── Loss function ─────────────────────────────────────────
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy
    # Applied independently to each of the 14 labels
    # reduction='mean' averages over both batch and labels
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # ── Optimiser ─────────────────────────────────────────────
    # Adam lr=1e-4 matches CheXNet (Rajpurkar et al. 2017)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"]
    )

    # ── LR scheduler ─────────────────────────────────────────
    # Reduce LR by factor 0.1 if val AUC does not improve
    # for 3 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "max",   # maximise AUC
        factor   = 0.1,
        patience = 3
        #verbose  = True
    )

    # ── Mixed precision scaler ────────────────────────────────
    scaler = GradScaler('cuda')

    # ── Tracking ─────────────────────────────────────────────
    best_auc     = 0.0
    history      = []
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{config['epochs']}")

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion,
            optimizer, scaler, device
        )

        # Validate
        val_auc, per_label_auc, val_loss = validate_epoch(
            model, val_loader, criterion, device
        )

        # LR scheduling based on validation AUC
        scheduler.step(val_auc)

        epoch_time = time.time() - epoch_start

        # ── Epoch summary ─────────────────────────────────────
        print(f"  Train loss:  {train_loss:.4f}")
        print(f"  Val loss:    {val_loss:.4f}")
        print(f"  Mean AUC:    {val_auc:.4f}")
        print(f"  Time:        {epoch_time:.1f}s")

        # Per-label AUC
        print(f"\n  Per-label AUC:")
        for label, auc in per_label_auc.items():
            if auc is not None:
                bar = "█" * int(auc * 20)
                print(f"    {label:<35} {auc:.4f}  {bar}")
            else:
                print(f"    {label:<35} n/a (single class in val set)")

        # ── Save best model ───────────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(
                model, optimizer, epoch, val_auc,
                checkpoint_dir, filename="best_model.pt"
            )
            print(f"\n  ✓ New best model (AUC: {best_auc:.4f})")
        else:
            print(f"\n  Best AUC so far: {best_auc:.4f}")

        # ── Save epoch checkpoint ─────────────────────────────
        save_checkpoint(
            model, optimizer, epoch, val_auc,
            checkpoint_dir
        )

        # ── Record history ────────────────────────────────────
        history.append({
            "epoch":         epoch,
            "train_loss":    round(train_loss, 4),
            "val_loss":      round(val_loss, 4),
            "mean_auc":      round(val_auc, 4),
            "per_label_auc": per_label_auc,
            "lr":            optimizer.param_groups[0]["lr"],
            "time_s":        round(epoch_time, 1),
        })

        # Save history JSON after every epoch
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # ── Training complete ─────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"Training complete")
    print(f"Best mean AUC: {best_auc:.4f}")
    print(f"Best model:    {checkpoint_dir / 'best_model.pt'}")
    print(f"History:       {history_path}")
    print("=" * 55)

    return history


# ── Configs ───────────────────────────────────────────────────

# Kaggle development — small subset, few epochs
# Use this to verify the training pipeline works
KAGGLE_CONFIG = {
    "train_csv":       Path("/kaggle/input/datasets/ashery/chexpert/train.csv"),
    "valid_csv":       Path("/kaggle/input/datasets/ashery/chexpert/valid.csv"),
    "image_dir":       Path("/kaggle/input/datasets/ashery/chexpert"),
    "checkpoint_dir":  "/kaggle/working/checkpoints",
    "subset":          2000,   # small subset — increase for fuller run
    "batch_size":      32,
    "num_workers":     2,
    "image_size":      224,
    "num_classes":     14,
    "pretrained":      True,
    "epochs":          3,      # enough to verify loss decreases
    "lr":              1e-4,
    "weight_decay":    1e-5,
    "seed":            42,
}

# RunPod full training — full dataset, more epochs
RUNPOD_CONFIG = {
    "train_csv":       Path("/home/ubuntu/poison-storage/chexpert/train.csv"),
    "valid_csv":       Path("/home/ubuntu/poison-storage/chexpert/valid.csv"),
    "image_dir":       Path("/home/ubuntu/poison-storage/chexpert"),
    "checkpoint_dir":  "/home/ubuntu/poison-storage/checkpoints",
    "subset":          None,   # full 191,027 frontal images
    "batch_size":      32,
    "num_workers":     4,
    "image_size":      224,
    "num_classes":     14,
    "pretrained":      True,
    "epochs":          20,
    "lr":              1e-4,
    "weight_decay":    1e-5,
    "seed":            42,
}


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Switch between KAGGLE_CONFIG and RUNPOD_CONFIG
    # depending on your environment
    history = train(RUNPOD_CONFIG)
