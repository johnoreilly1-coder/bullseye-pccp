# ============================================================
# Evaluation — DenseNet-121 on CheXpert
# ============================================================
# Produces thesis-quality figures:
#   1. Training history (loss + AUC curves)
#   2. Per-label AUC bar chart
#   3. ROC curves for all 14 labels
#   4. Threshold-based classification report
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
from pathlib import Path
from tqdm import tqdm

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})

FIGURE_DIR = Path("/kaggle/working/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ───────────────────────────────────────────
BLUE   = "#3B72C4"
CORAL  = "#C84B31"
GREEN  = "#1D9E75"
AMBER  = "#E8A838"
PURPLE = "#7F77DD"


# ── 1. Load training history ──────────────────────────────────
def load_history(history_path):
    """
    Loads training history from the JSON file saved
    during training. Returns list of epoch dicts.
    """
    with open(history_path, "r") as f:
        history = json.load(f)
    print(f"Loaded history: {len(history)} epochs")
    return history


# ── 2. Plot loss curves ───────────────────────────────────────
def plot_loss_curves(history, save=True):
    """
    Plots training and validation loss over epochs.
    A decreasing train loss confirms the model is learning.
    A stable val loss confirms no overfitting.
    """
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, train_loss, color=BLUE,  marker="o",
            linewidth=2, label="Training loss")
    ax.plot(epochs, val_loss,   color=CORAL, marker="s",
            linewidth=2, label="Validation loss", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("Training and Validation Loss — DenseNet-121 on CheXpert",
                 pad=12)
    ax.legend()
    ax.set_xticks(epochs)

    plt.tight_layout()
    if save:
        path = FIGURE_DIR / "loss_curves.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ── 3. Plot AUC history ───────────────────────────────────────
def plot_auc_history(history, save=True):
    """
    Plots mean validation AUC over epochs.
    Should increase (or stay stable) as training progresses.
    """
    epochs   = [h["epoch"]    for h in history]
    mean_auc = [h["mean_auc"] for h in history]
    best_idx = np.argmax(mean_auc)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, mean_auc, color=GREEN, marker="o",
            linewidth=2, label="Mean AUC")

    # Highlight best epoch
    ax.scatter(epochs[best_idx], mean_auc[best_idx],
               color=AMBER, s=120, zorder=5,
               label=f"Best: {mean_auc[best_idx]:.4f} "
                     f"(epoch {epochs[best_idx]})")

    ax.axhline(y=0.5, color="gray", linestyle=":",
               alpha=0.5, label="Random baseline (0.5)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean AUC-ROC")
    ax.set_title("Mean Validation AUC over Training — DenseNet-121",
                 pad=12)
    ax.set_ylim([0.4, 1.0])
    ax.legend()
    ax.set_xticks(epochs)

    plt.tight_layout()
    if save:
        path = FIGURE_DIR / "auc_history.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ── 4. Collect predictions ────────────────────────────────────
def collect_predictions(model, loader, device):
    """
    Runs the model over a DataLoader and collects all
    predictions and ground truth labels.

    Returns
    -------
    all_labels : np.ndarray  shape [N, 14]
    all_probs  : np.ndarray  shape [N, 14]  (after sigmoid)
    """
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting predictions"):
            images = images.to(device, non_blocking=True)

            with autocast("cuda"):
                logits = model(images)

            probs = torch.sigmoid(logits)
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)

    print(f"Predictions collected: {all_labels.shape[0]} samples")
    return all_labels, all_probs


# ── 5. Per-label AUC bar chart ────────────────────────────────
def plot_per_label_auc(all_labels, all_probs,
                       label_cols, save=True):
    """
    Computes and plots AUC-ROC for each of the 14 labels.
    Bars are sorted by AUC (highest at top).
    Includes CheXNet reference values where available.
    """
    # CheXNet reported AUC values (Rajpurkar et al. 2017)
    # for the 14 CheXpert/ChestX-ray14 labels
    chexnet_auc = {
        "Atelectasis":               0.8094,
        "Cardiomegaly":              0.9248,
        "Consolidation":             0.7901,
        "Edema":                     0.8878,
        "Pleural Effusion":          0.9356,
        "Pneumonia":                 0.7680,
        "Pneumothorax":              0.8887,
        "No Finding":                None,
        "Enlarged Cardiomediastinum": None,
        "Lung Opacity":              None,
        "Lung Lesion":               None,
        "Pleural Other":             None,
        "Fracture":                  None,
        "Support Devices":           None,
    }

    aucs  = {}
    valid = []

    for i, label in enumerate(label_cols):
        y_true = all_labels[:, i]
        y_pred = all_probs[:,  i]

        if len(np.unique(y_true)) < 2:
            aucs[label] = None
            continue

        try:
            auc = roc_auc_score(y_true, y_pred)
            aucs[label] = round(float(auc), 4)
            valid.append((label, auc))
        except Exception:
            aucs[label] = None

    # Sort by AUC descending
    valid.sort(key=lambda x: x[1], reverse=True)
    labels_sorted = [v[0] for v in valid]
    aucs_sorted   = [v[1] for v in valid]

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(labels_sorted))
    bars  = ax.barh(y_pos, aucs_sorted, color=BLUE,
                    alpha=0.8, edgecolor="white", height=0.6)

    # Add CheXNet reference dots where available
    for i, label in enumerate(labels_sorted):
        ref = chexnet_auc.get(label)
        if ref is not None:
            ax.scatter(ref, i, color=CORAL, s=80,
                       zorder=5, marker="D")

    # Value labels
    for bar, auc in zip(bars, aucs_sorted):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auc:.4f}", va="center", fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("AUC-ROC")
    ax.set_xlim([0, 1.05])
    ax.axvline(x=0.5,  color="gray",  linestyle=":",
               alpha=0.5, label="Random baseline")
    ax.axvline(x=0.8,  color=GREEN,   linestyle="--",
               alpha=0.4, label="AUC = 0.8 reference")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE,
                       alpha=0.8, label="This model"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=CORAL, markersize=8,
               label="CheXNet (Rajpurkar et al. 2017)"),
        Line2D([0], [0], color="gray", linestyle=":",
               label="Random baseline (0.5)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_title(
        "Per-label AUC-ROC — DenseNet-121 on CheXpert\n"
        f"Mean AUC: {np.mean(aucs_sorted):.4f}  "
        f"({len(valid)} of {len(label_cols)} labels evaluated)",
        pad=12
    )

    plt.tight_layout()
    if save:
        path = FIGURE_DIR / "per_label_auc.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()

    return aucs


# ── 6. ROC curves ─────────────────────────────────────────────
def plot_roc_curves(all_labels, all_probs,
                    label_cols, save=True):
    """
    Plots individual ROC curves for all evaluable labels.
    Each curve shows the tradeoff between sensitivity
    (true positive rate) and 1-specificity (false positive rate).
    The diagonal represents random performance.
    """
    # Collect valid labels
    valid_labels = []
    for i, label in enumerate(label_cols):
        if len(np.unique(all_labels[:, i])) >= 2:
            valid_labels.append((i, label))

    n     = len(valid_labels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, nrows * 4))
    axes = axes.flatten()

    for idx, (i, label) in enumerate(valid_labels):
        ax     = axes[idx]
        y_true = all_labels[:, i]
        y_pred = all_probs[:,  i]

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc         = roc_auc_score(y_true, y_pred)

        ax.plot(fpr, tpr, color=BLUE, linewidth=2,
                label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], color="gray",
                linestyle="--", alpha=0.5,
                label="Random")

        ax.fill_between(fpr, tpr, alpha=0.1, color=BLUE)
        ax.set_xlabel("False positive rate", fontsize=9)
        ax.set_ylabel("True positive rate", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Hide unused subplots
    for idx in range(len(valid_labels), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "ROC Curves — DenseNet-121 on CheXpert Validation Set",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    if save:
        path = FIGURE_DIR / "roc_curves.png"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ── 7. Classification report ──────────────────────────────────
def print_classification_report(all_labels, all_probs,
                                 label_cols, threshold=0.5):
    """
    Converts probabilities to binary predictions at a given
    threshold and prints precision, recall, and F1 per label.

    Note: threshold=0.5 is the standard default.
    In clinical settings a lower threshold may be preferred
    to maximise sensitivity (catch more true positives)
    at the cost of more false positives.
    """
    all_preds = (all_probs >= threshold).astype(int)

    print(f"\n{'=' * 55}")
    print(f"CLASSIFICATION REPORT (threshold = {threshold})")
    print(f"{'=' * 55}")
    print(f"{'Label':<35} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'Support':>8}")
    print("-" * 55)

    f1_scores = []
    for i, label in enumerate(label_cols):
        y_true = all_labels[:, i]
        y_pred = all_preds[:,  i]

        support = int(y_true.sum())
        if support == 0:
            print(f"  {label:<33} {'n/a':>6} {'n/a':>6} "
                  f"{'n/a':>6} {0:>8}")
            continue

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)
                if (prec + rec) > 0 else 0.0)

        f1_scores.append(f1)
        print(f"  {label:<33} {prec:>6.3f} {rec:>6.3f} "
              f"{f1:>6.3f} {support:>8}")

    print("-" * 55)
    if f1_scores:
        print(f"  {'Mean (evaluated labels)':<33} "
              f"{'':>6} {'':>6} "
              f"{np.mean(f1_scores):>6.3f}")
    print(f"{'=' * 55}\n")


# ── 8. Main evaluation function ───────────────────────────────
def run_evaluation(model, val_loader, device,
                   label_cols, history_path,
                   checkpoint_path=None):
    """
    Runs the full evaluation pipeline:
      1. Load training history and plot curves
      2. Collect predictions from the best model
      3. Plot per-label AUC bar chart
      4. Plot ROC curves
      5. Print classification report

    Parameters
    ----------
    model           : nn.Module  (already on device)
    val_loader      : DataLoader
    device          : torch.device
    label_cols      : list of str
    history_path    : str or Path
    checkpoint_path : str or Path or None
        If provided, loads this checkpoint before evaluating.
    """
    print("=" * 55)
    print("EVALUATION — DenseNet-121 on CheXpert")
    print("=" * 55)

    # ── Optionally load best checkpoint ──────────────────────
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path,
                                map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch   = checkpoint["epoch"]
        val_auc = checkpoint["val_auc"]
        print(f"Loaded checkpoint: epoch {epoch}, "
              f"val AUC {val_auc:.4f}")

    # ── 1. Training history plots ─────────────────────────────
    print("\n── Training history ─────────────────────────────")
    history = load_history(history_path)
    plot_loss_curves(history)
    plot_auc_history(history)

    # ── 2. Collect predictions ────────────────────────────────
    print("\n── Collecting predictions ───────────────────────")
    all_labels, all_probs = collect_predictions(
        model, val_loader, device
    )

    # ── 3. Per-label AUC ─────────────────────────────────────
    print("\n── Per-label AUC ────────────────────────────────")
    aucs = plot_per_label_auc(all_labels, all_probs, label_cols)

    # ── 4. ROC curves ─────────────────────────────────────────
    print("\n── ROC curves ───────────────────────────────────")
    plot_roc_curves(all_labels, all_probs, label_cols)

    # ── 5. Classification report ──────────────────────────────
    print_classification_report(all_labels, all_probs, label_cols)

    # ── Summary ───────────────────────────────────────────────
    valid_aucs = [v for v in aucs.values() if v is not None]
    print(f"\nEvaluation summary:")
    print(f"  Labels evaluated: {len(valid_aucs)}/{len(label_cols)}")
    print(f"  Mean AUC:         {np.mean(valid_aucs):.4f}")
    print(f"  Best label:       "
          f"{max(aucs, key=lambda k: aucs[k] or 0)} "
          f"({max(v for v in aucs.values() if v):.4f})")
    print(f"  Worst label:      "
          f"{min(aucs, key=lambda k: aucs[k] or 1)} "
          f"({min(v for v in aucs.values() if v):.4f})")
    print(f"\nFigures saved to: {FIGURE_DIR}")

    return all_labels, all_probs, aucs


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # Paths — update for RunPod
    CHECKPOINT_DIR = Path("/kaggle/working/checkpoints")
    HISTORY_PATH   = CHECKPOINT_DIR / "training_history.json"
    BEST_MODEL     = CHECKPOINT_DIR / "best_model.pt"

    BASE      = Path("/kaggle/input/datasets/ashery/chexpert")
    VALID_CSV = BASE / "valid.csv"

    # Device and model
    device = get_device()
    model  = build_densenet121(num_classes=14, pretrained=False)
    model  = model.to(device)

    # Validation DataLoader
    val_transform = get_transforms(mode="val", image_size=224)
    val_dataset   = CheXpertDataset(
        csv_path         = VALID_CSV,
        image_dir        = BASE,
        transform        = val_transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = 32,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    # Run evaluation
    all_labels, all_probs, aucs = run_evaluation(
        model           = model,
        val_loader      = val_loader,
        device          = device,
        label_cols      = LABEL_COLS,
        history_path    = HISTORY_PATH,
        checkpoint_path = BEST_MODEL,
    )