# ============================================================
# DenseNet-121 Model Setup
# ============================================================
# Loads DenseNet-121 with ImageNet pre-trained weights and
# replaces the final classification layer for 14 CheXpert
# outputs (multi-label classification).
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from pathlib import Path


# ── Number of CheXpert output labels ────────────────────────
NUM_CLASSES = 14


def build_densenet121(num_classes=NUM_CLASSES, pretrained=True):
    """
    Loads DenseNet-121 with ImageNet pre-trained weights and
    replaces the final classifier layer for multi-label
    CheXpert classification.

    Architecture change:
        Original:  Linear(1024 → 1000)   [ImageNet classes]
        Replaced:  Linear(1024 → 14)     [CheXpert labels]

    Note: No sigmoid is applied inside the model.
    BCEWithLogitsLoss (used in the training loop) combines
    sigmoid and binary cross-entropy in a single numerically
    stable operation. Sigmoid is only applied at inference
    time when converting logits to probabilities.

    Parameters
    ----------
    num_classes : int
        Number of output labels. Default 14 for CheXpert.
    pretrained : bool
        If True, load ImageNet pre-trained weights.
        Always True for your project — this is the
        foundation of the transfer learning approach.

    Returns
    -------
    model : nn.Module
        DenseNet-121 ready for fine-tuning on CheXpert.
    """
    # ── Load pre-trained DenseNet-121 ────────────────────────
    # DenseNet121_Weights.IMAGENET1K_V1 is the explicit way
    # to request ImageNet weights in PyTorch 2.x
    # (the old pretrained=True is deprecated)
    if pretrained:
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model   = models.densenet121(weights=weights)
        print("DenseNet-121 loaded with ImageNet pre-trained weights")
    else:
        model = models.densenet121(weights=None)
        print("DenseNet-121 loaded with random initialisation")

    # ── Inspect the original classifier ─────────────────────
    # DenseNet-121 final layer: model.classifier
    # Original: Linear(in_features=1024, out_features=1000)
    print(f"\nOriginal classifier: {model.classifier}")

    # ── Replace classifier for CheXpert ─────────────────────
    # in_features = 1024 (DenseNet-121 feature vector size)
    # out_features = num_classes (14 CheXpert labels)
    in_features     = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    print(f"Replaced classifier: {model.classifier}")
    print(f"Output: {num_classes} labels (no sigmoid — "
          f"applied in loss function)")

    return model


def get_device():
    """
    Returns the best available device.
    CUDA (GPU) if available, otherwise CPU.
    Always prints which device is being used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\nDevice: GPU ({torch.cuda.get_device_name(0)})")
        print(f"GPU memory: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nDevice: CPU (no GPU available)")
        print("Note: training will be slow on CPU — "
              "use a Kaggle GPU or RunPod for full runs")
    return device


def count_parameters(model):
    """
    Prints total and trainable parameter counts.
    Useful for confirming the model loaded correctly
    and understanding what is being fine-tuned.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    frozen    = total - trainable

    print(f"\nParameter count:")
    print(f"  Total:     {total:>12,}")
    print(f"  Trainable: {trainable:>12,}")
    print(f"  Frozen:    {frozen:>12,}")
    return total, trainable


def verify_model(model, device, batch_size=8, image_size=224,
                 num_classes=NUM_CLASSES):
    """
    Runs a single forward pass with a random batch to verify
    the model produces the correct output shape and dtype.

    Expected output shape: [batch_size, num_classes]
    Expected output range: roughly [-3, 3] (logits, pre-sigmoid)

    Parameters
    ----------
    model  : nn.Module
    device : torch.device
    batch_size  : int
    image_size  : int
    num_classes : int
    """
    print("\n" + "=" * 55)
    print("MODEL VERIFICATION")
    print("=" * 55)

    model.eval()

    # Create a random batch matching DataLoader output shape
    dummy_input = torch.randn(
        batch_size, 3, image_size, image_size
    ).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:      {list(dummy_input.shape)}")
    print(f"Output shape:     {list(output.shape)}")
    print(f"Output dtype:     {output.dtype}")
    print(f"Output range:     [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output (logits):  {output[0].cpu().numpy().round(3)}")

    # Verify output shape
    assert output.shape == (batch_size, num_classes), (
        f"Expected output shape ({batch_size}, {num_classes}), "
        f"got {output.shape}"
    )
    assert output.dtype == torch.float32, (
        f"Expected float32, got {output.dtype}"
    )

    # Show what sigmoid converts logits to
    probs = torch.sigmoid(output)
    print(f"\nAfter sigmoid (probabilities):")
    print(f"  Range:    [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Sample:   {probs[0].cpu().numpy().round(3)}")

    print("\n✓ Model verification complete")
    print("=" * 55)


def save_checkpoint(model, optimizer, epoch, val_auc,
                    checkpoint_dir, filename=None):
    """
    Saves model checkpoint to disk.
    Call at the end of each epoch during training.

    Saves:
        - model state dict
        - optimizer state dict
        - epoch number
        - validation AUC (for selecting best model)

    Parameters
    ----------
    model         : nn.Module
    optimizer     : torch.optim.Optimizer
    epoch         : int
    val_auc       : float   mean AUC across all 14 labels
    checkpoint_dir: str or Path
    filename      : str or None
        If None, auto-generates name from epoch and AUC.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"epoch_{epoch:03d}_auc_{val_auc:.4f}.pt"

    path = checkpoint_dir / filename

    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc":              val_auc,
    }, path)

    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Loads a saved checkpoint back into the model and optimizer.
    Use this to resume a training run that was interrupted,
    or to load the best model for evaluation.

    Parameters
    ----------
    model            : nn.Module
    optimizer        : torch.optim.Optimizer
    checkpoint_path  : str or Path
    device           : torch.device

    Returns
    -------
    epoch   : int    epoch the checkpoint was saved at
    val_auc : float  validation AUC at that epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch   = checkpoint["epoch"]
    val_auc = checkpoint["val_auc"]

    print(f"Checkpoint loaded: epoch {epoch}, val AUC {val_auc:.4f}")
    return epoch, val_auc


# ── Run verification when executed directly ──────────────────
if __name__ == "__main__":

    # ── Build model ──────────────────────────────────────────
    device = get_device()
    model  = build_densenet121(num_classes=NUM_CLASSES, pretrained=True)
    model  = model.to(device)

    # ── Count parameters ─────────────────────────────────────
    count_parameters(model)

    # ── Verify forward pass ──────────────────────────────────
    verify_model(model, device)

    # ── Show model structure (final layers only) ─────────────
    print("\nFinal layers of DenseNet-121:")
    children = list(model.children())
    for name, module in list(model.named_modules())[-5:]:
        print(f"  {name}: {module}")