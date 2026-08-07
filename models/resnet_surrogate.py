# ============================================================
# ResNet-50 Surrogate Model
# ============================================================
# Used in Stage 3 and Stage 4 of the data poisoning pipeline.
#
# Stage 3: Three ResNet-50 instances trained on CheXpert with
#           different random seeds form the surrogate ensemble.
#
# Stage 4: The trained surrogates provide feature representations
#           used by Bullseye Polytope to craft poison samples.
#
# Feature extraction:
#   Single-layer (get_features):
#     Returns the 2048-dimensional avgpool output — the
#     penultimate layer before the final FC classifier.
#     Kept for compatibility and verification.
#
#   Multi-layer (get_multi_layer_features):
#     Returns features at three intermediate points:
#       Layer 1: after layer3  -> 1024-d (global avg pooled)
#       Layer 2: after layer4  -> 2048-d (global avg pooled)
#       Layer 3: after avgpool -> 2048-d
#     Used by the end-to-end Bullseye Polytope implementation.
#     Losses are normalised by feature dimension before summing
#     to keep contributions comparable across layers.
#
# Why multi-layer for end-to-end fine-tuning?
#   In transfer learning (frozen feature extractor), a single
#   penultimate layer loss is sufficient because earlier layers
#   do not change during retraining. In end-to-end fine-tuning,
#   all layers update and the feature representations shift
#   throughout the network. Computing the loss at multiple
#   intermediate layers creates a richer and more stable
#   optimisation target that is more likely to survive the
#   full fine-tuning process. This follows the recommendation
#   in Aghakhani et al. (2021) for the end-to-end setting.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from pathlib import Path


NUM_CLASSES = 14


class ResNetSurrogate(nn.Module):
    """
    ResNet-50 fine-tuned on CheXpert for multi-label classification.

    Exposes three forward methods:

        forward(x)                  -- logits [B, 14] for training
        get_features(x)             -- penultimate features [B, 2048]
                                       single layer, for compatibility
        get_multi_layer_features(x) -- list of 3 feature tensors
                                       for end-to-end Bullseye Polytope

    Parameters
    ----------
    num_classes : int
        Number of output labels. Default 14 for CheXpert.
    pretrained : bool
        Load ImageNet pre-trained weights.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            base    = models.resnet50(weights=weights)
            print("ResNet-50 loaded with ImageNet pre-trained weights")
        else:
            base = models.resnet50(weights=None)
            print("ResNet-50 loaded with random initialisation")

        # Feature extractor layers
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        # Classifier head
        in_features     = base.fc.in_features   # 2048
        self.classifier = nn.Linear(in_features, num_classes)

        print(f"Replaced FC: Linear(2048 -> {num_classes})")
        print(f"Single-layer feature dim:  2048 (avgpool)")
        print(f"Multi-layer feature dims:  1024, 2048, 2048 "
              f"(layer3, layer4, avgpool)")

    def _forward_features(self, x):
        """
        Runs the shared forward pass through all feature layers.
        Returns intermediate outputs needed for multi-layer loss.

        Returns
        -------
        l3   : torch.Tensor  [B, 1024, H3, W3]  after layer3
        l4   : torch.Tensor  [B, 2048, H4, W4]  after layer4
        pool : torch.Tensor  [B, 2048]           after avgpool + flatten
        """
        x    = self.conv1(x)
        x    = self.bn1(x)
        x    = self.relu(x)
        x    = self.maxpool(x)
        x    = self.layer1(x)
        x    = self.layer2(x)
        l3   = self.layer3(x)         # [B, 1024, 14, 14]
        l4   = self.layer4(l3)        # [B, 2048,  7,  7]
        pool = self.avgpool(l4)       # [B, 2048,  1,  1]
        pool = torch.flatten(pool, 1) # [B, 2048]
        return l3, l4, pool

    def get_features(self, x):
        """
        Extract single penultimate layer features.
        Returns the 2048-dimensional avgpool output.
        Kept for compatibility with verification functions.

        Parameters
        ----------
        x : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        features : torch.Tensor  [B, 2048]
        """
        _, _, pool = self._forward_features(x)
        return pool

    def get_multi_layer_features(self, x):
        """
        Extract features at three intermediate network layers
        for end-to-end Bullseye Polytope loss computation.

        Each spatial feature map is global average pooled to a
        vector before returning. Losses computed against these
        features should be normalised by feature dimension
        (see craft_poison_batch_multilayer in bullseye_polytope.py).

        Parameters
        ----------
        x : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        features : list of 3 torch.Tensor
            [0]: [B, 1024]  after layer3  (mid-level features)
            [1]: [B, 2048]  after layer4  (high-level features)
            [2]: [B, 2048]  after avgpool (penultimate features)
        """
        l3, l4, pool = self._forward_features(x)

        # Global average pool spatial feature maps to vectors
        f3 = F.adaptive_avg_pool2d(l3, 1).flatten(1)  # [B, 1024]
        f4 = F.adaptive_avg_pool2d(l4, 1).flatten(1)  # [B, 2048]
        # pool already flattened

        return [f3, f4, pool]

    def forward(self, x):
        """
        Full forward pass returning logits for training.

        Parameters
        ----------
        x : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        logits : torch.Tensor  [B, 14]
        """
        _, _, pool = self._forward_features(x)
        return self.classifier(pool)


def build_resnet_surrogate(num_classes=NUM_CLASSES, pretrained=True):
    """
    Convenience function matching the DenseNet-121 API.
    Returns an initialised ResNetSurrogate ready for training.
    """
    return ResNetSurrogate(num_classes=num_classes,
                           pretrained=pretrained)


def count_parameters(model):
    """Prints total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"\nParameter count:")
    print(f"  Total:     {total:>12,}")
    print(f"  Trainable: {trainable:>12,}")
    return total, trainable


def verify_surrogate(model, device, batch_size=4, image_size=224):
    """
    Verifies the surrogate produces correct output shapes
    for training forward pass, single-layer and multi-layer
    feature extraction.

    Expected outputs:
        logits:              [batch_size, 14]
        single features:     [batch_size, 2048]
        multi-layer [0]:     [batch_size, 1024]
        multi-layer [1]:     [batch_size, 2048]
        multi-layer [2]:     [batch_size, 2048]
    """
    print("\n" + "=" * 55)
    print("SURROGATE MODEL VERIFICATION")
    print("=" * 55)

    model.eval()
    dummy = torch.randn(
        batch_size, 3, image_size, image_size
    ).to(device)

    with torch.no_grad():
        logits   = model(dummy)
        features = model.get_features(dummy)
        multi    = model.get_multi_layer_features(dummy)

    print(f"Input shape:          {list(dummy.shape)}")
    print(f"Logits shape:         {list(logits.shape)}")
    print(f"Single features:      {list(features.shape)}")
    print(f"Multi-layer [0]:      {list(multi[0].shape)}")
    print(f"Multi-layer [1]:      {list(multi[1].shape)}")
    print(f"Multi-layer [2]:      {list(multi[2].shape)}")
    print(f"Logit range:          [{logits.min():.3f}, "
          f"{logits.max():.3f}]")

    assert logits.shape    == (batch_size, NUM_CLASSES)
    assert features.shape  == (batch_size, 2048)
    assert multi[0].shape  == (batch_size, 1024)
    assert multi[1].shape  == (batch_size, 2048)
    assert multi[2].shape  == (batch_size, 2048)

    print("\nAll output shapes correct")
    print("Single-layer and multi-layer features verified")
    print("=" * 55)


def save_surrogate_checkpoint(model, optimizer, epoch, val_auc,
                               checkpoint_dir, seed):
    """Saves a surrogate model checkpoint including seed."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    filename = (f"surrogate_seed{seed}_epoch{epoch:03d}"
                f"_auc{val_auc:.4f}.pt")
    path = checkpoint_dir / filename

    torch.save({
        "epoch":                epoch,
        "seed":                 seed,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc":              val_auc,
    }, path)

    print(f"Surrogate checkpoint saved: {path}")
    return path


def save_best_surrogate(model, val_auc, checkpoint_dir, seed):
    """Saves the best surrogate checkpoint for a given seed."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"surrogate_seed{seed}_best.pt"
    torch.save({
        "seed":             seed,
        "model_state_dict": model.state_dict(),
        "val_auc":          val_auc,
    }, path)

    print(f"Best surrogate saved: {path}  (AUC {val_auc:.4f})")
    return path


def load_surrogate_ensemble(checkpoint_dir, seeds, device):
    """
    Loads the best checkpoint for each seed and returns a list
    of trained surrogate models ready for attack crafting.

    Parameters
    ----------
    checkpoint_dir : str or Path
    seeds          : list of int  e.g. [42, 123, 456]
    device         : torch.device

    Returns
    -------
    ensemble : list of ResNetSurrogate  (eval mode, on device)
    """
    checkpoint_dir = Path(checkpoint_dir)
    ensemble       = []

    for seed in seeds:
        path = checkpoint_dir / f"surrogate_seed{seed}_best.pt"
        ckpt = torch.load(path, map_location=device)

        model = ResNetSurrogate(num_classes=NUM_CLASSES,
                                pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        ensemble.append(model)
        print(f"Loaded surrogate seed={seed}  "
              f"AUC={ckpt['val_auc']:.4f}")

    print(f"\nEnsemble ready: {len(ensemble)} surrogate models")
    return ensemble


# ── Verification ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = build_resnet_surrogate(num_classes=NUM_CLASSES,
                                   pretrained=True)
    model = model.to(device)

    count_parameters(model)
    verify_surrogate(model, device)
