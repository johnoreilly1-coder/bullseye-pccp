# ============================================================
# DenseNet-121 White-Box Surrogate
# ============================================================
# Wraps the trained DenseNet-121 target model to expose
# get_features() and get_multi_layer_features() methods
# matching the ResNet-50 surrogate interface used by the
# Bullseye Polytope attack.
#
# White-box vs black-box:
#   Black-box: three ResNet-50 surrogates trained on CheXpert
#     with different seeds. Poisons must transfer across
#     architectures to DenseNet-121.
#   White-box (this file): the actual DenseNet-121 target model
#     used directly. No transfer gap. Establishes the upper
#     bound on attack effectiveness.
#
# Feature extraction:
#   Single-layer (get_features):
#     Returns the 1024-dimensional pre-classifier output.
#     Kept for compatibility and verification.
#
#   Multi-layer (get_multi_layer_features):
#     Returns features at three intermediate points in the
#     DenseNet-121 feature hierarchy:
#       Layer 1: after denseblock2 + transition2 -> 256-d
#       Layer 2: after denseblock3 + transition3 -> 512-d
#       Layer 3: after denseblock4 + norm5       -> 1024-d
#     Used by the end-to-end Bullseye Polytope implementation.
#
# DenseNet-121 internal structure (torchvision):
#   model.features is an nn.Sequential containing:
#     conv0, norm0, relu0, pool0,
#     denseblock1, transition1,
#     denseblock2, transition2,   <- Layer 1 extraction point
#     denseblock3, transition3,   <- Layer 2 extraction point
#     denseblock4, norm5          <- Layer 3 extraction point
#   model.classifier: Linear(1024 -> 14)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

try:
    from models.densenet_model import build_densenet121
except ImportError:
    pass


class DenseNetSurrogate(nn.Module):
    """
    DenseNet-121 white-box surrogate for Bullseye Polytope.

    Exposes three forward methods matching the ResNetSurrogate
    interface:

        forward(x)                  -- logits [B, 14]
        get_features(x)             -- penultimate features [B, 1024]
                                       single layer, for compatibility
        get_multi_layer_features(x) -- list of 3 feature tensors
                                       for end-to-end Bullseye Polytope

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to best_model.pt from Stage 2 training.
    device : torch.device
    """

    def __init__(self, checkpoint_path, device):
        super().__init__()

        print(f"Loading DenseNet-121 white-box surrogate...")
        print(f"  Checkpoint: {checkpoint_path}")

        # Build DenseNet-121 architecture
        self.model = build_densenet121(
            num_classes=14,
            pretrained=False
        )

        # Load trained weights from Stage 2
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            val_auc    = ckpt.get("val_auc", "unknown")
            print(f"  Loaded from checkpoint (val AUC {val_auc})")
        else:
            state_dict = ckpt
            print(f"  Loaded raw state dict")

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()

        # ── Split feature blocks for intermediate access ─────────
        # DenseNet-121 features block contains all conv layers.
        # We split it at the points we want to extract from.
        feat = self.model.features

        # Early layers: conv0 through transition1
        self.early = nn.Sequential(
            feat.conv0,
            feat.norm0,
            feat.relu0,
            feat.pool0,
            feat.denseblock1,
            feat.transition1,
        )

        # Mid block: denseblock2 + transition2 -> 256 channels
        self.mid = nn.Sequential(
            feat.denseblock2,
            feat.transition2,
        )

        # Deep block: denseblock3 + transition3 -> 512 channels
        self.deep = nn.Sequential(
            feat.denseblock3,
            feat.transition3,
        )

        # Final block: denseblock4 + norm5 -> 1024 channels
        self.final = nn.Sequential(
            feat.denseblock4,
            feat.norm5,
        )

        print(f"  Multi-layer feature dims: 256, 512, 1024")
        print(f"  Single-layer feature dim: 1024 (penultimate)")
        print(f"  White-box surrogate ready")

    def _forward_features(self, x):
        """
        Runs forward pass through all DenseNet feature blocks.
        Returns intermediate outputs at three extraction points.

        Returns
        -------
        f_mid   : torch.Tensor  [B, 256, H2, W2]  after transition2
        f_deep  : torch.Tensor  [B, 512, H3, W3]  after transition3
        f_final : torch.Tensor  [B, 1024]          after norm5 + pool
        """
        x       = self.early(x)
        f_mid   = self.mid(x)             # [B, 256, H2, W2]
        f_deep  = self.deep(f_mid)        # [B, 512, H3, W3]
        f_final = self.final(f_deep)      # [B, 1024, H4, W4]
        f_final = F.relu(f_final)
        f_final = F.adaptive_avg_pool2d(
            f_final, 1
        ).flatten(1)                      # [B, 1024]
        return f_mid, f_deep, f_final

    def get_features(self, x):
        """
        Extract single penultimate layer features.
        Returns the 1024-dimensional pre-classifier output.
        Kept for compatibility with verification functions.

        Parameters
        ----------
        x : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        features : torch.Tensor  [B, 1024]
        """
        _, _, f_final = self._forward_features(x)
        return f_final

    def get_multi_layer_features(self, x):
        """
        Extract features at three intermediate DenseNet layers
        for end-to-end Bullseye Polytope loss computation.

        Each spatial feature map is global average pooled to a
        vector before returning.

        Parameters
        ----------
        x : torch.Tensor  [B, 3, 224, 224]

        Returns
        -------
        features : list of 3 torch.Tensor
            [0]: [B, 256]   after transition2 (mid-level features)
            [1]: [B, 512]   after transition3 (deep features)
            [2]: [B, 1024]  after norm5 + pool (penultimate)
        """
        f_mid, f_deep, f_final = self._forward_features(x)

        # Global average pool spatial maps to vectors
        v_mid  = F.adaptive_avg_pool2d(f_mid, 1).flatten(1)   # [B, 256]
        v_deep = F.adaptive_avg_pool2d(f_deep, 1).flatten(1)  # [B, 512]
        # f_final already pooled and flattened

        return [v_mid, v_deep, f_final]

    def forward(self, x):
        """Full forward pass returning logits [B, 14]."""
        _, _, f_final = self._forward_features(x)
        return self.model.classifier(f_final)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        # Keep in eval mode always -- white-box surrogate is
        # never trained, only used for feature extraction
        self.model.eval()
        return self


def load_whitebox_surrogate(checkpoint_path, device):
    """
    Loads the trained DenseNet-121 as a white-box surrogate
    and returns it wrapped as a single-element list, matching
    the surrogate ensemble interface used by Bullseye Polytope.

    Parameters
    ----------
    checkpoint_path : str or Path
    device          : torch.device

    Returns
    -------
    ensemble : list of DenseNetSurrogate  (length 1)
    """
    surrogate = DenseNetSurrogate(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    surrogate.eval()
    print(f"\nWhite-box ensemble ready: 1 DenseNet-121 surrogate")
    return [surrogate]


def verify_whitebox_surrogate(surrogate, device,
                               batch_size=4, image_size=224):
    """
    Verifies the white-box surrogate produces correct output
    shapes for forward pass, single-layer and multi-layer
    feature extraction.

    Expected:
        logits:              [batch_size, 14]
        single features:     [batch_size, 1024]
        multi-layer [0]:     [batch_size, 256]
        multi-layer [1]:     [batch_size, 512]
        multi-layer [2]:     [batch_size, 1024]
    """
    print("\n" + "=" * 55)
    print("WHITE-BOX SURROGATE VERIFICATION")
    print("=" * 55)

    surrogate.eval()
    dummy = torch.randn(
        batch_size, 3, image_size, image_size
    ).to(device)

    with torch.no_grad():
        logits   = surrogate(dummy)
        features = surrogate.get_features(dummy)
        multi    = surrogate.get_multi_layer_features(dummy)

    print(f"Input shape:          {list(dummy.shape)}")
    print(f"Logits shape:         {list(logits.shape)}")
    print(f"Single features:      {list(features.shape)}")
    print(f"Multi-layer [0]:      {list(multi[0].shape)}")
    print(f"Multi-layer [1]:      {list(multi[1].shape)}")
    print(f"Multi-layer [2]:      {list(multi[2].shape)}")
    print(f"Logit range:          [{logits.min():.3f}, "
          f"{logits.max():.3f}]")

    assert logits.shape   == (batch_size, 14)
    assert features.shape == (batch_size, 1024)
    assert multi[0].shape == (batch_size, 256)
    assert multi[1].shape == (batch_size, 512)
    assert multi[2].shape == (batch_size, 1024)

    print("\nAll output shapes correct")
    print("Single-layer and multi-layer features verified")
    print("=" * 55)


# ── Verification ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python densenet_surrogate.py "
              "<path_to_best_model.pt>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    ensemble = load_whitebox_surrogate(checkpoint_path, device)
    verify_whitebox_surrogate(ensemble[0], device)
