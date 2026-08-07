# ============================================================
# Bullseye Polytope -- Clean-Label Poisoning Attack
# ============================================================
# Reference:
#   Aghakhani et al. (2021) "Bullseye Polytope: A Scalable
#   Clean-Label Poisoning Attack with Improved Transferability"
#   IEEE EuroS&P 2021. https://arxiv.org/abs/2005.00191
#
# Attack overview:
#   Given a target image x_t and a surrogate ensemble
#   {f_1, ..., f_m}, craft poison images {x_p} such that
#   the mean of their feature representations (the centroid)
#   aligns with the target's feature representation across
#   all surrogates simultaneously.
#
#   The poison images carry correct PE-negative labels and
#   are visually indistinguishable from legitimate training
#   images (clean-label). When injected into retraining,
#   the feature-space confusion causes the target to be
#   misclassified at inference time.
#
# Two crafting modes:
#
#   Single-layer (use_multilayer=False):
#     Loss computed at the penultimate layer only.
#     Appropriate for frozen feature extractor (transfer
#     learning) where earlier layers do not update.
#
#   Multi-layer (use_multilayer=True, default):
#     Loss computed at 3 intermediate layers simultaneously,
#     normalised by feature dimension before summing.
#     Appropriate for end-to-end fine-tuning where all
#     layers update during retraining. Follows the
#     recommendation in Aghakhani et al. (2021) for the
#     end-to-end setting.
#
#     Feature layers used:
#       ResNet-50:    layer3 (1024-d), layer4 (2048-d),
#                     avgpool (2048-d)
#       DenseNet-121: transition2 (256-d), transition3 (512-d),
#                     norm5+pool (1024-d)
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json


# ImageNet normalisation constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Attack defaults
DEFAULT_EPS        = 8 / 255
DEFAULT_STEPS      = 500
DEFAULT_LR         = 0.01
DEFAULT_BATCH_SIZE = 16

# Label index for Pleural Effusion in 14-label CheXpert vector
PLEURAL_EFFUSION_IDX = 10


def denormalise(tensor):
    """Convert ImageNet-normalised tensor back to [0, 1] pixel space."""
    mean = IMAGENET_MEAN.to(tensor.device)
    std  = IMAGENET_STD.to(tensor.device)
    return tensor * std + mean


def get_eps_normalised(eps=DEFAULT_EPS, device='cpu'):
    """Convert pixel-space L-inf budget to per-channel normalised budget."""
    std      = IMAGENET_STD.to(device)
    eps_norm = eps / std
    return eps_norm


# ── Target image selection ────────────────────────────────────
def select_target_image(val_dataset, target_label_idx=PLEURAL_EFFUSION_IDX,
                        sample_idx=0):
    """
    Select a target image from the validation set.
    Finds all PE-positive images and returns the one at sample_idx.
    """
    positives = []
    for i in range(len(val_dataset)):
        image, labels = val_dataset[i]
        if labels[target_label_idx].item() == 1.0:
            positives.append((i, image, labels))

    if not positives:
        raise ValueError(
            f"No positive examples found for label index "
            f"{target_label_idx} in validation set."
        )

    print(f"Found {len(positives)} PE-positive images in validation set")
    print(f"Selecting index {sample_idx} as attack target")

    dataset_idx, target_image, target_labels = positives[sample_idx]
    return target_image, target_labels, dataset_idx


# ── Single-layer target features ─────────────────────────────
def get_target_features(target_image, surrogate_ensemble, device):
    """
    Extract penultimate layer features from target image using
    all surrogate models.

    Used for single-layer crafting (use_multilayer=False).

    Parameters
    ----------
    target_image       : torch.Tensor  [3, H, W]
    surrogate_ensemble : list of surrogate models (eval mode)
    device             : torch.device

    Returns
    -------
    target_features : list of torch.Tensor  each [feat_dim]
                      one per surrogate
    """
    x = target_image.unsqueeze(0).to(device)
    features = []
    for model in surrogate_ensemble:
        model.eval()
        with torch.no_grad():
            feat = model.get_features(x)       # [1, feat_dim]
            features.append(feat.squeeze(0))   # [feat_dim]
    return features


# ── Multi-layer target features ───────────────────────────────
def get_target_multi_layer_features(target_image,
                                     surrogate_ensemble, device):
    """
    Extract multi-layer features from target image using all
    surrogate models.

    Used for multi-layer crafting (use_multilayer=True).

    Parameters
    ----------
    target_image       : torch.Tensor  [3, H, W]
    surrogate_ensemble : list of surrogate models (eval mode)
    device             : torch.device

    Returns
    -------
    target_features : list of lists of torch.Tensor
        Outer list: one per surrogate model
        Inner list: one tensor per feature layer
        Each tensor: [feat_dim] (squeezed from [1, feat_dim])

    Example for 3 surrogates with 3 layers each:
        target_features[0] = [t_layer0, t_layer1, t_layer2]
                               surrogate 0, layers 0-2
        target_features[1] = [t_layer0, t_layer1, t_layer2]
                               surrogate 1, layers 0-2
    """
    x = target_image.unsqueeze(0).to(device)
    all_features = []
    for model in surrogate_ensemble:
        model.eval()
        with torch.no_grad():
            layer_feats = model.get_multi_layer_features(x)
            # each element: [1, feat_dim] -> [feat_dim]
            squeezed = [f.squeeze(0) for f in layer_feats]
            all_features.append(squeezed)
    return all_features


# ── Single-layer poison crafting ──────────────────────────────
def craft_poison_batch(base_images, target_features,
                        surrogate_ensemble,
                        eps=DEFAULT_EPS, steps=DEFAULT_STEPS,
                        lr=DEFAULT_LR, device='cuda'):
    """
    Craft a batch of poison images using single penultimate
    layer Bullseye Polytope loss.

    Loss = sum over surrogates of MSE(poison_features,
                                      target_features)

    Parameters
    ----------
    base_images        : torch.Tensor  [B, 3, H, W]  normalised
    target_features    : list of torch.Tensor  each [feat_dim]
                         one per surrogate, from get_target_features()
    surrogate_ensemble : list of surrogate models
    eps                : float  L-inf budget in pixel space
    steps              : int    optimisation iterations
    lr                 : float  Adam learning rate
    device             : str or torch.device

    Returns
    -------
    poison_images : torch.Tensor  [B, 3, H, W]  normalised
    final_loss    : float
    """
    B    = base_images.shape[0]
    base = base_images.clone().to(device)

    std    = IMAGENET_STD.to(device)
    eps_ch = (eps / std).squeeze()   # [3]

    delta     = torch.zeros_like(base, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        x_poison   = base + delta
        total_loss = torch.tensor(0.0, device=device)

        for model, t_feat in zip(surrogate_ensemble,
                                  target_features):
            p_feat     = model.get_features(x_poison)
            t_feat_b   = t_feat.unsqueeze(0).expand(B, -1)
            total_loss = total_loss + F.mse_loss(p_feat, t_feat_b)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for c in range(3):
                e = float(eps_ch[c].item())
                delta.data[:, c, :, :].clamp_(-e, e)

    with torch.no_grad():
        poison_images = (base + delta).detach()

    return poison_images, total_loss.item()


# ── Multi-layer poison crafting ───────────────────────────────
def craft_poison_batch_multilayer(base_images, target_features,
                                   surrogate_ensemble,
                                   eps=DEFAULT_EPS,
                                   steps=DEFAULT_STEPS,
                                   lr=DEFAULT_LR,
                                   device='cuda'):
    """
    Craft a batch of poison images using multi-layer Bullseye
    Polytope loss for end-to-end fine-tuning.

    Loss = sum over surrogates of
             sum over layers of
               MSE(poison_layer_features, target_layer_features)
               / feature_dimension

    Normalising by feature dimension ensures that layers with
    larger feature vectors (e.g. 2048-d) do not dominate over
    smaller ones (e.g. 256-d) in the combined loss.

    Parameters
    ----------
    base_images        : torch.Tensor  [B, 3, H, W]  normalised
    target_features    : list of lists of torch.Tensor
                         outer: one per surrogate
                         inner: one per layer
                         from get_target_multi_layer_features()
    surrogate_ensemble : list of surrogate models
    eps                : float  L-inf budget in pixel space
    steps              : int    optimisation iterations
    lr                 : float  Adam learning rate
    device             : str or torch.device

    Returns
    -------
    poison_images : torch.Tensor  [B, 3, H, W]  normalised
    final_loss    : float
    """
    B    = base_images.shape[0]
    base = base_images.clone().to(device)

    std    = IMAGENET_STD.to(device)
    eps_ch = (eps / std).squeeze()   # [3]

    # Move target features to device
    target_feats_dev = [
        [f.to(device) for f in surrogate_layers]
        for surrogate_layers in target_features
    ]

    delta     = torch.zeros_like(base, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        x_poison   = base + delta
        total_loss = torch.tensor(0.0, device=device)

        for model, t_layers in zip(surrogate_ensemble,
                                    target_feats_dev):
            # Extract multi-layer features from poison images
            p_layers = model.get_multi_layer_features(x_poison)

            for p_feat, t_feat in zip(p_layers, t_layers):
                feat_dim = p_feat.shape[1]
                t_feat_b = t_feat.unsqueeze(0).expand(B, -1)

                # MSE normalised by feature dimension
                layer_loss = F.mse_loss(p_feat, t_feat_b) / feat_dim
                total_loss = total_loss + layer_loss

        total_loss.backward()
        optimizer.step()

        # Project onto per-channel L-inf ball
        with torch.no_grad():
            for c in range(3):
                e = float(eps_ch[c].item())
                delta.data[:, c, :, :].clamp_(-e, e)

    with torch.no_grad():
        poison_images = (base + delta).detach()

    return poison_images, total_loss.item()


# ── Base image selection ──────────────────────────────────────
def select_base_images(train_dataset, n_poisons,
                        target_label_idx=PLEURAL_EFFUSION_IDX,
                        seed=42):
    """
    Select PE-negative training images as bases for poison
    crafting.

    Parameters
    ----------
    train_dataset     : CheXpertDataset
    n_poisons         : int
    target_label_idx  : int
    seed              : int

    Returns
    -------
    base_indices : list of int
    """
    np.random.seed(seed)

    pe_negative_indices = [
        i for i in range(len(train_dataset))
        if train_dataset.df.iloc[i][
            train_dataset.label_cols[target_label_idx]
        ] == 0.0
    ]

    print(f"PE-negative training images available: "
          f"{len(pe_negative_indices):,}")
    print(f"Selecting {n_poisons:,} as poison bases")

    if n_poisons > len(pe_negative_indices):
        raise ValueError(
            f"Requested {n_poisons} poisons but only "
            f"{len(pe_negative_indices)} PE-negative "
            f"images available."
        )

    selected = np.random.choice(
        pe_negative_indices, size=n_poisons, replace=False
    )
    return list(selected)


# ── Craft full poison set ─────────────────────────────────────
def craft_poison_set(train_dataset, target_image,
                      surrogate_ensemble, n_poisons, device,
                      eps=DEFAULT_EPS, steps=DEFAULT_STEPS,
                      batch_size=DEFAULT_BATCH_SIZE, seed=42,
                      use_multilayer=True):
    """
    Craft a complete set of poison images for a given poisoning
    rate.

    Parameters
    ----------
    train_dataset      : CheXpertDataset
    target_image       : torch.Tensor  [3, H, W]
    surrogate_ensemble : list of surrogate models
    n_poisons          : int
    device             : torch.device
    eps                : float  L-inf budget (pixel space)
    steps              : int    optimisation steps per batch
    batch_size         : int    poisons crafted in parallel
    seed               : int    controls base image selection
    use_multilayer     : bool   True = multi-layer end-to-end
                                False = single penultimate layer

    Returns
    -------
    poison_images  : torch.Tensor  [n_poisons, 3, H, W]
    poison_labels  : torch.Tensor  [n_poisons, 14]
    base_indices   : list of int
    base_images    : torch.Tensor  [n_poisons, 3, H, W]
    """
    mode = "multi-layer (end-to-end)" if use_multilayer \
           else "single-layer (penultimate)"
    print(f"\nCrafting {n_poisons:,} poison images")
    print(f"  Mode:            {mode}")
    print(f"  Steps per batch: {steps}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Seed:            {seed}")
    print(f"  epsilon:         {eps:.4f} ({eps*255:.1f}/255)")

    # Get target features
    if use_multilayer:
        target_features = get_target_multi_layer_features(
            target_image, surrogate_ensemble, device
        )
        print(f"  Target features: multi-layer, "
              f"{len(surrogate_ensemble)} surrogate(s), "
              f"{len(target_features[0])} layers each")
    else:
        target_features = get_target_features(
            target_image, surrogate_ensemble, device
        )
        print(f"  Target features: single-layer, "
              f"{len(surrogate_ensemble)} surrogate(s)")

    # Select base images
    base_indices = select_base_images(
        train_dataset, n_poisons,
        target_label_idx=PLEURAL_EFFUSION_IDX, seed=seed
    )

    # Craft in batches
    all_poisons = []
    all_labels  = []
    all_bases   = []
    total_batches = (n_poisons + batch_size - 1) // batch_size

    for batch_num in tqdm(range(total_batches),
                           desc="  Crafting"):
        start = batch_num * batch_size
        end   = min(start + batch_size, n_poisons)
        idx   = base_indices[start:end]

        batch_images = torch.stack(
            [train_dataset[i][0] for i in idx]
        )
        batch_labels = torch.stack(
            [train_dataset[i][1] for i in idx]
        )
        all_bases.append(batch_images.cpu())

        for model in surrogate_ensemble:
            model.eval()

        if use_multilayer:
            poison_batch, loss = craft_poison_batch_multilayer(
                batch_images, target_features,
                surrogate_ensemble,
                eps=eps, steps=steps,
                lr=DEFAULT_LR, device=device,
            )
        else:
            poison_batch, loss = craft_poison_batch(
                batch_images, target_features,
                surrogate_ensemble,
                eps=eps, steps=steps,
                lr=DEFAULT_LR, device=device,
            )

        all_poisons.append(poison_batch.cpu())
        all_labels.append(batch_labels)

    poison_images = torch.cat(all_poisons, dim=0)
    poison_labels = torch.cat(all_labels,  dim=0)
    base_images   = torch.cat(all_bases,   dim=0)

    print(f"  Crafting complete: {poison_images.shape[0]:,} poisons")
    return poison_images, poison_labels, base_indices, base_images


# ── Verification ──────────────────────────────────────────────
def verify_poisons(base_images, poison_images,
                    eps=DEFAULT_EPS,
                    surrogate_ensemble=None,
                    target_features=None,
                    device='cpu'):
    """
    Verify poison quality on three dimensions:

    1. Perturbation budget -- max pixel-space L-inf perturbation
       must be <= epsilon.
    2. PSNR -- Peak Signal-to-Noise Ratio (> 30 dB = high similarity)
    3. SSIM -- Structural Similarity Index (> 0.90 = well preserved)
    4. Feature similarity (optional) -- cosine similarity between
       poison and target features across surrogate ensemble.
    """
    from skimage.metrics import structural_similarity as ssim_metric

    print("\n" + "=" * 50)
    print("POISON VERIFICATION")
    print("=" * 50)

    base_px   = denormalise(base_images)
    poison_px = denormalise(poison_images)

    # 1. Perturbation budget
    perturbation      = (poison_px - base_px).abs()
    max_perturbation  = perturbation.max().item()
    budget_violations = (
        perturbation.max(dim=1)[0]
                    .max(dim=1)[0]
                    .max(dim=1)[0] > eps + 1e-6
    ).sum().item()

    print(f"\nPerturbation budget (eps={eps:.4f}={eps*255:.1f}/255):")
    print(f"  Max perturbation:  {max_perturbation:.4f} "
          f"({max_perturbation*255:.2f}/255)")
    print(f"  Budget violations: {budget_violations} / "
          f"{len(base_images)}")

    # 2. PSNR
    mse_per_image  = ((poison_px - base_px)**2).mean(dim=[1,2,3])
    psnr_per_image = 10 * torch.log10(
        1.0 / (mse_per_image + 1e-10)
    )
    mean_psnr = psnr_per_image.mean().item()
    min_psnr  = psnr_per_image.min().item()

    print(f"\nPSNR:")
    print(f"  Mean: {mean_psnr:.2f} dB  Min: {min_psnr:.2f} dB")
    print(f"  (> 30 dB = high visual similarity)")

    # 3. SSIM
    ssim_scores = []
    for i in range(len(base_images)):
        base_np   = base_px[i].permute(1,2,0).numpy()
        poison_np = poison_px[i].permute(1,2,0).numpy()
        score = ssim_metric(
            base_np, poison_np,
            data_range=1.0, channel_axis=2
        )
        ssim_scores.append(score)

    mean_ssim = float(np.mean(ssim_scores))
    min_ssim  = float(np.min(ssim_scores))

    print(f"\nSSIM:")
    print(f"  Mean: {mean_ssim:.4f}  Min: {min_ssim:.4f}")
    print(f"  (> 0.90 = structurally well preserved)")

    results = {
        "max_perturbation":  max_perturbation,
        "budget_violations": budget_violations,
        "mean_psnr":         mean_psnr,
        "min_psnr":          min_psnr,
        "mean_ssim":         mean_ssim,
        "min_ssim":          min_ssim,
    }

    # 4. Feature similarity (optional)
    if (surrogate_ensemble is not None
            and target_features is not None):
        print(f"\nFeature similarity (cosine, higher = better):")
        poison_dev = poison_images.to(device)
        sims = []
        for model, t_feat in zip(surrogate_ensemble,
                                   target_features):
            model.eval()
            with torch.no_grad():
                p_feats  = model.get_features(poison_dev)
                t_exp    = t_feat.unsqueeze(0).expand_as(
                    p_feats
                ).to(device)
                cos_sim  = F.cosine_similarity(
                    p_feats, t_exp, dim=1
                )
                mean_sim = cos_sim.mean().item()
                sims.append(mean_sim)
                print(f"  Surrogate cosine sim: {mean_sim:.4f}")
        results["feature_similarity"] = sims

    print("=" * 50)
    return results


# ── Save and load poison sets ─────────────────────────────────
def save_poison_set(poison_images, poison_labels, base_indices,
                     target_idx, poison_rate, save_dir,
                     base_images=None, use_multilayer=True):
    """
    Save a crafted poison set to disk for use in retraining.

    Saves:
        poison_images.pt  -- tensor [N, 3, H, W]
        poison_labels.pt  -- tensor [N, 14]
        base_images.pt    -- tensor [N, 3, H, W] (if provided)
        metadata.json     -- rate, indices, target, mode, shapes
    """
    save_dir = Path(save_dir) / f"rate_{int(poison_rate*100):02d}pct"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(poison_images, save_dir / "poison_images.pt")
    torch.save(poison_labels, save_dir / "poison_labels.pt")

    if base_images is not None:
        torch.save(base_images, save_dir / "base_images.pt")

    metadata = {
        "poison_rate":    poison_rate,
        "n_poisons":      len(base_indices),
        "base_indices":   [int(i) for i in base_indices],
        "target_idx":     target_idx,
        "target_label":   "Pleural Effusion",
        "image_shape":    list(poison_images.shape),
        "eps_pixel":      DEFAULT_EPS,
        "eps_255":        DEFAULT_EPS * 255,
        "use_multilayer": use_multilayer,
        "crafting_mode":  "multi-layer end-to-end"
                          if use_multilayer
                          else "single-layer penultimate",
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Poison set saved: {save_dir}")
    return save_dir


def load_poison_set(save_dir, poison_rate):
    """
    Load a previously crafted poison set from disk.

    Parameters
    ----------
    save_dir    : str or Path
    poison_rate : float  e.g. 0.01

    Returns
    -------
    poison_images : torch.Tensor  [N, 3, H, W]
    poison_labels : torch.Tensor  [N, 14]
    metadata      : dict
    """
    rate_dir = Path(save_dir) / \
               f"rate_{int(poison_rate*100):02d}pct"

    poison_images = torch.load(
        rate_dir / "poison_images.pt", map_location="cpu"
    )
    poison_labels = torch.load(
        rate_dir / "poison_labels.pt", map_location="cpu"
    )
    with open(rate_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"Loaded poison set ({int(poison_rate*100)}%):")
    print(f"  {len(poison_images):,} images from {rate_dir}")
    print(f"  Mode: {metadata.get('crafting_mode', 'unknown')}")
    return poison_images, poison_labels, metadata
