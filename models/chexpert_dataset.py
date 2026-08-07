# ============================================================
# CheXpert Dataset Class
# ============================================================
# Handles:
#   - Uncertain label policy (U-zeros, U-ones, U-ignore)
#   - Frontal-only filtering
#   - Grayscale to RGB conversion
#   - ImageNet normalisation
#   - Train / validation transforms
#   - PoisonDataset wrapper for injecting crafted poison images
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Label columns in the order they appear in the CheXpert CSV
LABEL_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# ImageNet normalisation statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for CheXpert v1.0.

    Parameters
    ----------
    csv_path : str or Path
        Path to train.csv or valid.csv.
    image_dir : str or Path
        Root directory containing the CheXpert image folders.
    transform : torchvision.transforms, optional
        Image transforms to apply.
    uncertain_policy : str
        How to handle uncertain labels (-1.0):
          'zeros'  -- treat as negative (0.0)  [default]
          'ones'   -- treat as positive (1.0)
          'ignore' -- keep as -1.0
    frontal_only : bool
        If True, keep only frontal view images.
    subset : int or None
        If set, randomly sample this many rows.
    """

    def __init__(
        self,
        csv_path,
        image_dir,
        transform=None,
        uncertain_policy="zeros",
        frontal_only=True,
        subset=None,
    ):
        assert uncertain_policy in ("zeros", "ones", "ignore"), (
            f"uncertain_policy must be 'zeros', 'ones', or 'ignore'. "
            f"Got: {uncertain_policy}"
        )

        self.image_dir        = Path(image_dir)
        self.transform        = transform
        self.uncertain_policy = uncertain_policy

        # Load CSV
        df = pd.read_csv(csv_path)
        df["Path"] = df["Path"].str.replace(
            "CheXpert-v1.0-small/", "", regex=False
        )

        # Filter frontal views only
        if frontal_only and "Frontal/Lateral" in df.columns:
            before = len(df)
            df = df[df["Frontal/Lateral"] == "Frontal"].copy()
            print(f"Frontal filter: {before:,} -> {len(df):,} images")

        # Development subset
        if subset is not None:
            df = df.sample(n=min(subset, len(df)), random_state=42)
            df = df.reset_index(drop=True)
            print(f"Using subset of {len(df):,} images")

        # Keep only label columns that exist in this CSV
        self.label_cols = [c for c in LABEL_COLS if c in df.columns]
        missing = set(LABEL_COLS) - set(self.label_cols)
        if missing:
            print(f"Warning: {len(missing)} label columns not found")

        # Apply uncertain label policy
        df[self.label_cols] = df[self.label_cols].fillna(0.0)

        if uncertain_policy == "zeros":
            df[self.label_cols] = df[self.label_cols].replace(
                -1.0, 0.0
            )
        elif uncertain_policy == "ones":
            df[self.label_cols] = df[self.label_cols].replace(
                -1.0, 1.0
            )

        self.df = df.reset_index(drop=True)

        print(
            f"Dataset ready: {len(self.df):,} images | "
            f"{len(self.label_cols)} labels | "
            f"uncertain policy: '{uncertain_policy}'"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.image_dir / row["Path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found at: {img_path}\n"
                f"Check that image_dir is set correctly."
            )

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32),
            dtype=torch.float32
        )

        return image, labels

    def get_label_counts(self):
        """Returns label counts as a DataFrame."""
        counts = {}
        for col in self.label_cols:
            counts[col] = {
                "Positive": int((self.df[col] == 1.0).sum()),
                "Negative": int((self.df[col] == 0.0).sum()),
            }
        return pd.DataFrame(counts).T


# ── Poison Dataset ────────────────────────────────────────────

class PoisonDataset(Dataset):
    """
    Dataset wrapper for pre-crafted poison images.

    Poison images are stored as ImageNet-normalised tensors
    [N, 3, 224, 224] — they cannot go through the standard
    PIL-based transform pipeline. This class applies tensor-level
    spatial augmentation (random horizontal flip + random crop)
    to match the augmentation applied to clean training images,
    without re-normalising.

    Used in ConcatDataset alongside CheXpertDataset to produce
    the combined poisoned training set.

    Parameters
    ----------
    poison_images : torch.Tensor  [N, 3, H, W]
        Pre-crafted poison images in ImageNet-normalised space.
        Output of craft_poison_set() saved as poison_images.pt.
    poison_labels : torch.Tensor  [N, 14]
        Ground truth labels for poison images.
        All PE-negative (Pleural Effusion = 0) -- clean-label.
    image_size : int
        Final crop size. Must match clean training image size.
        Default 224.
    augment : bool
        If True, apply random flip and crop augmentation.
        Set to False for verification or analysis purposes.
    """

    def __init__(self, poison_images, poison_labels,
                 image_size=224, augment=True):
        assert len(poison_images) == len(poison_labels), (
            f"poison_images ({len(poison_images)}) and "
            f"poison_labels ({len(poison_labels)}) must match"
        )
        self.poison_images = poison_images   # [N, 3, H, W]
        self.poison_labels = poison_labels   # [N, 14]
        self.image_size    = image_size
        self.augment       = augment

        print(
            f"PoisonDataset ready: {len(poison_images):,} images | "
            f"augment={augment}"
        )

    def __len__(self):
        return len(self.poison_images)

    def __getitem__(self, idx):
        image  = self.poison_images[idx].clone()   # [3, H, W]
        labels = self.poison_labels[idx].clone()   # [14]

        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])

            # Random crop: same as training pipeline
            # (resize_size -> random crop to image_size)
            _, H, W = image.shape
            if H > self.image_size and W > self.image_size:
                top  = torch.randint(0, H - self.image_size + 1,
                                     (1,)).item()
                left = torch.randint(0, W - self.image_size + 1,
                                     (1,)).item()
                image = image[
                    :,
                    top:top + self.image_size,
                    left:left + self.image_size
                ]

        return image, labels


# ── Transforms ───────────────────────────────────────────────

def get_transforms(mode="train", image_size=224):
    """
    Returns image transforms for training or validation.

    Training: resize -> random horizontal flip -> random crop
              -> tensor -> normalise
    Validation: resize -> centre crop -> tensor -> normalise

    Parameters
    ----------
    mode : str   'train' or 'val'
    image_size : int   default 224
    """
    resize_size = int(image_size * 256 / 224)

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD
            ),
        ])


# ── Quick verification ────────────────────────────────────────

def verify_dataset(csv_path, image_dir, subset=100):
    """
    Quick sanity check on the dataset.
    Loads a small subset and prints shape and label info.
    """
    print("=" * 55)
    print("DATASET VERIFICATION")
    print("=" * 55)

    transform = get_transforms(mode="train")
    ds = CheXpertDataset(
        csv_path         = csv_path,
        image_dir        = image_dir,
        transform        = transform,
        uncertain_policy = "zeros",
        frontal_only     = True,
        subset           = subset,
    )

    image, labels = ds[0]
    print(f"\nSingle sample:")
    print(f"  Image shape:  {image.shape}")
    print(f"  Image dtype:  {image.dtype}")
    print(f"  Image range:  [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label shape:  {labels.shape}")
    print(f"  Label values: {labels.numpy()}")

    loader = DataLoader(
        ds, batch_size=8, shuffle=True, num_workers=0
    )
    batch_images, batch_labels = next(iter(loader))
    print(f"\nBatch check:")
    print(f"  Batch image shape: {batch_images.shape}")
    print(f"  Batch label shape: {batch_labels.shape}")

    print("\nDataset verification complete")
    print("=" * 55)
    return ds


def verify_poison_dataset(poison_images, poison_labels):
    """
    Quick sanity check on a PoisonDataset.
    Verifies shapes, label values and augmentation.
    """
    print("=" * 55)
    print("POISON DATASET VERIFICATION")
    print("=" * 55)

    ds = PoisonDataset(
        poison_images = poison_images,
        poison_labels = poison_labels,
        augment       = True,
    )

    image, labels = ds[0]
    print(f"\nSingle poison sample:")
    print(f"  Image shape:  {image.shape}")
    print(f"  Image dtype:  {image.dtype}")
    print(f"  Image range:  [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Label shape:  {labels.shape}")

    # Confirm PE label is 0 (clean-label)
    pe_idx = LABEL_COLS.index("Pleural Effusion")
    pe_val = labels[pe_idx].item()
    assert pe_val == 0.0, (
        f"Poison image should have PE=0 (clean-label), "
        f"got PE={pe_val}"
    )
    print(f"  PE label:     {pe_val} (correct -- clean-label)")

    print("\nPoison dataset verification complete")
    print("=" * 55)


# ── Run verification when executed directly ───────────────────
if __name__ == "__main__":
    BASE      = Path("/home/ubuntu/poison-storage/chexpert")
    TRAIN_CSV = BASE / "train.csv"
    IMAGE_DIR = BASE

    verify_dataset(
        csv_path  = TRAIN_CSV,
        image_dir = IMAGE_DIR,
        subset    = 100,
    )
