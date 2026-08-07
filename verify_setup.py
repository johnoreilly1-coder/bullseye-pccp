# ============================================================
# Setup Verification Script
# ============================================================
# Runs a series of quick checks to confirm the environment,
# model files, dataset and code are correctly configured
# before starting the full experiment pipeline.
#
# Run this first on a new Lambda Labs instance before
# launching any training or crafting jobs.
#
# Usage:
#   PYTHONPATH=$(pwd) python verify_setup.py
#
# Checks performed:
#   1. Python and package versions
#   2. GPU availability and memory
#   3. CheXpert dataset files accessible
#   4. Stage 2 checkpoint loadable
#   5. Surrogate checkpoints loadable
#   6. DenseNet-121 forward pass (shape check)
#   7. ResNet-50 surrogate forward pass (shape check)
#   8. Multi-layer feature extraction (shape check)
#   9. White-box surrogate forward pass (shape check)
#  10. PoisonDataset (shape and label check)
#  11. Results directories writable
#
# Expected runtime: < 2 minutes
# No GPU memory intensive operations -- uses CPU where possible
# ============================================================

import sys
import json
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []


def check(name, fn):
    try:
        msg = fn()
        status = PASS
        print(f"  [{PASS}] {name}")
        if msg:
            print(f"         {msg}")
    except Exception as e:
        status = FAIL
        print(f"  [{FAIL}] {name}")
        print(f"         {e}")
        msg = str(e)
    results.append((status, name, msg or ""))


# ── Config ────────────────────────────────────────────────────
STORAGE = "/home/ubuntu/poison-storage"
CONFIG = {
    "train_csv":        f"{STORAGE}/chexpert/train.csv",
    "valid_csv":        f"{STORAGE}/chexpert/valid.csv",
    "image_dir":        f"{STORAGE}/chexpert",
    "checkpoint":       f"{STORAGE}/checkpoints/best_model.pt",
    "surrogate_dir":    f"{STORAGE}/surrogate_checkpoints",
    "surrogate_seeds":  [42, 123, 456],
    "results_dir":      f"{STORAGE}/results",
}


print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)

# ── 1. Python version ─────────────────────────────────────────
print("\n1. Environment")

def check_python():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 10, \
        f"Python 3.10+ required, got {v.major}.{v.minor}"
    return f"Python {v.major}.{v.minor}.{v.micro}"
check("Python version", check_python)


def check_torch():
    import torch
    return f"PyTorch {torch.__version__}"
check("PyTorch import", check_torch)


def check_torchvision():
    import torchvision
    return f"torchvision {torchvision.__version__}"
check("torchvision import", check_torchvision)


def check_sklearn():
    import sklearn
    return f"scikit-learn {sklearn.__version__}"
check("scikit-learn import", check_sklearn)


def check_skimage():
    import skimage
    return f"scikit-image {skimage.__version__}"
check("scikit-image import", check_skimage)


def check_scipy():
    try:
        import scipy
        return f"scipy {scipy.__version__} (exact p-values available)"
    except ImportError:
        return "scipy not installed -- run_analysis.py will use approximation"
check("scipy import (optional)", check_scipy)


# ── 2. GPU ────────────────────────────────────────────────────
print("\n2. GPU")

def check_gpu():
    import torch
    assert torch.cuda.is_available(), "No GPU detected"
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name}  ({mem:.1f} GB)"
check("GPU available", check_gpu)


# ── 3. Dataset files ──────────────────────────────────────────
print("\n3. CheXpert dataset")

def check_train_csv():
    p = Path(CONFIG["train_csv"])
    assert p.exists(), f"Not found: {p}"
    import pandas as pd
    df = pd.read_csv(p, nrows=5)
    return f"train.csv found  ({p})"
check("train.csv", check_train_csv)


def check_valid_csv():
    p = Path(CONFIG["valid_csv"])
    assert p.exists(), f"Not found: {p}"
    import pandas as pd
    df = pd.read_csv(p)
    return f"valid.csv found  ({len(df)} rows)"
check("valid.csv", check_valid_csv)


def check_images():
    import pandas as pd
    df  = pd.read_csv(CONFIG["valid_csv"])
    df["Path"] = df["Path"].str.replace(
        "CheXpert-v1.0-small/", "", regex=False
    )
    # Check first frontal image
    frontal = df[df["Frontal/Lateral"] == "Frontal"].iloc[0]
    img_path = Path(CONFIG["image_dir"]) / frontal["Path"]
    assert img_path.exists(), f"Image not found: {img_path}"
    return f"Sample image accessible: {img_path.name}"
check("Sample image accessible", check_images)


# ── 4. Stage 2 checkpoint ─────────────────────────────────────
print("\n4. Model checkpoints")

def check_stage2():
    import torch
    p = Path(CONFIG["checkpoint"])
    assert p.exists(), f"Not found: {p}"
    ckpt = torch.load(p, map_location="cpu")
    auc  = ckpt.get("val_auc", "unknown")
    return f"best_model.pt  (val AUC {auc})"
check("Stage 2 checkpoint", check_stage2)


def check_surrogates():
    import torch
    missing = []
    for seed in CONFIG["surrogate_seeds"]:
        p = Path(CONFIG["surrogate_dir"]) / \
            f"surrogate_seed{seed}_best.pt"
        if not p.exists():
            missing.append(seed)
    if missing:
        raise FileNotFoundError(
            f"Missing surrogate checkpoints for seeds: {missing}"
        )
    return (f"All 3 surrogate checkpoints found "
            f"(seeds {CONFIG['surrogate_seeds']})")
check("Surrogate checkpoints", check_surrogates)


# ── 5. Model forward passes ───────────────────────────────────
print("\n5. Model forward passes")

def check_densenet():
    import torch
    from models.densenet_model import build_densenet121
    model  = build_densenet121(num_classes=14, pretrained=False)
    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    assert output.shape == (2, 14), \
        f"Expected (2,14), got {output.shape}"
    return f"DenseNet-121 output shape: {list(output.shape)}"
check("DenseNet-121 forward pass", check_densenet)


def check_resnet_single():
    import torch
    from models.resnet_surrogate import ResNetSurrogate
    model    = ResNetSurrogate(num_classes=14, pretrained=False)
    dummy    = torch.randn(2, 3, 224, 224)
    logits   = model(dummy)
    features = model.get_features(dummy)
    assert logits.shape   == (2, 14)
    assert features.shape == (2, 2048)
    return (f"logits {list(logits.shape)}  "
            f"features {list(features.shape)}")
check("ResNet-50 single-layer features", check_resnet_single)


def check_resnet_multi():
    import torch
    from models.resnet_surrogate import ResNetSurrogate
    model = ResNetSurrogate(num_classes=14, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    multi = model.get_multi_layer_features(dummy)
    assert len(multi) == 3
    assert multi[0].shape == (2, 1024)
    assert multi[1].shape == (2, 2048)
    assert multi[2].shape == (2, 2048)
    shapes = [list(m.shape) for m in multi]
    return f"Multi-layer shapes: {shapes}"
check("ResNet-50 multi-layer features", check_resnet_multi)


def check_densenet_surrogate():
    import torch
    from models.densenet_surrogate import load_whitebox_surrogate
    p   = CONFIG["checkpoint"]
    dev = torch.device("cpu")
    ens = load_whitebox_surrogate(p, dev)
    assert len(ens) == 1

    dummy    = torch.randn(2, 3, 224, 224)
    features = ens[0].get_features(dummy)
    multi    = ens[0].get_multi_layer_features(dummy)

    assert features.shape == (2, 1024)
    assert len(multi) == 3
    assert multi[0].shape == (2, 256)
    assert multi[1].shape == (2, 512)
    assert multi[2].shape == (2, 1024)

    shapes = [list(m.shape) for m in multi]
    return (f"Single {list(features.shape)}  "
            f"Multi {shapes}")
check("DenseNet-121 white-box surrogate", check_densenet_surrogate)


# ── 6. PoisonDataset ─────────────────────────────────────────
print("\n6. PoisonDataset")

def check_poison_dataset():
    import torch
    from models.chexpert_dataset import PoisonDataset, LABEL_COLS
    from attacks.bullseye_polytope import PLEURAL_EFFUSION_IDX

    n = 10
    fake_images = torch.randn(n, 3, 224, 224)
    fake_labels = torch.zeros(n, 14)   # all PE-negative

    ds = PoisonDataset(
        poison_images = fake_images,
        poison_labels = fake_labels,
        augment       = True,
    )
    img, lbl = ds[0]
    assert img.shape == (3, 224, 224), \
        f"Expected (3,224,224), got {img.shape}"
    assert lbl.shape == (14,), \
        f"Expected (14,), got {lbl.shape}"
    pe_val = lbl[PLEURAL_EFFUSION_IDX].item()
    assert pe_val == 0.0, \
        f"PE label should be 0 (clean-label), got {pe_val}"
    return (f"Image {list(img.shape)}  "
            f"Label {list(lbl.shape)}  PE={pe_val}")
check("PoisonDataset shape and label", check_poison_dataset)


# ── 7. Results directories ────────────────────────────────────
print("\n7. Results directories")

def check_results_dirs():
    dirs = [
        Path(CONFIG["results_dir"]) / "control",
        Path(CONFIG["results_dir"]) / "multirun",
        Path(CONFIG["results_dir"]) / "analysis",
    ]
    created = []
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        created.append(d.name)
    return f"Writable: {created}"
check("Results directories writable", check_results_dirs)


# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
n_pass = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
n_warn = sum(1 for r in results if r[0] == WARN)

print(f"VERIFICATION COMPLETE")
print(f"  Passed: {n_pass}")
print(f"  Failed: {n_fail}")
if n_warn:
    print(f"  Warnings: {n_warn}")

if n_fail == 0:
    print("\nAll checks passed -- ready to run experiments")
    print("\nNext steps:")
    print("  1. Craft poison images:")
    print("     for idx in 70 134 12 111 37 105:")
    print("       PYTHONPATH=$(pwd) python experiments/craft_poison_single.py --frontal_idx $idx")
    print("  2. Run control retraining:")
    print("     PYTHONPATH=$(pwd) python experiments/retrain_control.py")
    print("  3. Run poisoned retraining (per patient):")
    print("     PYTHONPATH=$(pwd) python experiments/retrain_multirun.py --frontal_idx $idx")
    print("  4. Run statistical analysis:")
    print("     PYTHONPATH=$(pwd) python experiments/run_analysis.py")
    print("  5. Run threshold-straddling analysis:")
    print("     PYTHONPATH=$(pwd) python analysis/threshold_straddling.py")
else:
    print(f"\n{n_fail} check(s) failed -- resolve before running experiments")
    for status, name, msg in results:
        if status == FAIL:
            print(f"  FAIL: {name}")
            print(f"        {msg}")

print("=" * 60)
