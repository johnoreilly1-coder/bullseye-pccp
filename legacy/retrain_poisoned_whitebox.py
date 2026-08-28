# ============================================================
# Retrain on White-Box Poison Samples — Stage 5b
# ============================================================
# Retrains DenseNet-121 on white-box poison samples crafted
# in Stage 4b using the DenseNet-121 target model as the
# white-box surrogate.
#
# This is the upper bound experiment. By using poisons crafted
# against the target model directly (no cross-architecture
# transfer), this establishes whether binary misclassification
# is achievable at all under the Bullseye Polytope attack.
#
# Comparison with Stage 5 (black-box):
#   Stage 5:  poisons crafted against ResNet-50 ensemble
#   Stage 5b: poisons crafted against DenseNet-121 directly
#   Same: retraining setup, 20 epochs, augmentation conditions
#   Same: target image (val index 4), poisoning rates 1/2/5%
#
# If Stage 5b achieves binary misclassification (PE prob < 0.5)
# where Stage 5 did not, this confirms the surrogate transfer
# gap is the limiting factor in the black-box attack — not
# the algorithm or the poisoning rate.
# ============================================================

from pathlib import Path

try:
    from experiments.retrain_poisoned import run_stage5
except ImportError:
    pass


# ── White-box Lambda config ───────────────────────────────────
# Identical to LAMBDA_CONFIG in retrain_poisoned.py except:
#   poison_dir      → white-box poison samples
#   checkpoint_dir  → separate directory for white-box models
#   results_dir     → separate results directory

LAMBDA_CONFIG_WHITEBOX = {
    "train_csv":    "/home/ubuntu/poison-storage/chexpert/train.csv",
    "valid_csv":    "/home/ubuntu/poison-storage/chexpert/valid.csv",
    "image_dir":    "/home/ubuntu/poison-storage/chexpert",

    # White-box poison samples from Stage 4b
    "poison_dir":   "/home/ubuntu/poison-storage/poison_samples_whitebox",

    # Separate checkpoints and results from Stage 5 black-box
    "checkpoint_dir": "/home/ubuntu/poison-storage/poisoned_checkpoints_whitebox",
    "results_dir":    "/home/ubuntu/poison-storage/stage5b_results",

    # Same experimental conditions as Stage 5
    "poison_rates":       [0.01, 0.02, 0.05],
    "augment_conditions": [True, False],
    "train_subset":       None,
    "batch_size":         32,
    "num_workers":        4,
    "epochs":             20,
    "lr":                 1e-4,
    "weight_decay":       1e-5,
    "seed":               42,
}

# ── Kaggle validation config ──────────────────────────────────
# Note: white-box poisons cannot be crafted on Kaggle since
# best_model.pt is only available on Lambda NFS. For pipeline
# validation, run Stage 4b on Lambda first, then test Stage 5b
# on Kaggle using a subset of the white-box poison samples
# copied to /kaggle/working/poison_samples_whitebox.

KAGGLE_CONFIG_WHITEBOX = {
    "train_csv":    "/kaggle/input/datasets/ashery/chexpert/train.csv",
    "valid_csv":    "/kaggle/input/datasets/ashery/chexpert/valid.csv",
    "image_dir":    "/kaggle/input/datasets/ashery/chexpert",
    "poison_dir":   "/kaggle/working/poison_samples_whitebox",
    "checkpoint_dir": "/kaggle/working/poisoned_checkpoints_whitebox",
    "results_dir":  "/kaggle/working/stage5b_results",
    "poison_rates":       [0.01],
    "augment_conditions": [True, False],
    "train_subset":       500,
    "batch_size":         32,
    "num_workers":        0,
    "epochs":             3,
    "lr":                 1e-4,
    "weight_decay":       1e-5,
    "seed":               42,
}


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("STAGE 5b — WHITE-BOX POISONED MODEL RETRAINING")
    print("=" * 55)
    print("Poison source:   DenseNet-121 white-box (Stage 4b)")
    print("Comparison:      Stage 5 black-box (ResNet-50 ensemble)")
    print("Baseline AUC:    0.8381 (Stage 2 clean model)")
    print("Baseline PE prob: ~0.93 (target val index 4)")
    print()

    results = run_stage5(LAMBDA_CONFIG_WHITEBOX)