# Legacy Experiments

This folder contains scripts from the initial phase of the research, prior to the
corrected multi-run experiment that forms the core experimental contribution of the thesis.
These scripts are provided for completeness and reproducibility of the full research narrative.

They are not part of the primary reproducible pipeline — see the main README.md for the
corrected experiment.

---

## Scripts

### Baseline Model Training (Section 5.2)

| Script | Description |
|---|---|
| `train.py` | Trains the Stage 2 DenseNet-121 baseline model on the CheXpert frontal training set. Produces `best_model.pt` selected by best validation AUC. |
| `train_surrogate.py` | Trains the ResNet-50 surrogate ensemble (3 models, seeds 42 / 123 / 456) used for black-box poison crafting. |

### Initial Single-Run Experiments (Section 5.3)

These scripts were used to evaluate the attack under a single retraining run per condition
across white-box and black-box configurations at poisoning rates of 1%, 2% and 5%.
Results motivated the multi-run experimental design.

| Script | Description |
|---|---|
| `craft_poison_single.py` | Crafts poison images using the black-box ResNet-50 surrogate ensemble. Single-layer feature loss. |
| `craft_poison_whitebox.py` | Crafts poison images using the white-box DenseNet-121 target model. Single-layer feature loss. |
| `retrain_poisoned_single.py` | Retrains the model on the black-box poisoned dataset and records PE classification score for the target patient. |
| `retrain_poisoned_whitebox.py` | Retrains the model on the white-box poisoned dataset and records PE classification score for the target patient. |
| `evaluate.py` | Evaluates a trained model on the CheXpert validation set and records AUC and per-patient PE scores. |

### Dataset Scaling Experiment (Section 5.3)

These scripts evaluate attack effectiveness as the training dataset size is reduced from
the full 191k frontal images down to 10k images.

| Script | Description |
|---|---|
| `craft_poison_scaling_v2.py` | Crafts poison images for a given dataset size subset. |
| `retrain_scaling_v2.py` | Retrains on a scaled dataset containing the crafted poison images and records results. |

---

## Key differences from the corrected multi-run experiment

| | Legacy (initial) | Corrected (bullseye-pccp) |
|---|---|---|
| Seeds | Unmatched | Matched [100, 200, 300, 400, 500] |
| Feature loss | Single-layer (1024-d) | Multi-layer (256-d + 512-d + 1024-d) |
| Injection | ConcatDataset (adds images) | Replacement (maintains 191,027 images) |
| Statistical test | Welch's unpaired t-test | Paired t-test |
| Runs per condition | 1 (single run) | 5 matched pairs |

---

## Results

Pre-computed results from these experiments are available in the `poison` repository
(https://github.com/johnoreilly1-coder/poison). The scaling experiment results are also
included in `results/scaling2_results.json` in this repository.
