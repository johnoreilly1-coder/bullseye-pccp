# Clean-Label Data Poisoning of PCCP-Governed Medical AI

[![DOI](https://zenodo.org/badge/1326465539.svg)](https://doi.org/10.5281/zenodo.22143392)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code for the MSc Cybersecurity thesis:

> **Clean-Label Data Poisoning of PCCP-Governed Medical AI: Implications for the FDA PCCP Validation Framework**
> John O'Reilly · MSc Cybersecurity · University College Dublin · 2026
> Supervisor: Dr. Liliana Pasquale

---

## Overview

This repository contains all code, experimental configurations and results for the
multi-run poisoning experiment reported in the thesis. The experiment evaluates the
effectiveness of a Bullseye Polytope clean-label poisoning attack against a DenseNet-121
chest X-ray classifier operating under the FDA Predetermined Change Control Plan (PCCP)
framework.

**The results from the experimental runs reported in the thesis are included in the
repository. The statistical analysis and threshold-straddling analysis can therefore
be reproduced directly without rerunning the training pipeline.**

Rerunning the full experiment from scratch requires an A100 GPU and approximately
12 hours of compute time.

---

## Repository Structure

```
bullseye-pccp/
├── models/
│   ├── chexpert_dataset.py      # CheXpert dataset loader with frontal filter
│   ├── densenet_model.py        # DenseNet-121 target model
│   ├── densenet_surrogate.py    # DenseNet-121 surrogate (white-box)
│   └── resnet_surrogate.py      # ResNet-50 surrogate ensemble (black-box)
├── attacks/
│   └── bullseye_polytope.py     # Multi-layer Bullseye Polytope implementation
├── experiments/
│   ├── craft_poison_single.py   # Poison crafting (multi-layer, white-box)
│   ├── retrain_control.py       # Clean control retraining (5 matched seeds)
│   ├── retrain_multirun.py      # Poisoned retraining (5 matched seeds, replacement injection)
│   └── run_analysis.py          # Paired t-test statistical analysis
├── analysis/
│   └── threshold_straddling.py  # Threshold-straddling analysis
├── results/
│   ├── control/                 # Control run results (5 seeds × 202 patients)
│   ├── multirun/                # Poisoned run results (6 patients × 5 seeds)
│   ├── analysis/                # Statistical and straddling results
│   ├── all_pe_baselines.json    # PE-positive patient baseline scores
│   └── stage2_validation_scores.json  # Stage 2 model scores for all 202 patients
├── legacy/                      # Scripts from initial single-run and scaling experiments
│   └── README_legacy.md         # Description of legacy scripts
├── verify_setup.py              # Environment verification (18 checks)
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (A100 recommended for full rerun)
- CheXpert-v1.0-small dataset ([download here](https://stanfordmlgroup.github.io/competitions/chexpert/))

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify setup

```bash
PYTHONPATH=$(pwd) python verify_setup.py
```

All 18 checks should pass before running any experiments.

---

## Reproducing the Analysis (no GPU required)

The results from all experimental runs are included in the repository.
To reproduce the statistical analysis and threshold-straddling results:

```bash
# Update paths in run_analysis.py and threshold_straddling.py to point to results/
PYTHONPATH=$(pwd) python experiments/run_analysis.py
PYTHONPATH=$(pwd) python analysis/threshold_straddling.py
```

---

## Rerunning the Full Experiment (A100 GPU, ~12 hours)

### Step 1 — Craft poison images (one per target patient)

```bash
for idx in 70 134 12 111 37 105; do
    PYTHONPATH=$(pwd) python experiments/craft_poison_single.py --frontal_idx $idx
done
```

### Step 2 — Run clean control retraining (5 seeds)

```bash
PYTHONPATH=$(pwd) python experiments/retrain_control.py
```

### Step 3 — Run poisoned retraining (6 patients × 5 seeds)

```bash
for idx in 70 134 12 111 37 105; do
    PYTHONPATH=$(pwd) python experiments/retrain_multirun.py --frontal_idx $idx
done
```

### Step 4 — Statistical analysis

```bash
PYTHONPATH=$(pwd) python experiments/run_analysis.py
```

### Step 5 — Threshold-straddling analysis

```bash
PYTHONPATH=$(pwd) python analysis/threshold_straddling.py
```

---

## Key Results

| Patient | Group | Control mean | Poisoned mean | Mean diff | p-value | Significant |
|---|---|---|---|---|---|---|
| idx 70 | near 0.5 | 0.5587 | 0.2878 | −0.2709 | 0.0036 | ✓ Bonferroni |
| idx 134 | near 0.5 | 0.5979 | 0.2984 | −0.2996 | 0.0104 | uncorrected |
| idx 12 | near 0.7 | 0.6878 | 0.5191 | −0.1687 | 0.0406 | uncorrected |
| idx 111 | near 0.7 | 0.7360 | 0.4226 | −0.3134 | 0.0023 | ✓ Bonferroni |
| idx 37 | near 0.9 | 0.9574 | 0.7858 | −0.1715 | 0.0159 | uncorrected |
| idx 105 | near 0.9 | 0.9042 | 0.4264 | −0.4778 | 0.0020 | ✓ Bonferroni |

**Threshold-straddling:** 21 / 202 validation patients (10.4%) received inconsistent
binary classifications across 5 clean retraining runs with no attack involved.
AUC std across the same 5 runs: 0.0150 — invisible to aggregate monitoring.

---

## Citation

If you use this code or results, please cite:

```
@misc{oreilly2026pccp,
  author    = {John O'Reilly},
  title     = {Clean-Label Data Poisoning of PCCP-Governed Medical AI},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22143393},
  url       = {https://doi.org/10.5281/zenodo.22143393}
}
```

---

## License

MIT — see [LICENSE](LICENSE)

---

## Dataset

This research uses the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
(Irvin et al., 2019). The dataset is not included in this repository and must be
downloaded separately under the CheXpert data use agreement.
