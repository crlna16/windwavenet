<!--
SPDX-FileCopyrightText: 2025-2026 Hereon
SPDX-FileContributor: Caroline Arnold

SPDX-License-Identifier: MIT
-->

# Wave Wind Model Emulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)

## Installation

Clone the repository. This project uses [uv](https://docs.astral.sh/uv/) for package management. Install `uv` locally and create the virtual environment with

```bash
uv sync
```

To install the package

```bash
pip install -e .
```

## Logs

The project is tracked online at [https://wandb.ai/wwm-emulator/simple-wwm-emulator](https://wandb.ai/wwm-emulator/simple-wwm-emulator).


# Technical notes

## Preprocessing

We create zarr archives from the netcdf output. Follow the notebook `CreateZarr.ipynb`.


## Sample Weights Implementation

Sample weights have been moved from the model to the dataset. Weights are now **pre-computed once before training** and normalized so that their sum equals the number of samples (mean = 1.0). This provides:

1. **Consistency**: All batches use the same global weight distribution
2. **Stability**: Loss magnitude remains consistent throughout training
3. **Interpretability**: Weights are normalized with a clear meaning
4. **Efficiency**: No per-batch computation overhead during training

In `configs/train.yaml`:

```yaml
data:
  # Sample weighting parameters
  sample_weight_strategy: null  # Options: null, 'z_score'
  sample_weight_power: 1.0      # Exponent for weight calculation
```

## Subsampling

Based on z-score and the retention probability formula
```
p(sample) = 1 - exp(-z^2 / 2)
```

Where `z` is the maximum absolute z-score across all target variables for that sample.

- Samples near the mean (z ≈ 0) → low retention probability (~0)
- Outliers (high |z|) → high retention probability (~1)
- Probabilities are scaled so the maximum equals `subsample_prob`

In your config file:
```yaml
data:
  # ... other settings ...
  subsample_prob: 0.3  # Keep up to 30% of samples, preferring outliers
```

## Classical ML Integration

This describes the integration for classical machine learning models (XGBoost, RandomForest, AdaBoost, LightGBM, etc.) with the Lightning/Hydra framework.

The ML integration is **completely separated** from the deep learning training pipeline but shares the same:
- Data loading infrastructure (`WWMDataModule`)
- Configuration system (Hydra)
- Logging and metrics tracking
- Sample weighting strategy

### Train XGBoost model (default):
```bash
python src/train_ml.py
```

### Train with different models:
```bash
# Random Forest
python src/train_ml.py experiment=ml/random_forest

# etc for other models
```
