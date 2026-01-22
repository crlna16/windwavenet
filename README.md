<!--
SPDX-FileCopyrightText: 2025 Hereon
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


# Sample Weights Implementation

## Overview

Sample weights have been moved from the model to the dataset. Weights are now **pre-computed once before training** and normalized so that their sum equals the number of samples (mean = 1.0). This provides:

1. **Consistency**: All batches use the same global weight distribution
2. **Stability**: Loss magnitude remains consistent throughout training
3. **Interpretability**: Weights are normalized with a clear meaning
4. **Efficiency**: No per-batch computation overhead during training

## Architecture

### Dataset (`SimpleWWMDataset`)
- Stores sample weights as a numpy array
- Provides `compute_weights()` method to calculate and normalize weights
- Returns weights in `__getitem__()` along with features and targets

### Data Module (`WWMDataModule`)
- Accepts `sample_weight_strategy` and `sample_weight_power` parameters
- Provides `compute_sample_weights()` method to trigger weight computation for all datasets

### Model (`WWMEmulator`)
- Accepts precomputed weights from batch data
- No longer computes weights dynamically
- Simply applies the weights during loss calculation

### Callback (`ComputeSampleWeightsCallback`)
- Runs once at the beginning of training (`on_fit_start`)
- Calls `datamodule.compute_sample_weights()` with the model
- Ensures weights are computed before any training steps

## Configuration

### In `configs/train.yaml`:

```yaml
data:
  # Sample weighting parameters (moved from model to data)
  sample_weight_strategy: null  # Options: null, 'abs_value'
  sample_weight_power: 1.0      # Exponent for weight calculation

callbacks:
  compute_sample_weights:
    _target_: src.utils.callbacks.ComputeSampleWeightsCallback
    compute_on_device: null  # null = use model's device
```

## Weight Calculation

For `sample_weight_strategy = "abs_value"`:

```python
# Per-sample weight
weight = (1 + |target|)^power

# Normalization (per target variable)
normalized_weight = weight * (n_samples / sum(weights))
```

This ensures:
- Larger absolute target values get higher weights
- Sum of weights = number of samples
- Mean weight = 1.0
- Loss scale remains consistent

### Examples with different powers:

| Target Value | Power=1.0 | Power=2.0 | Power=4.0 |
|--------------|-----------|-----------|-----------|
| 0.0          | 1.0       | 1.0       | 1.0       |
| 0.1          | 1.1       | 1.21      | 1.46      |
| 0.5          | 1.5       | 2.25      | 5.06      |
| 1.0          | 2.0       | 4.0       | 16.0      |

Higher power values create more aggressive weighting toward large values.

## Usage

### Enable sample weighting:

```yaml
data:
  sample_weight_strategy: "abs_value"
  sample_weight_power: 2.0
```

### Disable sample weighting:

```yaml
data:
  sample_weight_strategy: null
```

## Implementation Details

1. **Weight Storage**: Weights are stored as numpy arrays in the dataset (`self.weights`)

2. **Batch Data**:
   - Without weights: `(features, targets)`
   - With weights: `(features, targets, weights)`

3. **Normalization**: Performed per target variable to ensure each target's weights sum to `n_samples`

4. **Device Handling**: Weight computation can be done on any device (CPU/GPU) specified in the callback

5. **Compatibility**: If no weights are configured, the model automatically uses uniform weights (all 1.0)

## Migration Notes

### Before (per-batch normalization):
- Weights computed dynamically in `model.model_step()`
- Normalized per batch
- Variable loss magnitude between batches

### After (global normalization):
- Weights computed once before training
- Normalized globally across entire dataset
- Consistent loss magnitude throughout training

### Breaking Changes:
- `sample_weight_strategy`, `sample_weight_power`, `normalize_sample_weights` moved from model config to data config
- `normalize_sample_weights` removed (always True now - weights always sum to n_samples)
- New callback `ComputeSampleWeightsCallback` must be included in config

## Files Modified

1. `src/data/SimpleWWMDataset.py`: Added weight storage and computation
2. `src/data/WWMDataModule.py`: Added weight computation orchestration
3. `src/model/WWMEmulator.py`: Removed dynamic weight computation, accepts precomputed weights
4. `src/utils/callbacks.py`: New callback for triggering weight computation
5. `configs/train.yaml`: Moved parameters from model to data, added callback

# Classical ML Integration

This directory contains the integration for classical machine learning models (XGBoost, RandomForest, AdaBoost, LightGBM, etc.) with the Lightning/Hydra framework.

## Overview

The ML integration is **completely separated** from the deep learning training pipeline but shares the same:
- Data loading infrastructure (`WWMDataModule`)
- Configuration system (Hydra)
- Logging and metrics tracking
- Sample weighting strategy

## Quick Start

### Train XGBoost model (default):
```bash
python src/train_ml.py
```

### Train with different models:
```bash
# Random Forest
python src/train_ml.py experiment=ml/random_forest

# Gradient Boosting
python src/train_ml.py experiment=ml/gradient_boosting

# AdaBoost
python src/train_ml.py experiment=ml/adaboost

# LightGBM (faster than XGBoost)
python src/train_ml.py experiment=ml/lightgbm

# Optimized XGBoost
python src/train_ml.py experiment=ml/xgboost_optimized
```

### With sample weighting:
```bash
python src/train_ml.py \
  data.sample_weight_strategy=abs_value \
  data.sample_weight_power=2.0
```

### Override hyperparameters:
```bash
python src/train_ml.py \
  experiment=ml/xgboost_optimized \
  model.model.n_estimators=5000 \
  model.model.learning_rate=0.01 \
  model.model.max_depth=12
```

## Architecture

### Files Structure:
```
src/
├── train_ml.py              # ML training script (separate from train.py)
└── model/
    └── MLWrapper.py         # Wrapper class for scikit-learn compatible models

configs/
├── train_ml.yaml            # Default ML training config
└── experiment/
    └── ml/                  # ML-specific experiment configs
        ├── xgboost_optimized.yaml
        ├── random_forest.yaml
        ├── gradient_boosting.yaml
        ├── adaboost.yaml
        └── lightgbm.yaml
```

### Key Components:

#### 1. `MLWrapper` Class
- Wraps any scikit-learn compatible model
- Provides consistent interface: `fit()`, `predict()`, `evaluate()`
- Computes same metrics as Lightning models (RMSE, binned RMSE)
- Supports sample weights
- Handles early stopping for models that support it

#### 2. `train_ml.py` Script
- Extracts numpy arrays from `WWMDataModule`
- Trains classical ML models
- Evaluates on validation and test sets
- Saves trained models with joblib

#### 3. Configuration System
- Uses same Hydra configs as DL training
- Shares data configuration
- Separate model configs for each ML algorithm

## Model Support

### Supported Models:

| Model | Library | Early Stopping | Sample Weights | Speed |
|-------|---------|----------------|----------------|-------|
| **XGBoost** | xgboost | ✅ | ✅ | Fast |
| **LightGBM** | lightgbm | ✅ | ✅ | Very Fast |
| **Random Forest** | scikit-learn | ❌ | ✅ | Medium |
| **Gradient Boosting** | scikit-learn | ❌ | ✅ | Slow |
| **AdaBoost** | scikit-learn | ❌ | ✅ | Medium |

### Adding New Models:

Any scikit-learn compatible model can be added. Just create a new experiment config:

```yaml
# configs/experiment/ml/my_model.yaml
# @package _global_

experiment_name: "my_model"

model:
  _target_: src.model.MLWrapper.MLModelWrapper
  target_names: ${data.targets}

  model:
    _target_: sklearn.linear_model.Ridge  # or any sklearn model
    alpha: 1.0
    random_state: ${seed}
```

## Output

Trained models are saved to:
```
logs/train_ml/<experiment_name>/runs/<timestamp>/model.joblib
```

Logs and metrics are saved to the same directory.

## Comparison with Deep Learning

### Advantages of Classical ML:
- ✅ Faster training (no epochs, no backprop)
- ✅ Less hyperparameter tuning needed
- ✅ No GPU required
- ✅ Better interpretability (feature importance)
- ✅ Works well on tabular data

### Advantages of Deep Learning:
- ✅ Better at learning complex patterns
- ✅ Scales better with more data
- ✅ Can leverage GPU acceleration
- ✅ More flexible architecture

## Performance Tips

1. **XGBoost/LightGBM are usually the best performers** for tabular data
2. **Use sample weighting** for imbalanced regression tasks
3. **Enable early stopping** to avoid overfitting (XGBoost/LightGBM)
4. **Tune hyperparameters** with Hydra sweeps (see example below)

## Hyperparameter Sweeps

Create a sweep config for grid/random search:

```yaml
# configs/sweeps/xgboost_sweep.yaml
defaults:
  - override /hydra/sweeper: optuna

experiment_name: "xgboost_sweep"

hydra:
  sweeper:
    n_trials: 50
    direction: minimize

    params:
      model.model.max_depth: range(5, 15)
      model.model.learning_rate: interval(0.01, 0.3)
      model.model.n_estimators: range(500, 3000, step=500)
      model.model.subsample: interval(0.6, 1.0)
      model.model.colsample_bytree: interval(0.6, 1.0)
```

Run sweep:
```bash
python src/train_ml.py -m hydra/sweeper=optuna \
  experiment=ml/xgboost_optimized \
  --config-name=train_ml
```

## Dependencies

Install required packages:
```bash
pip install xgboost lightgbm scikit-learn joblib
```

Optional for hyperparameter optimization:
```bash
pip install optuna
```

# Z-Score Based Subsampling Implementation

## Overview
Added the ability to subsample the training dataset based on z_score-derived probabilities. This feature allows preferential retention of outliers (samples with high z-scores) in the training data.

## Changes Made

### 1. SimpleWWMDataset.py
- **Added parameter**: `subsample_prob: Optional[float] = None`
  - When set (e.g., 0.3), enables subsampling based on z_score
  - Only applied during dataset initialization

- **Added method**: `_subsample_by_zscore()`
  - Computes z-scores for all targets
  - Calculates retention probability: `p = 1 - exp(-z^2 / 2)`
  - Scales probabilities so max equals `subsample_prob`
  - Applies stochastic sampling to keep/discard samples
  - Logs subsampling statistics

- **Updated type hints**: Changed `sample_weight_power` to `Optional[float]` to allow `None` values

### 2. WWMDataModule.py
- **Added parameter**: `subsample_prob: Optional[float] = None`
- **Training dataset only**: Parameter is passed only to `ds_train`
- **Validation/test datasets**: Always use `subsample_prob=None` (no subsampling)
- **Updated type hints**: Changed `sample_weight_power` to `Optional[float]`

### 3. configs/train.yaml
- **Added configuration option**: `subsample_prob: null`
- **Documentation**: Added comments explaining usage
- **Example values**: Suggested range 0-1 (e.g., 0.3 for 30% max retention)

## How It Works

### Retention Probability Formula
```
p(sample) = 1 - exp(-z^2 / 2)
```

Where `z` is the maximum absolute z-score across all target variables for that sample.

### Properties
- Samples near the mean (z ≈ 0) → low retention probability (~0)
- Outliers (high |z|) → high retention probability (~1)
- Probabilities are scaled so the maximum equals `subsample_prob`

### Example Usage

In your config file:
```yaml
data:
  # ... other settings ...
  subsample_prob: 0.3  # Keep up to 30% of samples, preferring outliers
```

Or from command line:
```bash
python src/train.py data.subsample_prob=0.3
```

## Testing

A test script is provided at `test_subsampling.py` that demonstrates:
1. Creating synthetic data with outliers
2. Computing retention probabilities
3. Applying subsampling
4. Verifying outlier preference

Run it with:
```bash
python test_subsampling.py
```

## Implementation Details

### When Subsampling Occurs
- **During**: Dataset initialization (`__init__`)
- **After**: Data loading and scaling
- **Before**: Weight computation

### Memory Efficiency
- Data is loaded into memory first
- Subsampling filters the tensors (`self.X` and `self.y`)
- Reduced memory footprint after subsampling

### Logging
The implementation logs:
- Number of samples before/after subsampling
- Retention rate percentage
- Retention probability statistics (mean, std, min, max)

## Notes

- **Training only**: Subsampling is never applied to validation or test sets
- **Stochastic**: Results vary between runs unless seed is fixed
- **Configurable**: Easy to enable/disable via config file
- **Non-intrusive**: Setting `subsample_prob: null` disables the feature entirely
