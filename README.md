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
