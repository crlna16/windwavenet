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
  sample_weight_variable: sigWaveHeight  # Variable whose z-score drives the weights
```

The weighting variable can be any variable present in the dataset, or a
`delta_<var>` time difference (e.g. `delta_sigWaveHeight`), which is derived on
the fly if not already present. Weights are applied to both the training and
validation losses, so `val/loss` — the metric monitored for checkpointing and
early stopping — reflects the weighted objective. RMSE metrics remain unweighted.

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

## Graph backbones: mesh edge encoding

The GAT backbones (`MultiHeadGAT`, `MultiHeadGATVN`, `MultiHeadGATLSTMVN`)
attend over the SCHISM mesh edges. Attention is strictly **one-hop**: in each
layer a node attends only to its direct mesh neighbours (and itself), so the
spatial receptive field is `num_layers` hops. Context beyond that comes from
stacking layers and, in the VN variants, from a virtual node that exchanges
information with all nodes of the sampled subgraph in a single step.

**Why encode edge lengths?** The mesh is unstructured: edge lengths vary by
orders of magnitude between refined coastal areas and the coarse open sea.
Without extra information, a hop across a 100 m coastal edge and a hop across
a multi-km offshore edge would look identical to the network. Each edge is
therefore assigned an inverse-distance weight

```
w = 1 / (1 + length / edge_weight_scale)
```

which is expanded through `n_rbf` Gaussian radial basis functions and passed
as edge features to `GATv2Conv`, where they become per-head attention biases.
This tells attention how much physical space each hop covers, so neighbours
can be weighted by physical proximity rather than hop count alone.

**Choosing `edge_weight_scale`** (data config): the value should be comparable
to the typical (median) mesh edge length. An edge of exactly that length gets
weight 0.5, and the weight distribution spreads over (0, 1) where the RBF
centres live. If the scale is much larger than the mesh edges, all weights
cluster near 1.0, most RBF features return near-zero constants, and the
encoding carries no signal.

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
