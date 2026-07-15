<!--
SPDX-FileCopyrightText: 2025-2026 Hereon
SPDX-FileContributor: Caroline Arnold

SPDX-License-Identifier: MIT
-->

# Wind Wave Model Emulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)

## Overview

A machine-learning emulator for the WWM (Wind Wave Model) component of a
SCHISM ocean simulation. Given atmospheric/ocean forcing and the previous wave
state on an unstructured SCHISM mesh, the emulator predicts the next wave state
(significant wave height, mean wave period, mean wave direction, ...).

The project is built on **PyTorch Lightning** + **Hydra** (config-driven) and
supports several model families that share one data/preprocessing pipeline:

| Family | Backbone (`src/model/backbones/`) | Lightning module | Data module |
| --- | --- | --- | --- |
| Pointwise MLP | `MLP` | `WWMEmulator` | `WWMDataModule` |
| Temporal LSTM | `LSTMNet` | `WWMEmulator` | `SequenceWWMDataModule` |
| Graph attention | `MultiHeadGAT` | `GraphWWMEmulator` | `GraphWWMDataModule` |
| Graph + virtual node | `MultiHeadGATVN` | `GraphWWMEmulator` | `GraphWWMDataModule` |
| Graph + temporal virtual node | `MultiHeadGATLSTMVN` | `GraphWWMEmulator` | `CenterHistoryGraphWWMDataModule` |
| Classical ML | XGBoost / scikit-learn estimator | `MLModelWrapper` | `WWMDataModule` |

Key capabilities:

- **Multi-target prediction** with per-variable loss weights and per-variable
  output constraints: `semipositive` (ReLU), `positive` (softplus),
  `angle_deg` (mod 360), `angle_rad` (mod 2π), `sigmoid`, and the paired
  `unit_circle` constraint for `sin_*`/`cos_*` direction pairs.
- **Circular error metrics** for angular targets and **binned RMSE** over
  configurable value ranges (`target_bins`).
- **Sample weighting** for imbalanced regression (`z_score`, `density`,
  `histogram`), pre-computed once per training run (see below).
- **Graph models over the SCHISM mesh** with physical edge-length encoding
  via inverse-distance weights expanded through Gaussian RBFs (see below).
- **Evaluation modes**: teacher-forced one-step testing, full-domain
  prediction, and pseudo-coupled autoregressive rollout.
- **Reproducible experiments** via Hydra `experiment=` configs under
  `configs/experiment/` and inference-timing benchmarks.

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

## Usage

All entrypoints live in `src/` and are configured with Hydra. Select a model /
data combination with an `experiment=` config from `configs/experiment/` and
override any field on the command line (e.g. `trainer.max_epochs=50`).

| Script | Base config | Purpose |
| --- | --- | --- |
| `src/train.py` | `configs/train.yaml` | Train a deep-learning model (MLP / LSTM / GAT). |
| `src/train_ml.py` | `configs/train_ml.yaml` | Train a classical ML model (XGBoost / scikit-learn). |
| `src/eval.py` | `configs/eval.yaml` | Teacher-forced one-step testing of a checkpoint. |
| `src/predict.py` | `configs/predict.yaml` | Full-domain prediction for pointwise / sequence models. |
| `src/predict_gcn.py` | `configs/predict_gcn.yaml` | Full-domain prediction for graph models (greedy mesh partitioning). |
| `src/rollout.py` | `configs/rollout.yaml` | Pseudo-coupled autoregressive rollout evaluation. |
| `src/benchmark_inference.py` | — (argparse) | Benchmark inference latency/memory on CPU/CUDA. |

```bash
# Train
python src/train.py experiment=swh_1h_mlp
python src/train.py experiment=swh_1h_lstm
python src/train.py experiment=swh_1h_multi-head-gat-vn
python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn

# Evaluate / predict a checkpoint
python src/eval.py experiment=swh_1h_mlp ckpt_path=<ckpt>
python src/predict.py experiment=swh_1h_lstm ckpt_path=<ckpt>
python src/predict_gcn.py experiment=swh_1h_multi-head-gat-vn ckpt_path=<ckpt>
```

The available experiment configs (see `configs/experiment/`) cover single-target
`swh_1h_*` and multi-target `swh_mwp_mwd_1h_*` variants for each backbone.

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
  sample_weight_strategy: null  # Options: null, 'z_score', 'density', 'histogram'
  sample_weight_power: 1.0      # Exponent for weight calculation
  sample_weight_variable: sigWaveHeight  # Variable whose z-score drives the weights
```

The weighting variable can be any variable present in the dataset, or a
`delta_<var>` time difference (e.g. `delta_sigWaveHeight`), which is derived on
the fly if not already present. Weights are applied to both the training and
validation losses, so `val/loss` — the metric monitored for checkpointing and
early stopping — reflects the weighted objective. RMSE metrics remain unweighted.

## Rollout evaluation (pseudo-coupled)

`src/rollout.py` evaluates a trained checkpoint in the configuration it will
face when coupled back to SCHISM: forcing features (wind, elevation,
currents, ...) come from ground truth at every step — mimicking the external
update by the host model — while every `prev_<target>` feature is fed back
from the emulator's **own** prediction at the previous step. This measures
how errors accumulate in the wave channel over lead time, which one-step
(teacher-forced) test metrics cannot show.

```bash
python src/rollout.py experiment=swh_1h_mlp ckpt_path=<ckpt>
python src/rollout.py experiment=swh_1h_lstm ckpt_path=<ckpt>
python src/rollout.py experiment=swh_1h_multi-head-gat-vn ckpt_path=<ckpt>
python src/rollout.py experiment=swh_1h_multi-head-gat-lstm-vn ckpt_path=<ckpt>

# limit the horizon to 48 steps
python src/rollout.py experiment=swh_1h_lstm ckpt_path=<ckpt> rollout.steps=48
```

The model family (MLP / LSTM / graph) is detected from `data._target_`; pass
the same experiment config the checkpoint was trained with. Graph models run
over the full mesh using the same greedy domain partitioning as
`predict_gcn.py`. Delta targets are not supported.

Outputs in the Hydra run directory:

- `rollout_metrics_lead.csv` — per-lead-time RMSE / MAE / bias per target,
  a persistence baseline (initial state held constant), and per-bin RMSE
  for configured `target_bins`. Plot RMSE vs. lead to see drift.
- `rollout_predictions.nc` — full-domain predictions per rollout step
  (disable with `rollout.save_predictions=false` if too large).

If `data.boundary_nodes_csv` is set, boundary nodes keep ground-truth
`prev_*` values (mimicking prescribed boundary conditions in the coupled
setup) and are excluded from metrics.

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

### Train with a different estimator:
The estimator is configured under `model.model` in `configs/train_ml.yaml`
(default: `xgboost.XGBRegressor`). `MLModelWrapper` accepts any
scikit-learn-compatible estimator exposing `fit`/`predict`, so you can swap it
via a Hydra override (dropping any XGBoost-specific keyword args):

```bash
python src/train_ml.py \
  model.model._target_=sklearn.ensemble.RandomForestRegressor \
  ~model.callbacks
```
