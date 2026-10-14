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

Key capabilities:

- **Multi-target prediction** with per-variable loss weights and per-variable
  output constraints: `semipositive` (ReLU), `seminegative` (−ReLU(−x)),
  `positive` (softplus), `negative` (−softplus(−x)),
  `angle_deg` (mod 360), `angle_rad` (mod 2π), `sigmoid`, and the paired
  `unit_circle` constraint for `sin_*`/`cos_*` direction pairs.
- **Graph models over the SCHISM mesh** with physical edge-length encoding
  via inverse-distance edge weights.
- **Sample weighting** for imbalanced regression (`histogram`)
- **Evaluation modes**: teacher-forced one-step testing, full-domain
  prediction, and pseudo-coupled autoregressive rollout.
- **Reproducible experiments** via Hydra `experiment=` configs under
  `configs/experiment/`.

## Installation

Clone the repository. This project uses [uv](https://docs.astral.sh/uv/) for package management. Install `uv` locally and create the virtual environment with

```bash
uv sync
```

To install the package

```bash
pip install -e .
```

## Trained models

Availabel with descriptive model cards in the [Model Registry](https://codebase.helmholtz.cloud/hereon-ksn/wind-wave-model-emulator/-/ml/models).

## Logs

The project is tracked online at [https://wandb.ai/wwm-emulator/](https://wandb.ai/wwm-emulator/).

## Usage

All entrypoints live in `src/` and are configured with Hydra. Select a model /
data combination with an `experiment=` config from `configs/experiment/` and
override any field on the command line (e.g. `trainer.max_epochs=50`).

| Script | Base config | Purpose |
| --- | --- | --- |
| `src/train.py` | `configs/train.yaml` | Train a deep-learning model (MLP / LSTM / GAT). |
| `src/eval.py` | `configs/eval.yaml` | Teacher-forced one-step testing of a checkpoint. |
| `src/predict.py` | `configs/predict.yaml` | Full-domain prediction for pointwise / sequence models. |
| `src/predict_gcn.py` | `configs/predict_gcn.yaml` | Full-domain prediction for graph models (disjoint mesh partitioning). |
| `src/rollout.py` | `configs/rollout.yaml` | Pseudo-coupled autoregressive rollout evaluation. |

```bash
# Train
python src/train.py experiment=swh_1h_mlp
python src/train.py experiment=swh_1h_lstm
python src/train.py experiment=swh_1h_gat-vn
python src/train.py experiment=swh_1h_gat-lstm-vn

# Evaluate / predict a checkpoint
python src/eval.py experiment=swh_1h_mlp ckpt_path=<ckpt>
python src/predict.py experiment=swh_1h_lstm ckpt_path=<ckpt>
python src/predict_gcn.py experiment=swh_1h_gat-vn ckpt_path=<ckpt>
python src/predict_gcn.py experiment=swh_1h_gat-lstm-vn ckpt_path=<ckpt>

# Rollout mode
python src/rollout.py experiment=swh_1h_gat-lstm-vn ckpt_path=<ckpt>
```

The available experiment configs (see `configs/experiment/`) cover single-target
`swh_1h_*` and multi-target `multi_1h_*` variants for each backbone.

## Export to torchscript

Use `src/export_torchscript.py` and `scripts/create_torchscript.sh` to create torchscript models from the trained model checkpoints.

# Technical notes

## API Documentation

Automatically generated at https://wave-wind-model-emulator-267352.pages.hzdr.de/src.html whenever `main` is updated in the Helmholtz Gitlab (see `.gitlab-ci.yaml`). Based on docstrings.

## Preprocessing

We create zarr archives from the netcdf output. Follow the notebook `CreateZarr.ipynb`.


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
python src/rollout.py experiment=swh_1h_gat-vn ckpt_path=<ckpt>
python src/rollout.py experiment=swh_1h_gat-lstm-vn ckpt_path=<ckpt>

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

## Noise injection on `prev_*` inputs

At rollout time the `prev_<target>` inputs are the model's own predictions and
carry error; during training they are ground truth. The model therefore learns
to trust them more than it should, and one bad step can throw the trajectory
off. `model.prev_noise_std` perturbs those channels during training so the model
learns to recover the clean target from a corrupted previous state:

```yaml
model:
  prev_noise_std:
    sigWaveHeight: 0.08   # metres — physical units, NOT z-scored
```

Sigma is given in physical units and divided by the feature's std from
`data.scaling_config` internally, so `data.scaling_config` is required. Set it
from the model's measured one-step validation RMSE: that makes the corrupted
input distribution resemble what the model actually meets at lead 1.

This is a cheap (~1x-cost) way to make the model robust to compounding rollout
error, without the T x cost of rolling out multiple steps during training and
feeding back the model's own (detached) predictions. That autoregressive
training approach was removed from this repository for clarity; see
[AutoregressiveTraining.md](AutoregressiveTraining.md) if you need to
reinstate it (Sanchez-Gonzalez et al., ICML 2020).

**Applied to the training batch only.** Validation, test and predict batches stay
clean, so `val/loss` and the RMSE metrics remain comparable with runs trained
without noise. Expect `val/loss` to get slightly *worse* — that is the price
being paid for rollout robustness, and it means early stopping on `val/loss`
will fight the run. Raise the patience (or disable early stopping) for these
runs and select on rollout skill instead.

## Sample Weights Implementation

Weights are **pre-computed once before training** and normalized so that their sum equals the number of samples (mean = 1.0). In `configs/train.yaml`:

```yaml
data:
  # Sample weighting parameters
  sample_weight_strategy: null  # Options: null, 'z_score' (legacy), 'histogram' (preferred)
  sample_weight_power: 1.0      # Exponent for weight calculation
  sample_weight_variable: sigWaveHeight  # Variable whose distribution drives the weight
```

The weighting variable can be any variable present in the dataset, or a
`delta_<var>` time difference (e.g. `delta_sigWaveHeight`), which is derived on
the fly if not already present. Weights are applied to both the training and
validation losses, so `val/loss` — the metric monitored for checkpointing and
early stopping — reflects the weighted objective. RMSE metrics remain unweighted.
