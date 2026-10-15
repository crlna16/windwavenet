<!--
SPDX-FileCopyrightText: 2025-2026 Hereon
SPDX-FileContributor: Caroline Arnold

SPDX-License-Identifier: MIT
-->

# Re-training WindWaveNet

This guide explains how to re-train WindWaveNet when the training data change:
a new SCHISM/WWM-III run, a new grid, or additional wave variables. It assumes
you know SCHISM and WWM-III well; the machine-learning steps are explained as we
go. The LSTM experiment `configs/experiment/wwnet2_lstm.yaml` is used as the
running example.

## Overview

| Step | What you do | Needed for a new run | Needed for a new grid | Needed for new variables |
|---|---|---|---|---|
| 1 | Convert the SCHISM output to a Zarr archive | yes | yes | yes |
| 2 | Check the training, validation and test periods | yes | yes | – |
| 3 | Compute the scaling file (mean and std per variable) | yes | yes | yes |
| 4 | Select training nodes and boundary nodes | – | yes | – |
| 5 | Write the experiment config | yes | yes | yes |
| 6 | Train | yes | yes | yes |
| 7 | Evaluate in rollout mode and tune the noise | yes | yes | yes |
| 8 | Export to TorchScript for SCHISM | yes | yes | yes |

Steps 1, 3 and 4 are deliberately done in notebooks rather than scripts: each
of them needs a look at the data (time coverage, distributions, maps of the
selected nodes) before the result is used.

## Glossary

- **Features / inputs**: what the network receives at each node and hour:
  wind, water level, depth-averaged currents, depth, and the wave state of the
  previous hour (`prev_<variable>`).
- **Targets / outputs**: the wave variables the network predicts for the
  current hour.
- **Training / validation / test periods**: the network learns from the
  training period. The validation period is used during training to decide
  which version of the network to keep (and when to stop). The test period is
  only used at the very end, to report results on data the network has never
  influenced.
- **Epoch**: one pass over the whole training data. Training runs for many
  epochs; after each one the network is evaluated on the validation period.
- **Checkpoint**: the saved network weights (`.ckpt` file). Training keeps the
  checkpoint with the lowest validation loss.
- **Scaling**: the network works best when all inputs are of order 1. Each input
  is therefore standardised as `(x - mean) / std`, with mean and std taken from
  the training period.
- **One-step (teacher-forced) vs. rollout**: in training, `prev_*` inputs are
  always the true SCHISM/WWM values. Coupled to SCHISM, they are the network's
  own previous prediction, so errors can accumulate. A *rollout* mimics the
  coupled case offline and is the evaluation that matters.
- **Noise injection**: during training, random noise is added to the `prev_*`
  inputs so that the network learns not to trust them blindly. This makes
  rollouts much more stable.
- **Loss / loss weights**: the quantity training minimises, here the mean
  squared error between prediction and SCHISM/WWM, summed over the target
  variables. The loss weights decide how much each variable counts in that sum.
- **Output constraint**: a function applied to the network's raw output that
  keeps a prediction physical, e.g. always positive for wave height.
- **Configs**: all settings live in YAML files under `configs/` (managed with
  [Hydra](https://hydra.cc)). An *experiment config* in `configs/experiment/`
  collects everything for one model. Any setting can also be overridden on the
  command line, e.g. `trainer.max_epochs=50`.
- **wandb**: the web dashboard where training curves are logged
  (https://wandb.ai/wwm-emulator/).
- **TorchScript**: a self-contained file format for a trained network that can
  be loaded from Fortran via FTorch, without Python.

## Setup

```bash
uv sync           # creates the Python environment
wandb login       # training curves are logged to https://wandb.ai/wwm-emulator/
```

Training needs a GPU node; see `scripts/submit_train.sh` for a Slurm template.

## 1. Convert the SCHISM output to Zarr

Notebook: `notebooks/CreateZarrMultiTarget.ipynb`.

The training code reads one Zarr archive (a chunked, cloud-friendly
NetCDF-like store) instead of the daily `out2d_*.nc` files. In the notebook:

1. Point `files` to the `out2d_<day>.nc` files of the new run.
2. Check the `variables` list: it must contain every input and target variable
   plus the mesh variables (`SCHISM_hgrid_*`, `depth`, `dryFlagNode`). Names are
   the SCHISM output names (`sigWaveHeight`, `dissRateDepBreaking`, ...); see
   `WINDWAVENET_WWM_OUTPUTS.md` in the SCHISM repository for what each one is.
3. Adjust the chunk size in `encoding` to the number of mesh nodes of the grid
   (`(24, n_nodes)` = one day of hourly output per chunk).
4. Set the output path and run the `to_zarr` cell (it is a raw cell to prevent
   accidental overwrites; switch it to code to run it).

Before moving on, check:

- **Time step**: the output must be hourly and without gaps you did not
  intend. The code derives "samples per day" from the first two time steps.
- **Calendar**: periods are given as day-of-year of the store's first year
  (step 2), so note which year and days the store covers.
- **Constant variables**: a variable that is constant everywhere cannot be
  learned. This is the case for `rollerDissRate` in the current runs: the
  roller model is off in WWM-III, so it is 0 everywhere and its std is 0 (see
  step 3). Leave such variables out as input and target.
- **Dry nodes**: WWM sets all wave outputs to 0 at dry nodes; this is expected.

## 2. Check the training, validation and test periods

Periods are lists of `[first_day, last_day]` (day of year, both inclusive).
Several ranges can be combined. The current setup is:

```yaml
train_days: [[1, 152], [214, 305]]   # Jan-May, Aug-Oct
val_days:   [[153, 182], [306, 335]] # Jun, Nov
test_days:  [[183, 213], [336, 366]] # Jul, Dec
```

**As long as the training data cover 2020, keep these splits unchanged**, so
that new models remain comparable with the existing ones. If a run starts
later in the year (the SNS run starts on day 28), the missing days are simply
not used.

**If you have more data** (other years), reconsider the splits:

- All three periods should contain storms. The validation period decides which
  checkpoint is kept, so a calm validation period selects a model that is good
  in calm conditions and weak in storms.
- Keep the periods separate; the network must never see the test period during
  training or tuning. A whole separate year is the cleanest test period.
- Days are counted from 1 January of the store's first year and simply
  continue into the next year: day 366 is 1 January of the following year, or
  day 367 if the first year is a leap year such as 2020.

## 3. Compute the scaling file

Notebook: `notebooks/CreateZarrMultiTarget.ipynb`, section *Create scaling file*.

1. Set `trainds` to the **training period only** (the same ranges as
   `train_days`).
2. Run the cell that prints `mean` and `std` per variable. Variables that are
   targets are printed with the `prev_` prefix, because they enter the network
   as `prev_<variable>` inputs.
3. Copy the output into a new file in `src/data/scaling/`, e.g.
   `scale_<year>_<run>.yaml`.

Notes:

- Directions are not in the scaling file. They are given to the network as
  `sin_<direction>` and `cos_<direction>`, which are already of order 1. These
  are computed on the fly from the direction in degrees; nothing to do here.
- **Check that no std is 0.** The inputs are divided by their std, so a std of
  0 turns the input into NaN and training fails immediately. The notebook
  prints `std: 0.0000` for `rollerDissRate` in the current runs (step 1). The
  current scaling file carries a placeholder `std: 1.0` for
  `prev_rollerDissRate`, but the variable is left out of the `wwnet2_*`
  experiments altogether, which is the better option.
- The same statistics are used for the loss weights (step 5).

## 4. Select training nodes and boundary nodes (new grid only)

Notebook: `notebooks/SelectStationsMultiTarget.ipynb`.

**Training nodes.** Training on all mesh nodes is too expensive and dominated
by the densely resolved coast. The notebook puts a regular 10 km × 10 km grid
over the domain and keeps, in each cell, the mesh node closest to the cell
centre. Adjust the bin edges (`xbins`, `ybins`) to the new domain and write the
node indices to `src/data/node_selection/node_idx_closest_center_<grid>.csv`.
Testing and prediction always use all nodes.

**Boundary nodes.** Nodes at the open boundary can carry unphysical values
(prescribed or poorly resolved wave conditions). They are excluded from
training and from all evaluation metrics, and in rollouts they are held at the
true values (as the boundary condition would be in a coupled run). Identify
them, check them visually on a map of e.g. `sigWaveHeight`, and write their
indices to `src/data/node_selection/boundary_nodes_<grid>.csv` (one column
`nSCHISM_hgrid_node`).

> TODO (Caroline): describe how the boundary nodes of the current grid were
> identified, or add the code to the notebook.

**Graph models only.** The graph models weight the connection between two
neighbouring nodes by their distance, `1 / (1 + length / edge_weight_scale)`.
For a new mesh, set `data.edge_weight_scale` to its median edge length in
metres, and check `data.max_nodes` (the size of the sub-graphs the model is
trained on).

## 5. Write the experiment config

Copy an existing config, e.g.

```bash
cp configs/experiment/wwnet2_lstm.yaml configs/experiment/<new_name>.yaml
```

and go through it:

| Field | What to set |
|---|---|
| `experiment_name` | the new name (it also names the output folder) |
| `data.data_dir` | path to the new Zarr archive |
| `data.train_days`, `val_days`, `test_days` | periods from step 2 |
| `data.scaling_config` | scaling file from step 3 |
| `data.node_idx_csv` | training nodes from step 4 |
| `data.boundary_nodes_csv` | boundary nodes from step 4 |
| `data.features`, `data.targets` | see "Adding a new variable" below |
| `model.backbone.input_dim` | must equal the number of `features` |
| `model.backbone.output_dim` | must equal the number of `targets` |
| `model.output_constraints` | one entry per target, see below |
| `model.loss_weights` | one entry per target, see below |
| `model.prev_noise_std` | noise per target, see step 7 |
| `data.edge_weight_scale` | graph models only: median edge length of the mesh (step 4) |
| `logger.wandb.project`, `tags` | where the training curves appear |

> **Three settings are not derived automatically and must be updated by hand
> whenever the variables or the data change:**
>
> - `model.backbone.input_dim` = number of entries in `data.features`, and
>   `model.backbone.output_dim` = number of entries in `data.targets`. A
>   mismatch stops the run at the first batch with a shape error.
> - `model.loss_weights`: recompute from the new scaling file (below). Stale
>   weights do not cause an error; the training just balances the variables
>   badly.
> - `data.edge_weight_scale` (graph models): recompute for a new mesh. A wrong
>   value does not cause an error either.

### Adding a new variable

For every new target variable `X`:

1. Add `X` to `data.targets` and `prev_X` to `data.features`. For a direction,
   add both `sin_X` and `cos_X` (and `prev_sin_X`, `prev_cos_X`) instead of `X`.
2. Make sure `prev_X` has an entry in the scaling file (step 3).
3. Add an **output constraint** that keeps the prediction physical:
   - `positive`: always > 0 (wave heights, periods, lengths, velocities)
   - `negative`: always < 0 (e.g. `dissRateDepBreaking`)
   - `unit_circle`: for `sin_X`/`cos_X` pairs, keeps sin² + cos² = 1
   - `semipositive`/`seminegative` (ReLU-based) also exist, but can get stuck
     predicting exactly 0 during training; prefer `positive`/`negative`.
4. Add a **loss weight**. The training minimises the sum of the per-variable
   mean squared errors, so variables with large values (e.g. wavelength in
   metres) would otherwise dominate. Use `1 / std²` from the scaling file
   (`1 / 0.5` for sin/cos), divided by the mean of all weights so that they
   average to 1.
5. Increase `input_dim` and `output_dim` accordingly.

Two remarks:

- The per-variable losses shown in wandb (`val/loss_<variable>`) are *before*
  weighting, in the variable's own units squared. To compare variables, divide
  them by `std²`.
- More targets make the task harder for a network of fixed size. If the
  existing variables get noticeably worse after adding new ones, the network
  may need to be larger (`hidden_dim`).

## 6. Train

```bash
uv run python src/train.py experiment=<new_name>
```

Output goes to `logs/train/<experiment_name>/runs/<date>/`; the checkpoints are
in its `checkpoints/` subfolder. Training stops by itself when the validation
loss has not improved for a while (early stopping).

Watch in wandb:

- `train/loss_epoch` and `val/loss` should both decrease. The validation loss
  can be lower than the training loss if the validation period is calmer.
- Each `val/loss_<variable>` should decrease. A curve that stays perfectly flat
  means the network predicts a constant for that variable.
- If the loss turns into NaN, rerun with `trainer.detect_anomaly=true` to find
  where it starts.

## 7. Evaluate in rollout mode and tune the noise

The validation loss measures one-step skill. What matters in the coupled run is
how errors grow when the network is fed its own predictions, so models (and
noise settings) are compared on **rollouts over the validation period**.

**a) Train without noise first** (step 6, without `model.prev_noise_std`).

**b) Run rollouts over the validation period.** Start a rollout every 3rd day
of the validation period, each with a maximum lead time of about 5 days, and
average the RMSE per lead time over all start times. A single rollout reflects
the weather of that particular period; averaging over many start times gives a
fair comparison.

> TODO (Caroline): link the Slurm script that starts the rollouts and the
> notebook that evaluates them.

Each rollout writes `rollout_metrics_lead_from_<start>_<offset>.csv` with the
RMSE per lead time (`rmse_<variable>`) and a persistence baseline
(`rmse_persistence_<variable>`: the wave field of the start time held
constant). A useful model should beat persistence at all lead times.

**c) Train with noise.** Noise is set per target in physical units under
`model.prev_noise_std`. We set it as a fraction of each variable's std from the
scaling file, e.g. 0.5 × std:

```yaml
model:
  prev_noise_std:
    sigWaveHeight: 0.496      # 0.5 x std 0.991 m
    meanWavePeriod: 0.883     # 0.5 x std 1.766 s
    sin_meanWaveDirection: 0.35  # 0.5 x ~0.71
    ...
  prev_noise_constrain: true  # keep noised inputs physical (>= 0, on the unit circle)
```

See `configs/experiment/wwnet2_lstm_noise_std50.yaml` for a complete example.
Train one model per noise level (currently 0.25 × std and 0.5 × std).

Two side effects to expect: the validation loss gets *worse* with noise (it is
measured with clean inputs), so the early-stopping patience is raised in the
noise configs; and the checkpoint should be chosen by rollout skill, not by
validation loss.

**d) Compare** the rollouts of all noise levels (and the run without noise) as
in b). Pick the setting with the lowest RMSE over the lead times that matter
for the coupled run; also check high wave heights (`target_bins`) and the
persistence baseline.

**e) Only at the very end**, run the chosen model on the test period, in the
same way but with `data.test_days` set to the test ranges.

## 8. Export to TorchScript for SCHISM

SCHISM loads the network through FTorch as a TorchScript file (`.pt`). The
export bakes the scaling into the file, so SCHISM passes and receives physical
units.

1. Export the chosen checkpoint with its experiment config:

   ```bash
   uv run python src/export_torchscript.py --experiment <new_name> \
     --ckpt-path "logs/train/<experiment_name>/runs/<date>/.../checkpoints/<file>.ckpt" \
     --label <new_name>
   ```

   This writes `logs/trained_windwavenet_checkpoints/torchscript/<label>.pt`
   (change with `--output-dir`). The model type is detected from the experiment
   config. The export is checked against the original model on independent
   data; the run ends with `OK` or `FAILED verification`.

   **Pass every setting you changed on the command line at training time**
   with `--override`, e.g. `--override data.seq_len=48`. The network is
   rebuilt from the experiment config before the trained weights are loaded,
   so a different `seq_len`, `hidden_dim` etc. gives a wrong or failing export.
2. Copy the `.pt` file to where the SCHISM run expects it (`model_path` in
   `windwavenet.nml`).
3. In `windwavenet.nml`, `input_features` and `output_features` must list the
   variables **in the same order** as `data.features` and `data.targets`, and
   `sequence_length` (LSTM) and `edge_length_scale` (graphs) must match
   `data.seq_len` and `data.edge_weight_scale` of the training config. The
   exported file expects and returns physical units; the scaling is built in.

> New output variables also need changes on the SCHISM side: the Fortran
> coupling currently maps only a fixed set of outputs to SCHISM's wave arrays.
> See `WINDWAVENET_CHANGES.md` in the SCHISM repository.

## Common pitfalls

- `input_dim`/`output_dim` not matching the number of features/targets: the
  run fails at the first batch with a shape error.
- A variable with std = 0 in the scaling file: training produces NaN at once.
- Scaling computed over the wrong period or from a different run: the network
  trains, but worse; always recompute for new data.
- Forgetting the `prev_` entry for a new target: the variable is not fed back
  in rollouts.
- Judging a model by validation loss alone: always compare rollouts.
