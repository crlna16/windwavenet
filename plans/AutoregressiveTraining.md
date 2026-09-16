<!--
SPDX-FileCopyrightText: 2025-2026 Hereon
SPDX-FileContributor: Caroline Arnold

SPDX-License-Identifier: MIT
-->

# Autoregressive (rollout) training — removed feature

This document records the autoregressive (AR) **training** feature that was removed
from this repository for clarity, and everything needed to reinstantiate it.

**This is not about inference.** Pseudo-coupled autoregressive rollout *evaluation*
(`src/rollout.py`, `configs/rollout.yaml`) is a separate, still-supported feature and
was not touched. AR training and rollout evaluation share only the
`build_feedback_map()` utility (`src/data/dataset_utils.py`), which remains in the
codebase.

To reinstate, either check out the last commit before this removal, or recreate the
files below verbatim (their full source is embedded here) and re-add the config
files and command-line entries.

## Concept

Normal ("teacher-forced") training feeds the model ground-truth `prev_<target>`
values and computes a one-step loss. AR training instead:

1. Takes a window of `rollout_steps` (T) consecutive time steps per sample.
2. Runs the model forward at step 0, producing a prediction.
3. **Detaches** that prediction and writes it (rescaled to feature space) into the
   `prev_<target>` input slot for step 1.
4. Repeats for all T steps, then averages the loss over all T steps.

Detaching the fed-back prediction blocks gradients from later steps corrupting
earlier steps' learning — it is a free-running rollout for the *inputs*, but not for
the *gradient*. `delta_*` targets are rejected because feeding back a delta would
require reconstructing the absolute state, coupling reconstruction error into the
rollout in a way that needs separate treatment. There was **no curriculum
learning / scheduled sampling** — `rollout_steps` was a fixed hyperparameter.

The cheap ~1×-cost alternative — perturbing `prev_*` inputs with noise instead of the
model's own T-step rollout error — is **not removed**; see `model.prev_noise_std` in
[README.md](README.md) and `src/model/noise.py`.

## Architecture summary

- **`AutoregressiveMixin`** (`src/model/autoregressive.py`): a mixin combined with
  `WWMEmulator` / `GraphWWMEmulator` LightningModules. Provides `_init_autoregressive`,
  `on_fit_start` (loads the feedback map), `_forward_step`, `_rollout`,
  `_compute_ar_loss`, `model_step`, and a `validation_step` override that additionally
  logs `val/loss_step0` (the single-step, lead-time-0 loss).
- **Three concrete emulators**, one per backbone family:
  - `ARWWMEmulator` (`src/model/ARWWMEmulator.py`) — pointwise (MLP, `window_len=1`)
    and LSTM (`window_len=seq_len`) backbones.
  - `ARGraphWWMEmulator` (`src/model/ARGraphWWMEmulator.py`) — pointwise graph
    backbones (`MultiHeadGAT`, `MultiHeadGATVN`); re-forwards the same PyG `Batch`
    each step with `.x` replaced by that step's node-feature slice.
  - `ARCenterHistoryGraphWWMEmulator` (`src/model/ARCenterHistoryGraphWWMEmulator.py`)
    — `MultiHeadGATLSTMVN`; each step derives both the current-time `.x` for all
    nodes and the center node's `.x_center_seq` history from the buffer.
- **Four AR datasets** (`src/data/AutoregressiveWWMDatasets.py`), each returning a
  window of consecutive time steps instead of a single step:
  - `SimpleARWWMDataset` (pointwise/MLP): window shape `(rollout_steps, n_features)`.
  - `SequenceARWWMDataset` (LSTM): buffer shape
    `(seq_len + rollout_steps - 1, n_features)`; step `s` reads `[s, s+seq_len)`.
  - `GraphARWWMDataset` (pointwise graph): windows over `rollout_steps` for all
    subgraph nodes.
  - `CenterHistoryARWWMDataset` (graph+LSTM): buffer of
    `history_len + rollout_steps - 1` steps.
- **Four AR data modules**, each swapping in the AR dataset only for the `"fit"`
  stage (train/val); `"test"`/`"predict"` fall back to the plain single-step dataset
  from the base data module (since rollout *evaluation* is `src/rollout.py`'s job):
  `ARWWMDataModule`, `SequenceARWWMDataModule`, `GraphARWWMDataModule`,
  `CenterHistoryARWWMDataModule` (all in `src/data/`). Each builds `self.feedback_map`
  via `du.build_feedback_map()` once fit datasets exist.
- **`build_feedback_map()`** (`src/data/dataset_utils.py`, kept in the codebase) maps
  every `prev_<target>` feature to `(feature_idx, target_idx, mean, std)`, used to
  write a rescaled prediction back into the next step's input.

## Reinstating

### 1. `src/model/autoregressive.py`

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Mixin adding free-running autoregressive rollout training to WWM LightningModules.

Combine with ``WWMEmulator`` (pointwise/LSTM backbones) or
``GraphWWMEmulator`` (graph backbones) to train a model that rolls out
``rollout_steps`` predictions, feeding each one back into the
``prev_<target>`` input slot of the next step, and averages the loss over
all rollout steps. See ``src/model/ARWWMEmulator.py`` for the pointwise/LSTM
subclass and ``configs/experiment/*_ar.yaml`` for example configs.
"""

from typing import Dict, List, Tuple

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)

FeedbackMap = List[Tuple[int, int, float, float]]


class AutoregressiveMixin:
    """Adds rollout training to a WWMEmulator-like LightningModule.

    Subclasses must set ``self.window_len`` (1 for pointwise models,
    ``seq_len`` for sequence models) and be combined with a LightningModule
    that provides ``self.criterion``, ``self.target_names``,
    ``self.loss_weights``, and ``self.forward``. Requires a datamodule that
    exposes ``feedback_map`` (e.g. ``ARWWMDataModule``).
    """

    window_len: int = 1

    def _init_autoregressive(self, rollout_steps: int) -> None:
        """Store rollout config and placeholder feedback-map buffers.

        Args:
            rollout_steps: Number of future steps (T) to roll out and average
                the loss over.
        """
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
        self.rollout_steps = rollout_steps
        self._feedback_map_ready = False
        self.register_buffer("feedback_feat_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("feedback_target_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("feedback_mean", torch.zeros(0), persistent=False)
        self.register_buffer("feedback_std", torch.zeros(0), persistent=False)

    def on_fit_start(self) -> None:
        """Load the feedback map from the datamodule before training/validation begins."""
        super().on_fit_start()
        self._load_feedback_map()

    def _load_feedback_map(self) -> None:
        """Fetch ``feedback_map`` from the datamodule and register it as index/scale buffers."""
        datamodule = getattr(self.trainer, "datamodule", None)
        feedback_map: FeedbackMap = getattr(datamodule, "feedback_map", None)
        if feedback_map is None:
            raise RuntimeError(
                "Autoregressive training requires a datamodule exposing `feedback_map` "
                "(list of (feature_idx, target_idx, mean, std)), e.g. ARWWMDataModule."
            )
        if len(feedback_map) == 0:
            log.warning("Feedback map is empty: no `prev_<target>` feature will be updated during rollout.")

        feat_idx, tgt_idx, means, stds = zip(*feedback_map) if feedback_map else ((), (), (), ())
        device = self.device
        self.feedback_feat_idx = torch.tensor(feat_idx, dtype=torch.long, device=device)
        self.feedback_target_idx = torch.tensor(tgt_idx, dtype=torch.long, device=device)
        self.feedback_mean = torch.tensor(means, dtype=torch.float32, device=device)
        self.feedback_std = torch.tensor(stds, dtype=torch.float32, device=device)
        self._feedback_map_ready = True

    def _forward_step(self, x_buf: torch.Tensor, s: int) -> torch.Tensor:
        """Extract inputs for rollout step ``s`` from ``x_buf`` and run forward.

        Default implementation for pointwise (``window_len == 1``) and
        sequence/LSTM (``window_len == seq_len``) models: slices a plain
        tensor and forwards it directly. Graph models override this to wrap
        the slice into a PyG ``Data``/``Batch`` object before forwarding
        (see ``ARGraphWWMEmulator``).

        Args:
            x_buf: Rolling input buffer, see ``_rollout``.
            s: Current rollout step index (0-indexed).

        Returns:
            Predictions for step ``s``, shape ``(rows, n_targets)``.
        """
        window_len = self.window_len
        x_step = x_buf[:, s, :] if window_len == 1 else x_buf[:, s : s + window_len, :]
        return self.forward(x_step)

    def _rollout(self, x_buf: torch.Tensor) -> torch.Tensor:
        """Roll out ``rollout_steps`` predictions, feeding each back into ``x_buf``.

        Args:
            x_buf: Rolling input buffer of shape ``(rows, window_len + T - 1,
                n_features)``, where ``rows`` is ``batch`` for pointwise/LSTM
                models or ``total_nodes_in_pyg_batch`` for graph models. For
                pointwise models (``window_len == 1``) this is simply
                ``(rows, T, n_features)``. A clone is mutated internally with
                fed-back predictions; the input is not modified.

        Returns:
            Predictions of shape ``(rows, T, n_targets)``.
        """
        if not self._feedback_map_ready:
            raise RuntimeError("Feedback map not loaded; autoregressive training requires `trainer.fit`.")

        x_buf = x_buf.clone()
        window_len = self.window_len
        has_feedback = self.feedback_feat_idx.numel() > 0
        preds = []
        for s in range(self.rollout_steps):
            pred = self._forward_step(x_buf, s)
            preds.append(pred)

            # Last step's prediction feeds no further step, so it needs no write-back.
            if has_feedback and s < self.rollout_steps - 1:
                fed_back = pred[:, self.feedback_target_idx].detach()
                scaled = (fed_back - self.feedback_mean) / self.feedback_std
                x_buf[:, s + window_len, self.feedback_feat_idx] = scaled

        return torch.stack(preds, dim=1)

    def _compute_ar_loss(
        self, x_buf: torch.Tensor, y_win: torch.Tensor, sample_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Roll out predictions and average the loss over all ``rollout_steps``.

        Shared by the tuple-batch ``model_step`` below and graph AR emulators,
        which unpack their PyG ``Batch`` into the same ``(x_buf, y_win,
        sample_weights)`` shapes before calling this.

        Args:
            x_buf: Rolling input buffer, see ``_rollout``.
            y_win: Targets, shape ``(rows, T, n_targets)``.
            sample_weights: Per-step sample weights, shape ``(rows, T, n_targets)``.

        Returns:
            Tuple of (total_loss, per_variable_losses, predictions, targets), with
            predictions/targets flattened to ``(rows * T, num_targets)`` so that
            the base class's logging methods (which expect 2D tensors) work unchanged.
        """
        preds = self._rollout(x_buf)  # (rows, T, C)
        # Stashed for validation_step's single-step (lead time 0) loss log.
        self._last_ar_preds = preds
        self._last_ar_targets = y_win

        per_sample_loss = self.criterion(preds, y_win)  # (rows, T, C)

        weighted_loss = per_sample_loss * sample_weights

        per_variable_losses = {}
        for i, name in enumerate(self.target_names):
            per_variable_losses[name] = weighted_loss[:, :, i].mean()

        total_loss = sum(per_variable_losses[name] * self.loss_weights[name] for name in self.target_names)

        b, t, c = preds.shape
        return total_loss, per_variable_losses, preds.reshape(b * t, c), y_win.reshape(b * t, c)

    def model_step(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Roll out predictions over ``rollout_steps`` and average the loss over all of them.

        Args:
            batch: ``(x_buf, y_win)`` or ``(x_buf, y_win, sample_weights)``, where
                ``y_win`` and ``sample_weights`` have shape ``(batch, T, n_targets)``.

        Returns:
            Tuple of (total_loss, per_variable_losses, predictions, targets), with
            predictions/targets flattened to ``(batch * T, num_targets)`` so that
            the base class's logging methods (which expect 2D tensors) work unchanged.
        """
        if len(batch) == 3:
            x_buf, y_win, sample_weights = batch
        else:
            x_buf, y_win = batch
            sample_weights = torch.ones_like(y_win)

        return self._compute_ar_loss(x_buf, y_win, sample_weights)

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Run the T-step validation step, plus log the single-step (lead time 0) loss.

        Args:
            batch: Same format as ``model_step``.
            batch_idx: The index of the current batch.
        """
        super().validation_step(batch, batch_idx)
        if self.trainer.sanity_checking:
            return
        loss_step0 = self.criterion(self._last_ar_preds[:, 0, :], self._last_ar_targets[:, 0, :]).mean()
        self.log("val/loss_step0", loss_step0, on_step=False, on_epoch=True, sync_dist=True)
```

### 2. `src/model/ARWWMEmulator.py`

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Autoregressive (rollout) variant of WWMEmulator for pointwise/LSTM backbones."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.model.autoregressive import AutoregressiveMixin
from src.model.WWMEmulator import WWMEmulator


class ARWWMEmulator(AutoregressiveMixin, WWMEmulator):
    """WWMEmulator trained autoregressively over ``rollout_steps`` future steps.

    Point-wise (MLP) models use ``window_len=1`` (one step per rollout
    iteration); sequence (LSTM) models set ``window_len=seq_len`` so each
    rollout iteration consumes a full input sequence.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_names: List[str],
        optimizer: torch.optim.Optimizer,
        rollout_steps: int,
        window_len: int = 1,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        loss_weights: Optional[Dict[str, float]] = None,
        target_bins: Optional[Dict[str, List[float]]] = None,
        output_constraints: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the autoregressive WWM Emulator.

        Args:
            backbone: The neural network backbone.
            target_names: List of target variable names (e.g., ['sigWaveHeight']).
            optimizer: PyTorch optimizer (partially instantiated via Hydra).
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            window_len: Number of input time steps consumed per rollout iteration
                (1 for pointwise/MLP models, ``seq_len`` for LSTM models).
            scheduler: PyTorch learning rate scheduler (partially instantiated via Hydra).
            loss_weights: Optional dict mapping target names to loss weights.
            target_bins: Optional dict mapping target names to bin edges for binned metrics.
            output_constraints: Optional dict mapping target names to constraint types.
        """
        super().__init__(
            backbone=backbone,
            target_names=target_names,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_weights=loss_weights,
            target_bins=target_bins,
            output_constraints=output_constraints,
        )
        self.window_len = window_len
        self._init_autoregressive(rollout_steps=rollout_steps)
        self.save_hyperparameters("rollout_steps", "window_len", logger=False)
```

### 3. `src/model/ARGraphWWMEmulator.py`

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Autoregressive (rollout) variant of GraphWWMEmulator for pointwise graph backbones."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.autoregressive import AutoregressiveMixin
from src.model.GraphWWMEmulator import GraphWWMEmulator


class ARGraphWWMEmulator(AutoregressiveMixin, GraphWWMEmulator):
    """GraphWWMEmulator trained autoregressively over ``rollout_steps`` future steps.

    For pointwise graph backbones (``MultiHeadGAT``, ``MultiHeadGATVN``) that
    consume only current-time node features. Each rollout step re-forwards
    the SAME PyG ``Batch`` (topology is time-invariant) with ``x`` replaced
    by that step's node-feature slice; predictions are fed back into the
    ``prev_<target>`` feature of every subgraph node for the next step.
    """

    window_len: int = 1

    def __init__(
        self,
        backbone: nn.Module,
        target_names: List[str],
        optimizer: torch.optim.Optimizer,
        rollout_steps: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        loss_weights: Optional[Dict[str, float]] = None,
        target_bins: Optional[Dict[str, List[float]]] = None,
        output_constraints: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the autoregressive graph WWM Emulator.

        Args:
            backbone: The neural network backbone.
            target_names: List of target variable names (e.g., ['sigWaveHeight']).
            optimizer: PyTorch optimizer (partially instantiated via Hydra).
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            scheduler: PyTorch learning rate scheduler (partially instantiated via Hydra).
            loss_weights: Optional dict mapping target names to loss weights.
            target_bins: Optional dict mapping target names to bin edges for binned metrics.
            output_constraints: Optional dict mapping target names to constraint types.
        """
        super().__init__(
            backbone=backbone,
            target_names=target_names,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_weights=loss_weights,
            target_bins=target_bins,
            output_constraints=output_constraints,
        )
        self._init_autoregressive(rollout_steps=rollout_steps)
        self.save_hyperparameters("rollout_steps", logger=False)

        # Set per model_step() to the current PyG Batch; reused (mutated `.x`
        # only) across rollout steps since topology is time-invariant.
        self._graph_template = None

    def _forward_step(self, x_buf: torch.Tensor, s: int) -> torch.Tensor:
        """Forward rollout step ``s``: all subgraph nodes' current-time features.

        Args:
            x_buf: All-node feature buffer, shape ``(total_nodes, T, n_features)``.
            s: Current rollout step index (0-indexed).

        Returns:
            Predictions of shape ``(total_nodes, n_targets)``.
        """
        self._graph_template.x = x_buf[:, s, :]
        return self.forward(self._graph_template)

    def model_step(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Roll out predictions over ``rollout_steps`` for a windowed PyG batch.

        Args:
            batch: PyG Batch with ``x``/``y``/(optional) ``sample_weight`` of
                shape ``(total_nodes, T, n_features/n_targets/1)``. If the
                DataLoader returns a list, the first element is used.

        Returns:
            A tuple of (total_loss, per_variable_losses, predictions, targets).
        """
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        self._graph_template = batch
        x_buf = batch.x
        y_win = batch.y
        sample_weight = getattr(batch, "sample_weight", None)
        if sample_weight is None:
            sample_weight = torch.ones_like(y_win)

        return self._compute_ar_loss(x_buf, y_win, sample_weight)
```

### 4. `src/model/ARCenterHistoryGraphWWMEmulator.py`

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Autoregressive (rollout) variant of GraphWWMEmulator for center-history graph backbones."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.autoregressive import AutoregressiveMixin
from src.model.GraphWWMEmulator import GraphWWMEmulator


class ARCenterHistoryGraphWWMEmulator(AutoregressiveMixin, GraphWWMEmulator):
    """GraphWWMEmulator trained autoregressively for center-history backbones.

    For ``MultiHeadGATLSTMVN``, which needs both the current-time features of
    every subgraph node (``x``) and the center node's own temporal history
    (``x_center_seq``). Each rollout step re-forwards the SAME PyG ``Batch``
    (topology is time-invariant) with ``x``/``x_center_seq`` derived from the
    all-node feature buffer; predictions are fed back into the
    ``prev_<target>`` feature of every subgraph node for the next step.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_names: List[str],
        optimizer: torch.optim.Optimizer,
        rollout_steps: int,
        window_len: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        loss_weights: Optional[Dict[str, float]] = None,
        target_bins: Optional[Dict[str, List[float]]] = None,
        output_constraints: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the autoregressive center-history graph WWM Emulator.

        Args:
            backbone: The neural network backbone (``MultiHeadGATLSTMVN``).
            target_names: List of target variable names (e.g., ['sigWaveHeight']).
            optimizer: PyTorch optimizer (partially instantiated via Hydra).
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            window_len: Length of the center node's LSTM history window
                (must match ``data.history_len`` / the dataset's ``history_len``).
            scheduler: PyTorch learning rate scheduler (partially instantiated via Hydra).
            loss_weights: Optional dict mapping target names to loss weights.
            target_bins: Optional dict mapping target names to bin edges for binned metrics.
            output_constraints: Optional dict mapping target names to constraint types.
        """
        super().__init__(
            backbone=backbone,
            target_names=target_names,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_weights=loss_weights,
            target_bins=target_bins,
            output_constraints=output_constraints,
        )
        self.window_len = window_len
        self._init_autoregressive(rollout_steps=rollout_steps)
        self.save_hyperparameters("rollout_steps", "window_len", logger=False)

        # Set per model_step() to the current PyG Batch (mutated `.x`/`.x_center_seq`
        # only) and the global row position of each graph's center node.
        self._graph_template = None
        self._center_global_idx: Optional[torch.Tensor] = None

    def _forward_step(self, x_buf: torch.Tensor, s: int) -> torch.Tensor:
        """Forward rollout step ``s``: current features for all nodes + center node history.

        Args:
            x_buf: All-node feature buffer, shape
                ``(total_nodes, window_len + T - 1, n_features)``.
            s: Current rollout step index (0-indexed).

        Returns:
            Predictions of shape ``(total_nodes, n_targets)``.
        """
        window_len = self.window_len
        x_current = x_buf[:, s + window_len - 1, :]
        x_center_hist = x_buf[self._center_global_idx, s : s + window_len, :]
        self._graph_template.x = x_current
        self._graph_template.x_center_seq = x_center_hist
        return self.forward(self._graph_template)

    def model_step(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Roll out predictions over ``rollout_steps`` for a windowed PyG batch.

        Args:
            batch: PyG Batch with ``x``/``y``/(optional) ``sample_weight`` of
                shape ``(total_nodes, window_len + T - 1 or T, n_features/n_targets/1)``,
                plus ``center_idx`` (local per-graph center index) and ``ptr``
                (per-graph node offsets, added automatically by PyG batching).
                If the DataLoader returns a list, the first element is used.

        Returns:
            A tuple of (total_loss, per_variable_losses, predictions, targets).
        """
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        self._graph_template = batch
        # Global row position of each graph's center node in the batched x/y buffers.
        self._center_global_idx = batch.ptr[:-1] + batch.center_idx

        x_buf = batch.x
        y_win = batch.y
        sample_weight = getattr(batch, "sample_weight", None)
        if sample_weight is None:
            sample_weight = torch.ones_like(y_win)

        return self._compute_ar_loss(x_buf, y_win, sample_weight)
```

### 5. `src/data/AutoregressiveWWMDatasets.py`

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Autoregressive (rollout) training datasets for the WWM emulator.

These datasets return short windows of consecutive time steps instead of a
single step, so that a LightningModule using ``AutoregressiveMixin``
(``src/model/autoregressive.py``) can roll out ``rollout_steps`` predictions
and feed each one back into the ``prev_<target>`` input slot of the next
step.

Kept separate from ``WWMDatasets.py`` to avoid cluttering the core
(single-step) dataset implementations; classes here subclass the base
datasets and reuse their data loading/scaling/preprocessing pipeline.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from src.data import dataset_utils as du
from src.data.WWMDatasets import SequenceGraphWWMDataset, SequenceWWMDataset, SimpleWWMDataset, WWMGraphData
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)

DayRange = List[int]
DataRange = Union[DayRange, List[DayRange]]


def _reject_rollout_steps_below_one(rollout_steps: int) -> None:
    """Raise if ``rollout_steps`` is not a valid rollout horizon."""
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")


def _reject_delta_targets(targets: List[str]) -> None:
    """Raise if any target is a ``delta_*`` variable.

    Feeding back a delta prediction would require reconstructing the absolute
    state at every step, coupling reconstruction error into the rollout loss.
    """
    if any(str(t).startswith("delta_") for t in targets):
        raise ValueError(f"Autoregressive training does not support delta targets: {targets}")


class SimpleARWWMDataset(SimpleWWMDataset):
    """Point-wise dataset returning ``rollout_steps`` consecutive time steps.

    Each sample is a window of ``rollout_steps`` consecutive (time, node)
    pairs at a fixed node, used to train pointwise (MLP) models
    autoregressively: the model predicts step 0, its prediction is fed back
    into the ``prev_<target>`` feature for step 1, and so on.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
    ) -> None:
        """Initialize the autoregressive point-wise dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits (e.g., [[1, 10], [20, 30]]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            rollout_steps: Number of consecutive time steps per window (T).
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights
                (default: 'sigWaveHeight'). Any variable in the dataset, or 'delta_<var>'
                (derived on the fly if absent).
        """
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
        # Feeding back a delta prediction would require reconstructing the absolute
        # state at every step, coupling reconstruction error into the rollout loss.
        _reject_delta_targets(targets)

        self.rollout_steps = rollout_steps

        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )

        self.n_time = len(self.ds.time)

        # Valid window start positions per contiguous day range, so that no
        # window straddles the gap between ranges. Raises if a range is
        # shorter than rollout_steps.
        self._valid_starts = du.compute_valid_sequence_starts(self.range_lengths, self.rollout_steps)
        self.n_valid_times = len(self._valid_starts)

        # Reshape preloaded data for efficient windowed access: (n_time, n_nodes, n_features)
        n_features = self.X.shape[1]
        n_targets = self.y.shape[1]
        self.X = self.X.view(self.n_time, self.n_nodes, n_features)
        self.y = self.y.view(self.n_time, self.n_nodes, n_targets)

        log.info(
            f"SimpleARWWMDataset initialized with rollout_steps={rollout_steps}, "
            f"n_valid_times={self.n_valid_times}, n_nodes={self.n_nodes}"
        )

    def __len__(self) -> int:
        """Return the number of valid rollout windows.

        Returns:
            Number of samples: n_valid_times * n_nodes
        """
        return self.n_valid_times * self.n_nodes

    def __getitem__(self, idx: int) -> Tuple:
        """Get a rollout window sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (X_win, y_win[, weight_win]) as PyTorch tensors, where:
                - X_win: (rollout_steps, n_features) input features, one row
                  per time step in the window.
                - y_win: (rollout_steps, n_targets) targets at the same steps.
                - weight_win: (rollout_steps, 1) per-step sample weight, if
                  sample weighting is enabled.
        """
        # Map flat index to (window start, node); window starts skip range gaps
        time_start = int(self._valid_starts[idx // self.n_nodes])
        node_idx = idx % self.n_nodes

        X_win = self.X[time_start : time_start + self.rollout_steps, node_idx, :]
        y_win = self.y[time_start : time_start + self.rollout_steps, node_idx, :]

        if self.weights is not None:
            # Weights are stored flat over (time, node) with time as the slow axis.
            flat_idx = (time_start + torch.arange(self.rollout_steps)) * self.n_nodes + node_idx
            weight_win = self.weights[flat_idx]
            return X_win, y_win, weight_win

        return X_win, y_win


class SequenceARWWMDataset(SequenceWWMDataset):
    """Sequence dataset returning ``rollout_steps`` consecutive LSTM windows.

    Each sample is a buffer of ``seq_len + rollout_steps - 1`` consecutive
    time steps at a fixed node. Rollout step ``s`` consumes the sub-window
    ``[s, s + seq_len)`` (an LSTM input of length ``seq_len``) and predicts
    the target at ``s + seq_len - 1``; its prediction is fed back into the
    ``prev_<target>`` feature at buffer position ``s + seq_len`` before the
    next step, exactly like ``SimpleARWWMDataset`` but with a sliding
    ``seq_len``-step window instead of a single time step per rollout
    iteration.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        seq_len: int,
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
    ) -> None:
        """Initialize the autoregressive sequence dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits (e.g., [[1, 10], [20, 30]]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            seq_len: Length of the LSTM input window consumed per rollout iteration.
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights
                (default: 'sigWaveHeight'). Any variable in the dataset, or 'delta_<var>'
                (derived on the fly if absent).
        """
        _reject_rollout_steps_below_one(rollout_steps)
        _reject_delta_targets(targets)

        self.history_len = seq_len
        self.rollout_steps = rollout_steps

        # Combined buffer length: `seq_len` history steps for the first
        # rollout iteration, plus one extra step per subsequent iteration.
        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            seq_len=seq_len + rollout_steps - 1,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )

        log.info(
            f"SequenceARWWMDataset initialized with seq_len={seq_len}, rollout_steps={rollout_steps}, "
            f"n_valid_times={self.n_valid_times}, n_nodes={self.n_nodes}"
        )

    def __getitem__(self, idx: int) -> Tuple:
        """Get a rollout window sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (X_buf, y_win[, weight_win]) as PyTorch tensors, where:
                - X_buf: (seq_len + rollout_steps - 1, n_features) rolling
                  input buffer; rollout step ``s`` reads ``X_buf[s : s + seq_len]``.
                - y_win: (rollout_steps, n_targets) targets, one per rollout step.
                - weight_win: (rollout_steps, 1) per-step sample weight, if
                  sample weighting is enabled.
        """
        # Map flat index to (window start, node); window starts skip range gaps
        time_start = int(self._valid_starts[idx // self.n_nodes])
        node_idx = idx % self.n_nodes

        X_buf = self.X[time_start : time_start + self.seq_len, node_idx, :]

        present_times = time_start + self.history_len - 1 + torch.arange(self.rollout_steps)
        y_win = self.y[present_times, node_idx, :]

        if self.weights is not None:
            # Weights are stored flat over (time, node) with time as the slow axis.
            flat_idx = present_times * self.n_nodes + node_idx
            weight_win = self.weights[flat_idx]
            return X_buf, y_win, weight_win

        return X_buf, y_win


class GraphARWWMDataset(SequenceGraphWWMDataset):
    """Windowed graph dataset for AR rollout training of pointwise graph models.

    For ``MultiHeadGAT`` / ``MultiHeadGATVN`` (no per-node temporal history):
    every subgraph node's features and targets are windowed over
    ``rollout_steps`` consecutive time steps, so rollout step ``s`` reads
    ``x[:, s, :]`` (all nodes' current features) and predicts ``y[:, s, :]``
    (all nodes' targets); its prediction is fed back into the ``prev_<target>``
    feature at position ``s + 1`` before the next step.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
        max_nodes: int = 1000,
        edge_weight_scale: float = 1000.0,
        cache_mode: bool = True,
    ) -> None:
        """Initialize the autoregressive pointwise graph dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names used for constructing the graph.
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with center node indices.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.
            max_nodes: Maximum number of nodes in a subgraph.
            edge_weight_scale: Length scale (meters) for inverse-distance edge weights.
            cache_mode: If True, preload target and feature tensors into shared RAM.
        """
        _reject_rollout_steps_below_one(rollout_steps)
        _reject_delta_targets(targets)

        self.rollout_steps = rollout_steps

        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            graph_features=graph_features,
            seq_len=rollout_steps,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
            max_nodes=max_nodes,
            edge_weight_scale=edge_weight_scale,
            cache_mode=cache_mode,
        )

        log.info(
            f"GraphARWWMDataset initialized: rollout_steps={rollout_steps}, "
            f"{self.n_center_nodes} center nodes, {self._n_valid_times} valid times."
        )

    def _extract_target_window(self, time_start: int, subgraph_nodes) -> torch.Tensor:
        """Extract targets for a subgraph over ``rollout_steps`` consecutive time steps.

        Args:
            time_start: First time index of the window (0-indexed).
            subgraph_nodes: Array of 0-indexed positional node indices.

        Returns:
            Tensor of shape (n_subgraph_nodes, rollout_steps, n_targets).
        """
        time_indices = list(range(time_start, time_start + self.rollout_steps))
        if hasattr(self, "_targets_ram"):
            compact = self._global_to_compact[subgraph_nodes]
            return self._targets_ram[time_indices][:, compact, :].permute(1, 0, 2).contiguous()

        sample = self.ds.isel(time=time_indices, nSCHISM_hgrid_node=subgraph_nodes)
        y_da = sample[self.targets].to_dataarray(dim="variable")
        values = y_da.values.transpose(2, 1, 0).copy()  # (node, time, variable)
        return torch.from_numpy(values).float()

    def _extract_weight_window(self, subgraph_nodes, time_indices: List[int]) -> Optional[torch.Tensor]:
        """Extract per-node, per-step sample weights, or None if unavailable.

        Args:
            subgraph_nodes: Array of 0-indexed positional node indices.
            time_indices: Time indices of the window (0-indexed).

        Returns:
            Tensor of shape (n_subgraph_nodes, len(time_indices), 1), or None.
        """
        if self.weights is None:
            return None
        compact = self._global_to_compact[subgraph_nodes]  # (n_nodes,)
        times = torch.as_tensor(time_indices, dtype=torch.long)
        flat_idx = times.unsqueeze(0) * self.n_used_nodes + torch.as_tensor(compact, dtype=torch.long).unsqueeze(1)
        return self.weights[flat_idx]  # (n_nodes, len(time_indices), 1)

    def construct_graph(self, time_idx: int, center_node_pos: int) -> "WWMGraphData":
        """Construct a windowed graph sample for rollout training.

        Args:
            time_idx: First time index of the window (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            WWMGraphData with attributes:
                - x: (n_subgraph_nodes, rollout_steps, n_features) node features
                - edge_index: (2, n_edges) edge index tensor
                - edge_weight: (n_edges,) inverse-distance edge weights
                - y: (n_subgraph_nodes, rollout_steps, n_targets) targets
                - center_idx: local index of center node in x
                - sample_weight: (n_subgraph_nodes, rollout_steps, 1), or None
        """
        subgraph_nodes, edge_index, center_local_idx, edge_weight = self.graph_cache[center_node_pos]
        x = self._extract_sequence_node_features(time_idx, subgraph_nodes)
        y = self._extract_target_window(time_idx, subgraph_nodes)
        time_indices = list(range(time_idx, time_idx + self.rollout_steps))
        sample_weight = self._extract_weight_window(subgraph_nodes, time_indices)

        return WWMGraphData(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            center_idx=center_local_idx,
            sample_weight=sample_weight,
        )


class CenterHistoryARWWMDataset(SequenceGraphWWMDataset):
    """Windowed graph dataset for AR rollout training of center-history graph models.

    For ``MultiHeadGATLSTMVN``: every subgraph node's features are windowed
    over ``history_len + rollout_steps - 1`` consecutive time steps, so any
    node's ``prev_<target>`` feature can be fed back during rollout. Rollout
    step ``s`` derives:
      - ``x`` (current-time features for ALL nodes) from buffer position
        ``s + history_len - 1``;
      - ``x_center_seq`` (temporal history for the CENTER node only) from the
        buffer sub-window ``[s, s + history_len)``.
    Its prediction (for all subgraph nodes) is fed back into the
    ``prev_<target>`` feature at buffer position ``s + history_len`` before
    the next step. Targets are windowed at the same "present" times as ``x``.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        history_len: int,
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
        max_nodes: int = 1000,
        edge_weight_scale: float = 1000.0,
        cache_mode: bool = True,
    ) -> None:
        """Initialize the autoregressive center-history graph dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names used for constructing the graph.
            history_len: Length of the center node's LSTM history window.
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with center node indices.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.
            max_nodes: Maximum number of nodes in a subgraph.
            edge_weight_scale: Length scale (meters) for inverse-distance edge weights.
            cache_mode: If True, preload target and feature tensors into shared RAM.
        """
        _reject_rollout_steps_below_one(rollout_steps)
        _reject_delta_targets(targets)

        self.history_len = history_len
        self.rollout_steps = rollout_steps

        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            graph_features=graph_features,
            seq_len=history_len + rollout_steps - 1,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
            max_nodes=max_nodes,
            edge_weight_scale=edge_weight_scale,
            cache_mode=cache_mode,
        )

        log.info(
            f"CenterHistoryARWWMDataset initialized: history_len={history_len}, "
            f"rollout_steps={rollout_steps}, {self.n_center_nodes} center nodes, "
            f"{self._n_valid_times} valid times."
        )

    def _extract_target_window(self, time_start: int, subgraph_nodes) -> torch.Tensor:
        """Extract targets at the ``rollout_steps`` "present" times (aligned with ``x``).

        Args:
            time_start: First time index of the buffer window (0-indexed).
            subgraph_nodes: Array of 0-indexed positional node indices.

        Returns:
            Tensor of shape (n_subgraph_nodes, rollout_steps, n_targets).
        """
        present_start = time_start + self.history_len - 1
        time_indices = list(range(present_start, present_start + self.rollout_steps))
        if hasattr(self, "_targets_ram"):
            compact = self._global_to_compact[subgraph_nodes]
            return self._targets_ram[time_indices][:, compact, :].permute(1, 0, 2).contiguous()

        sample = self.ds.isel(time=time_indices, nSCHISM_hgrid_node=subgraph_nodes)
        y_da = sample[self.targets].to_dataarray(dim="variable")
        values = y_da.values.transpose(2, 1, 0).copy()  # (node, time, variable)
        return torch.from_numpy(values).float()

    def _extract_weight_window(self, subgraph_nodes, time_indices: List[int]) -> Optional[torch.Tensor]:
        """Extract per-node, per-step sample weights, or None if unavailable.

        Args:
            subgraph_nodes: Array of 0-indexed positional node indices.
            time_indices: Time indices of the window (0-indexed).

        Returns:
            Tensor of shape (n_subgraph_nodes, len(time_indices), 1), or None.
        """
        if self.weights is None:
            return None
        compact = self._global_to_compact[subgraph_nodes]  # (n_nodes,)
        times = torch.as_tensor(time_indices, dtype=torch.long)
        flat_idx = times.unsqueeze(0) * self.n_used_nodes + torch.as_tensor(compact, dtype=torch.long).unsqueeze(1)
        return self.weights[flat_idx]  # (n_nodes, len(time_indices), 1)

    def construct_graph(self, time_idx: int, center_node_pos: int) -> "WWMGraphData":
        """Construct a windowed center-history graph sample for rollout training.

        Args:
            time_idx: First time index of the buffer window (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            WWMGraphData with attributes:
                - x: (n_subgraph_nodes, history_len + rollout_steps - 1,
                  n_features) ALL-node feature buffer. ``ARCenterHistoryGraphWWMEmulator``
                  derives the current-time slice and the center node's
                  ``x_center_seq`` history from this per rollout step.
                - edge_index: (2, n_edges) edge index tensor
                - edge_weight: (n_edges,) inverse-distance edge weights
                - y: (n_subgraph_nodes, rollout_steps, n_targets) targets at
                  the "present" time of each rollout step
                - center_idx: local index of center node in x
                - sample_weight: (n_subgraph_nodes, rollout_steps, 1), or None
        """
        subgraph_nodes, edge_index, center_local_idx, edge_weight = self.graph_cache[center_node_pos]
        x = self._extract_sequence_node_features(time_idx, subgraph_nodes)
        y = self._extract_target_window(time_idx, subgraph_nodes)
        present_start = time_idx + self.history_len - 1
        time_indices = list(range(present_start, present_start + self.rollout_steps))
        sample_weight = self._extract_weight_window(subgraph_nodes, time_indices)

        return WWMGraphData(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            center_idx=center_local_idx,
            sample_weight=sample_weight,
        )
```

### 6. `src/data/ARWWMDataModule.py`

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""DataModule for autoregressive (rollout) training of pointwise WWM models."""

from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from src.data import dataset_utils as du
from src.data.AutoregressiveWWMDatasets import SimpleARWWMDataset
from src.data.WWMDataModule import WWMDataModule
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ARWWMDataModule(WWMDataModule):
    """Lightning DataModule for autoregressive point-wise WWM training.

    Train/validation samples are rollout windows of ``rollout_steps``
    consecutive time steps (``SimpleARWWMDataset``); test/predict keep the
    single-step ``SimpleWWMDataset`` inherited from ``WWMDataModule``, since
    autoregressive rollout *evaluation* is handled separately by
    ``src/rollout.py``.
    """

    def __init__(self, *args, rollout_steps: int, **kwargs) -> None:
        """Initialize the AR WWM DataModule.

        Args:
            *args: Positional args forwarded to ``WWMDataModule``.
            rollout_steps: Number of consecutive time steps (T) per training
                rollout window.
            **kwargs: Keyword args forwarded to ``WWMDataModule``.
        """
        self.rollout_steps = rollout_steps
        super().__init__(*args, **kwargs)
        # Populated in setup(stage="fit"); consumed by AutoregressiveMixin.
        self.feedback_map: List[Tuple[int, int, float, float]] = []
        self._ar_stage: Optional[str] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build datasets for *stage*, plus the feedback map after the fit datasets exist.

        Args:
            stage: Either 'fit', 'test', or 'predict'.
        """
        self._ar_stage = stage
        super().setup(stage)
        if stage == "fit" and not self.feedback_map:
            scaling_params = getattr(self.ds_train, "scaling_params", None)
            self.feedback_map = du.build_feedback_map(self.features, self.targets, scaling_params)

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        sample_weight_variable: str = "sigWaveHeight",
    ) -> Dataset:
        """Create the AR rollout dataset for fit, the plain dataset otherwise.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.

        Returns:
            A SimpleARWWMDataset (fit stage, i.e. train/val) or SimpleWWMDataset
            (test/predict stage).
        """
        if self._ar_stage != "fit":
            return super()._create_dataset(
                data_range, node_idx_csv, sample_weight_strategy, sample_weight_power, sample_weight_variable
            )

        return SimpleARWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            rollout_steps=self.rollout_steps,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=self.boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )
```

### 7. `src/data/SequenceARWWMDataModule.py`

Same pattern as `ARWWMDataModule`, but subclasses `SequenceWWMDataModule` and builds a
`SequenceARWWMDataset` for the fit stage (forwards `seq_len` in addition to
`rollout_steps`):

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""DataModule for autoregressive (rollout) training of sequence (LSTM) WWM models."""

from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from src.data import dataset_utils as du
from src.data.AutoregressiveWWMDatasets import SequenceARWWMDataset
from src.data.SequenceWWMDataModule import SequenceWWMDataModule
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class SequenceARWWMDataModule(SequenceWWMDataModule):
    """Lightning DataModule for autoregressive LSTM WWM training.

    Train/validation samples are rollout windows of ``seq_len + rollout_steps
    - 1`` consecutive time steps (``SequenceARWWMDataset``); test/predict
    keep the single-step ``SequenceWWMDataset`` inherited from
    ``SequenceWWMDataModule``, since autoregressive rollout *evaluation* is
    handled separately by ``src/rollout.py``.
    """

    def __init__(self, *args, rollout_steps: int, **kwargs) -> None:
        """Initialize the AR sequence WWM DataModule.

        Args:
            *args: Positional args forwarded to ``SequenceWWMDataModule``.
            rollout_steps: Number of consecutive rollout steps (T) per training window.
            **kwargs: Keyword args forwarded to ``SequenceWWMDataModule``
                (including ``seq_len``, the LSTM input window length).
        """
        self.rollout_steps = rollout_steps
        super().__init__(*args, **kwargs)
        # Populated in setup(stage="fit"); consumed by AutoregressiveMixin.
        self.feedback_map: List[Tuple[int, int, float, float]] = []
        self._ar_stage: Optional[str] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build datasets for *stage*, plus the feedback map after the fit datasets exist.

        Args:
            stage: Either 'fit', 'test', or 'predict'.
        """
        self._ar_stage = stage
        super().setup(stage)
        if stage == "fit" and not self.feedback_map:
            scaling_params = getattr(self.ds_train, "scaling_params", None)
            self.feedback_map = du.build_feedback_map(self.features, self.targets, scaling_params)

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        sample_weight_variable: str = "sigWaveHeight",
    ) -> Dataset:
        """Create the AR rollout dataset for fit, the plain dataset otherwise.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.

        Returns:
            A SequenceARWWMDataset (fit stage, i.e. train/val) or
            SequenceWWMDataset (test/predict stage).
        """
        if self._ar_stage != "fit":
            return super()._create_dataset(
                data_range, node_idx_csv, sample_weight_strategy, sample_weight_power, sample_weight_variable
            )

        return SequenceARWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            seq_len=self.seq_len,
            rollout_steps=self.rollout_steps,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=self.boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )
```

### 8. `src/data/GraphARWWMDataModule.py`

Same pattern, subclasses `GraphWWMDataModule`, builds `GraphARWWMDataset` (also
forwards `graph_features`, `max_nodes`, `edge_weight_scale`, `cache_mode`):

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""DataModule for autoregressive (rollout) training of pointwise graph WWM models."""

from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from src.data import dataset_utils as du
from src.data.AutoregressiveWWMDatasets import GraphARWWMDataset
from src.data.GraphWWMDataModule import GraphWWMDataModule
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class GraphARWWMDataModule(GraphWWMDataModule):
    """Lightning DataModule for autoregressive pointwise graph WWM training.

    Train/validation samples are rollout windows of ``rollout_steps``
    consecutive time steps (``GraphARWWMDataset``, for ``MultiHeadGAT`` /
    ``MultiHeadGATVN``); test/predict keep the single-step
    ``GraphWWMDataset`` inherited from ``GraphWWMDataModule``, since
    autoregressive rollout *evaluation* is handled separately by
    ``src/rollout.py``.
    """

    def __init__(self, *args, rollout_steps: int, **kwargs) -> None:
        """Initialize the AR graph WWM DataModule.

        Args:
            *args: Positional args forwarded to ``GraphWWMDataModule``.
            rollout_steps: Number of consecutive time steps (T) per training
                rollout window.
            **kwargs: Keyword args forwarded to ``GraphWWMDataModule``.
        """
        self.rollout_steps = rollout_steps
        super().__init__(*args, **kwargs)
        # Populated in setup(stage="fit"); consumed by AutoregressiveMixin.
        self.feedback_map: List[Tuple[int, int, float, float]] = []
        self._ar_stage: Optional[str] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build datasets for *stage*, plus the feedback map after the fit datasets exist.

        Args:
            stage: Either 'fit', 'test', or 'predict'.
        """
        self._ar_stage = stage
        super().setup(stage)
        if stage == "fit" and not self.feedback_map:
            scaling_params = getattr(self.ds_train, "scaling_params", None)
            self.feedback_map = du.build_feedback_map(self.features, self.targets, scaling_params)

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        sample_weight_variable: str = "sigWaveHeight",
    ) -> Dataset:
        """Create the AR rollout dataset for fit, the plain dataset otherwise.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.

        Returns:
            A GraphARWWMDataset (fit stage, i.e. train/val) or GraphWWMDataset
            (test/predict stage).
        """
        if self._ar_stage != "fit":
            return super()._create_dataset(
                data_range, node_idx_csv, sample_weight_strategy, sample_weight_power, sample_weight_variable
            )

        return GraphARWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            graph_features=self.graph_features,
            rollout_steps=self.rollout_steps,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=self.boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
            max_nodes=self.max_nodes,
            edge_weight_scale=self.edge_weight_scale,
            cache_mode=self.cache_mode,
        )
```

### 9. `src/data/CenterHistoryARWWMDataModule.py`

Same pattern, subclasses `CenterHistoryGraphWWMDataModule`, builds
`CenterHistoryARWWMDataset` with `history_len=self.seq_len`:

```python
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""DataModule for autoregressive (rollout) training of center-history graph WWM models."""

from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from src.data import dataset_utils as du
from src.data.AutoregressiveWWMDatasets import CenterHistoryARWWMDataset
from src.data.CenterHistoryGraphWWMDataModule import CenterHistoryGraphWWMDataModule
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class CenterHistoryARWWMDataModule(CenterHistoryGraphWWMDataModule):
    """Lightning DataModule for autoregressive center-history graph WWM training.

    Train/validation samples are rollout windows of ``seq_len + rollout_steps
    - 1`` consecutive time steps (``CenterHistoryARWWMDataset``, for
    ``MultiHeadGATLSTMVN``); test/predict keep the single-step
    ``CenterHistoryGraphWWMDataset`` inherited from
    ``CenterHistoryGraphWWMDataModule``, since autoregressive rollout
    *evaluation* is handled separately by ``src/rollout.py``.
    """

    def __init__(self, *args, rollout_steps: int, **kwargs) -> None:
        """Initialize the AR center-history graph WWM DataModule.

        Args:
            *args: Positional args forwarded to ``CenterHistoryGraphWWMDataModule``.
            rollout_steps: Number of consecutive rollout steps (T) per training window.
            **kwargs: Keyword args forwarded to ``CenterHistoryGraphWWMDataModule``
                (including ``seq_len``, the center node's LSTM history length).
        """
        self.rollout_steps = rollout_steps
        super().__init__(*args, **kwargs)
        # Populated in setup(stage="fit"); consumed by AutoregressiveMixin.
        self.feedback_map: List[Tuple[int, int, float, float]] = []
        self._ar_stage: Optional[str] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build datasets for *stage*, plus the feedback map after the fit datasets exist.

        Args:
            stage: Either 'fit', 'test', or 'predict'.
        """
        self._ar_stage = stage
        super().setup(stage)
        if stage == "fit" and not self.feedback_map:
            scaling_params = getattr(self.ds_train, "scaling_params", None)
            self.feedback_map = du.build_feedback_map(self.features, self.targets, scaling_params)

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        sample_weight_variable: str = "sigWaveHeight",
    ) -> Dataset:
        """Create the AR rollout dataset for fit, the plain dataset otherwise.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.

        Returns:
            A CenterHistoryARWWMDataset (fit stage, i.e. train/val) or
            CenterHistoryGraphWWMDataset (test/predict stage).
        """
        if self._ar_stage != "fit":
            return super()._create_dataset(
                data_range, node_idx_csv, sample_weight_strategy, sample_weight_power, sample_weight_variable
            )

        return CenterHistoryARWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            graph_features=self.graph_features,
            history_len=self.seq_len,
            rollout_steps=self.rollout_steps,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=self.boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
            max_nodes=self.max_nodes,
            edge_weight_scale=self.edge_weight_scale,
            cache_mode=self.cache_mode,
        )
```

## Hydra configs to reinstate

### Experiment configs (`configs/experiment/*_ar.yaml`)

Five files existed, one per backbone, all setting `data.rollout_steps: 6` and
wiring `model.rollout_steps: ${data.rollout_steps}`:

- `swh_1h_mlp_ar.yaml` — `data._target_: src.data.ARWWMDataModule.ARWWMDataModule`,
  `model._target_: src.model.ARWWMEmulator.ARWWMEmulator`, `model.window_len: 1`,
  MLP backbone.
- `swh_1h_lstm_ar.yaml` — `SequenceARWWMDataModule` / `ARWWMEmulator`,
  `model.window_len: ${data.seq_len}` (`data.seq_len: 24`), LSTMNet backbone.
- `swh_1h_multi-head-gat_ar.yaml` — `GraphARWWMDataModule` / `ARGraphWWMEmulator`,
  MultiHeadGAT backbone.
- `swh_1h_multi-head-gat-vn_ar.yaml` — `GraphARWWMDataModule` / `ARGraphWWMEmulator`,
  MultiHeadGATVN backbone.
- `swh_1h_multi-head-gat-lstm-vn_ar.yaml` —
  `CenterHistoryARWWMDataModule` / `ARCenterHistoryGraphWWMEmulator`,
  `data.seq_len: 24` (center node's LSTM history length),
  `model.window_len: ${data.seq_len}`, MultiHeadGATLSTMVN backbone.

All five otherwise mirror their non-AR counterpart (`swh_1h_mlp.yaml`,
`swh_1h_lstm.yaml`, etc.) for `data_dir`, day ranges, `features`/`targets`
(with `prev_sigWaveHeight` in `features`), scaling config, node/boundary CSVs, and
backbone hyperparameters, plus `tags: [..., "autoregressive"]` in `logger.wandb`.

Example (`swh_1h_mlp_ar.yaml`):

```yaml
# @package _global_

predict: false

experiment_name: "swh_1h_densenet_ar"

data:
  _target_: src.data.ARWWMDataModule.ARWWMDataModule
  data_dir: /work/ka1176/shared_data/2025-wwm-emulator-schism/processed_RUN04b_JUWELS_wpile.zarr
  train_days: [[1, 152], [214, 305]]
  val_days: [[153, 182], [306, 335]]
  test_days: [[183, 213], [336, 366]]
  features:
    - windSpeedX
    - windSpeedY
    - elevation
    - depthAverageVelX
    - depthAverageVelY
    - depth
    - prev_sigWaveHeight
  targets:
    - sigWaveHeight
  # Number of future steps (T) to roll out during training; the loss is
  # averaged across all T steps. Increase to encourage more stable
  # long-horizon rollouts; decrease if training is unstable or too slow.
  rollout_steps: 6
  sample_weight_strategy: histogram
  sample_weight_power: 0.2
  sample_weight_variable: delta_sigWaveHeight
  scaling_config: src/data/scaling/scale_2020_8months_hourly.yaml
  node_idx_csv: src/data/node_selection/node_idx_closest_center_10000.csv
  boundary_nodes_csv: src/data/node_selection/boundary_nodes.csv
  batch_size: 4096
  num_workers: 8

model:
  _target_: src.model.ARWWMEmulator.ARWWMEmulator

  target_names: ${data.targets}
  rollout_steps: ${data.rollout_steps}
  window_len: 1 # one time step consumed per rollout iteration (pointwise model)

  target_bins:
    sigWaveHeight: [0.5, 1.0, 2.0, 4.0]

  backbone:
    _target_: src.model.backbones.MLP.MLP
    input_dim: 7 # must match len(data.features)
    output_dim: 1 # must match len(data.targets)
    num_hidden_layers: 2 # number of hidden layers
    first_layer_neurons: 1024 # neurons in first hidden layer
    layer_scaling: halving # "constant" or "halving"
    dropout_rate: 0.03 # dropout probability (0.0 = no dropout)

  optimizer:
    lr: 0.0004

  output_constraints:
    sigWaveHeight: semipositive

trainer:
  devices: 1

logger:
  wandb:
    project: wwm-emulator-hourly-final
    tags: ["swh-1h", "densenet", "autoregressive"]
```

The other four experiment configs follow the identical shape — swap the `data`/
`model` `_target_`s and backbone block per the table above, and (for the graph
configs) add `graph_features: [SCHISM_hgrid_edge_nodes]`, `max_nodes`,
`edge_weight_scale`.

### CI/CD configs (`configs/cicd/cicd_train_ar_*.yaml`)

Four small/fast configs pointing at `tests/cicd_data/cicd.zarr` with
`rollout_steps: 3`, mirroring the non-AR `cicd_train_*.yaml` configs but with the AR
`_target_`s:

- `cicd_train_ar_mlp.yaml` — `ARWWMDataModule` / `ARWWMEmulator`, `window_len: 1`.
- `cicd_train_ar_lstm.yaml` — `SequenceARWWMDataModule` / `ARWWMEmulator`,
  `data.seq_len: 4`, `model.window_len: ${data.seq_len}`.
- `cicd_train_ar_multihead_gat_vn.yaml` — `GraphARWWMDataModule` /
  `ARGraphWWMEmulator`, MultiHeadGATVN backbone, `max_nodes: 20`.
- `cicd_train_ar_dual_path_gat_vn.yaml` — `CenterHistoryARWWMDataModule` /
  `ARCenterHistoryGraphWWMEmulator`, `data.seq_len: 2`, MultiHeadGATLSTMVN backbone,
  `max_nodes: 20`.

These were referenced by `tests/test_autoregressive.py::TestAutoregressiveTrainE2E`
(fast_dev_run E2E tests) — they were **not** run in `.gitlab-ci.yml`'s
`deep-learning-pipeline` job (only used inside pytest).

## Command-line entries removed from `scripts/submit_train.sh`

The following SLURM array `COMMANDS` entries launched AR sweeps over
`rollout_steps` and were removed:

```bash
"uv run python src/train.py experiment=swh_1h_mlp_ar trainer.devices=1 data.rollout_steps=1  logger.wandb.project=wwm-emulator-sweeps seed=25826"
"uv run python src/train.py experiment=swh_1h_mlp_ar trainer.devices=1 data.rollout_steps=3  logger.wandb.project=wwm-emulator-sweeps seed=25826"
"uv run python src/train.py experiment=swh_1h_mlp_ar trainer.devices=1 data.rollout_steps=6  logger.wandb.project=wwm-emulator-sweeps seed=25826"
"uv run python src/train.py experiment=swh_1h_mlp_ar trainer.devices=1 data.rollout_steps=24 logger.wandb.project=wwm-emulator-sweeps seed=25826"
"uv run python src/train.py experiment=swh_1h_lstm_ar trainer.devices=1 data.rollout_steps=1  logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
"uv run python src/train.py experiment=swh_1h_lstm_ar trainer.devices=1 data.rollout_steps=3  logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
"uv run python src/train.py experiment=swh_1h_lstm_ar trainer.devices=1 data.rollout_steps=6  logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
"uv run python src/train.py experiment=swh_1h_lstm_ar trainer.devices=1 data.rollout_steps=24 logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
"uv run python src/train.py experiment=swh_1h_multi-head-gat-vn_ar trainer.devices=1 data.rollout_steps=1 data.batch_size=64 logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
"uv run python src/train.py experiment=swh_1h_multi-head-gat-vn_ar trainer.devices=1 data.rollout_steps=6 data.batch_size=64 logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
"uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn_ar trainer.devices=1 data.rollout_steps=1 data.batch_size=64 logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
"uv run python src/train.py experiment=swh_1h_multi-head-gat-lstm-vn_ar trainer.devices=1 data.rollout_steps=6 data.batch_size=64 logger.wandb.project=wwm-emulator-sweeps seed=25826 trainer.max_epochs=5 data.node_idx_csv=src/data/node_selection/node_idx_closest_center_graph-tuning.csv"
```

## Tests removed (`tests/test_autoregressive.py`)

The removed test module covered:

- `TestBuildFeedbackMap` — moved to the shared dataset-utility tests since
  `build_feedback_map` is still used by `src/rollout.py` and `src/model/noise.py`
  (not deleted).
- `TestSimpleARWWMDataset` — rejects `rollout_steps < 1`, rejects `delta_*` targets,
  window shapes, length, and consecutive-target correctness.
- `TestARWWMDataModule` — fit stage builds AR dataset + `feedback_map`; test stage
  falls back to the plain dataset.
- `TestAutoregressiveMixin` / `_make_ar_emulator` helper — `rollout_steps=1` matches
  the plain `WWMEmulator.model_step`; feedback correctly writes rescaled predictions
  into the next step; loss averages over all T steps; gradients are blocked through
  the detached feedback path.
- `TestAutoregressiveTrainE2E` — full `src.train.train()` `fast_dev_run` runs for
  each of the four CI/CD AR configs, asserting `val/loss_step0` is logged.

## Related, still-supported feature: `prev_noise_std`

`model.prev_noise_std` (`src/model/noise.py`, `README.md`) remains in the codebase.
It is the ~1×-cost, non-autoregressive alternative: it perturbs `prev_*` inputs with
Gaussian noise (sized from the one-step validation RMSE) instead of rolling out
`rollout_steps` predictions, so it does not require any of the AR
datasets/datamodules/emulators documented above — only the (kept) `feedback_map`
mechanism to know which feature channels are `prev_<target>` and their scale.
