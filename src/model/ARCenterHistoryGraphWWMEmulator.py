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
        rollout_detach: bool = True,
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
            rollout_detach: If True (default), stop gradients through fed-back predictions
                (cheaper, more stable). If False, backpropagate through the full rollout (BPTT).
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
        self._init_autoregressive(rollout_steps=rollout_steps, rollout_detach=rollout_detach)
        self.save_hyperparameters("rollout_steps", "window_len", "rollout_detach", logger=False)

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
