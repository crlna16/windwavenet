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
