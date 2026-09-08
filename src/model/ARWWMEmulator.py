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
        rollout_detach: bool = True,
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
