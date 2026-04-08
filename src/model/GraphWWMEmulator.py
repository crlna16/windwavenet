# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import Dict, Tuple

import torch

from src.model.WWMEmulator import WWMEmulator


class GraphWWMEmulator(WWMEmulator):
    """Lightning Module for graph-based Wave Wind Model emulation.

    Extends WWMEmulator to handle PyTorch Geometric batch objects instead of
    plain tensor tuples. The backbone receives the full PyG Batch and is
    responsible for extracting x, edge_index, etc.
    """

    def forward(self, data) -> torch.Tensor:
        """Forward pass — delegates the PyG Batch to the backbone.

        Args:
            data: PyG Batch object.

        Returns:
            Model predictions.
        """
        return self.backbone(data)

    def model_step(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Perform a single model step on a PyG batch.

        Args:
            batch: PyG Batch object with attributes x, edge_index, y, etc.
                   If the DataLoader returns a list, the first element is used.

        Returns:
            A tuple of (total_loss, per_variable_losses, predictions, targets).
        """
        # Some PyG DataLoader versions wrap the batch in a list
        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        y = batch.y
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        preds = self.forward(batch)

        per_sample_loss = self.criterion(preds, y)

        per_variable_losses = {}
        for i, name in enumerate(self.target_names):
            per_variable_losses[name] = per_sample_loss[:, i].mean()

        total_loss = sum(per_variable_losses[name] * self.loss_weights[name] for name in self.target_names)

        return total_loss, per_variable_losses, preds, y
