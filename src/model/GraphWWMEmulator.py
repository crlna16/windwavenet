# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import Dict, Tuple

import torch

from src.model.WWMEmulator import WWMEmulator


class GraphWWMEmulator(WWMEmulator):
    """Lightning Module for graph-based Wind Wave Model emulation.

    Extends WWMEmulator to handle PyTorch Geometric batch objects instead of
    plain tensor tuples. The backbone receives the full PyG Batch and is
    responsible for extracting x, edge_index, etc.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        """Inject ``prev_*`` noise into the training batch's node features.

        Overrides the tuple-based hook in ``WWMEmulator``: graph batches carry
        the features on ``batch.x`` (and, for the center-history backbone, on
        ``batch.x_center_seq``). Validation, test and predict batches are left
        untouched.

        Args:
            batch: PyG Batch object, possibly wrapped in a list.
            dataloader_idx: Index of the dataloader the batch came from.

        Returns:
            The batch, with node features perturbed during training.
        """
        if self._prev_noise is None:
            return batch

        graph = batch[0] if isinstance(batch, (list, tuple)) else batch
        graph.x = self._add_prev_noise(graph.x)
        if getattr(graph, "x_center_seq", None) is not None:
            graph.x_center_seq = self._add_prev_noise(graph.x_center_seq)
        return batch

    def forward(self, data) -> torch.Tensor:
        """Forward pass — delegates the PyG Batch to the backbone.

        Args:
            data: PyG Batch object.

        Returns:
            Model predictions.
        """
        preds = self.backbone(data)
        return self._apply_output_constraints(preds)

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

        # Per-node z-score sample weights: (total_nodes, 1) broadcasts over
        # the target dimension of the per-node loss.
        sample_weight = getattr(batch, "sample_weight", None)
        if sample_weight is not None:
            per_sample_loss = per_sample_loss * sample_weight

        per_variable_losses = {}
        for i, name in enumerate(self.target_names):
            per_variable_losses[name] = per_sample_loss[:, i].mean()

        total_loss = sum(per_variable_losses[name] * self.loss_weights[name] for name in self.target_names)

        return total_loss, per_variable_losses, preds, y
