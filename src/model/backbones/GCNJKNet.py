# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge


class GCNJKNet(nn.Module):
    """GCN backbone with JumpingKnowledge skip connections for full-graph prediction.

    Applies multiple GCNConv layers and combines their representations via
    JumpingKnowledge before projecting to the output.  By retaining every
    layer's output, JumpingKnowledge counters oversmoothing: the final linear
    layer can attend to both local (shallow) and long-range (deep) embeddings.

    Supported jk_mode values:

    * ``'cat'``  — concatenate all layer outputs; fc input dimension is
      ``num_layers * hidden_dim``.  Most expressive.
    * ``'max'``  — element-wise max over layer outputs; fc input dimension is
      ``hidden_dim``.  Cheapest.
    * ``'lstm'`` — LSTM attention over the layer sequence; fc input dimension
      is ``hidden_dim``.  Most parameters inside the JK module.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 5,
        dropout: float = 0.0,
        jk_mode: str = "cat",
    ):
        """Initialize the GCN-JK full-graph prediction network.

        Args:
            input_dim: Number of input features per node.
            output_dim: Number of output features per node.
            hidden_dim: Number of hidden units in every GCN layer.
            num_layers: Number of GCN layers (must be >= 1).
            dropout: Dropout probability applied after each GCN layer.
            jk_mode: JumpingKnowledge aggregation mode: 'cat', 'max', or 'lstm'.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if jk_mode not in ("cat", "max", "lstm"):
            raise ValueError(f"jk_mode must be 'cat', 'max', or 'lstm', got '{jk_mode}'")

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_dim, num_layers=num_layers)

        jk_out_dim = num_layers * hidden_dim if jk_mode == "cat" else hidden_dim
        self.fc = nn.Linear(jk_out_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            data: PyG Batch object with attributes ``x``, ``edge_index``, and
                optionally ``edge_weight``.

        Returns:
            Predictions of shape ``(total_nodes_in_batch, output_dim)``.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        x = self.jk(layer_outputs)

        return self.fc(x)
