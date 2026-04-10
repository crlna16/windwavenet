# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNFullNet(nn.Module):
    """GCN backbone that predicts a value for every node in the subgraph.

    Applies multiple GCNConv layers, then projects each node's embedding
    to the output dimension.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        """Initialize the GCN full-graph prediction network.

        Args:
            input_dim: Number of input features per node.
            output_dim: Number of output features per node.
            hidden_dim: Number of hidden units in GCN layers (default: 128).
            num_layers: Number of GCN layers (default: 3).
            dropout: Dropout probability after each hidden layer (default: 0.0).
        """
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=False))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            data: PyG Batch object with attributes x, edge_index, edge_weight.

        Returns:
            Predictions of shape (total_nodes_in_batch, output_dim) for all nodes.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.fc(x)
