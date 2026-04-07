# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNCenterNet(nn.Module):
    """GCN backbone that predicts a scalar value for the center node only.

    Applies multiple GCNConv layers to propagate information across the graph,
    then extracts the center node embedding and projects it to the output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        """Initialize the GCN center-prediction network.

        Args:
            input_dim: Number of input features per node.
            output_dim: Number of output features (e.g., 1 for scalar prediction).
            hidden_dim: Number of hidden units in GCN layers (default: 128).
            num_layers: Number of GCN layers (default: 3).
            dropout: Dropout probability after each hidden layer (default: 0.0).
        """
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            data: PyG Batch object with attributes x, edge_index, center_idx, ptr.

        Returns:
            Predictions of shape (batch_size, output_dim) for center nodes only.
        """
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Extract center node embeddings using batched graph offsets
        center_indices = data.ptr[:-1] + data.center_idx
        x = x[center_indices]

        return self.fc(x)
