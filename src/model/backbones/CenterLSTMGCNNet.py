# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CenterLSTMGCNNet(nn.Module):
    """Two-stage backbone: LSTM on center node history, then GCN on all nodes.

    **Stage 1 — Temporal encoding (LSTM):**
    An LSTM processes the center node's temporal history
    ``data.x_center_seq`` of shape ``(batch_size, seq_len, input_dim)``
    and produces a fixed-size embedding ``h_T`` of shape
    ``(batch_size, lstm_hidden_dim)``.

    **Stage 2 — Spatial propagation (GCN):**
    The temporal embedding is scattered to the center node position in
    the flat batched graph.  Non-center nodes receive a zero vector.
    The augmented feature ``[current_features ∥ temporal_embed]`` of
    dimension ``input_dim + lstm_hidden_dim`` is then processed by
    stacked GCNConv layers that propagate the temporal context spatially.

    Finally a linear head projects each node's embedding to the output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        num_gcn_layers: int = 3,
        dropout: float = 0.0,
    ):
        """Initialize the CenterLSTMGCN network.

        Args:
            input_dim: Number of input features per node (current time step).
            output_dim: Number of output features per node.
            hidden_dim: Hidden dimension of GCN layers.
            lstm_hidden_dim: Hidden dimension of the LSTM.
            num_lstm_layers: Number of stacked LSTM layers.
            num_gcn_layers: Number of GCN message-passing layers.
            dropout: Dropout probability after each GCN hidden layer.
        """
        super().__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.dropout = dropout

        # Stage 1: LSTM for center node temporal encoding
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Stage 2: GCN layers on augmented features
        gcn_input_dim = input_dim + lstm_hidden_dim
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(gcn_input_dim, hidden_dim, add_self_loops=False))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the two-stage network.

        Args:
            data: PyG Batch object with attributes:
                - ``x``: ``(total_nodes, input_dim)`` current-time features.
                - ``x_center_seq``: ``(batch_size, seq_len, input_dim)``
                  temporal history for each graph's center node.
                - ``center_idx``: ``(batch_size,)`` local index of the center
                  node within each sub-graph.
                - ``edge_index``: ``(2, n_edges)`` graph connectivity.
                - ``edge_weight``: optional ``(n_edges,)`` edge weights.
                - ``ptr``: ``(batch_size + 1,)`` batch pointer from PyG.

        Returns:
            Predictions of shape ``(total_nodes, output_dim)``.
        """
        x = data.x  # (total_nodes, input_dim)
        x_center_seq = data.x_center_seq  # (batch_size, seq_len, input_dim)
        center_idx = data.center_idx  # (batch_size,)
        ptr = data.ptr  # (batch_size + 1,)
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        # --- Stage 1: encode center node history ---
        # LSTM output: last hidden state at final time step
        lstm_out, _ = self.lstm(x_center_seq)  # (batch_size, seq_len, lstm_hidden_dim)
        h_T = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)

        # Scatter h_T to center node positions in the flat batched graph.
        # In PyG batching, node i of graph g has global index ptr[g] + i.
        total_nodes = x.size(0)
        temporal_embed = x.new_zeros(total_nodes, self.lstm_hidden_dim)
        global_center_idx = ptr[:-1] + center_idx  # (batch_size,)
        temporal_embed[global_center_idx] = h_T

        # --- Stage 2: GCN on augmented features ---
        x = torch.cat([x, temporal_embed], dim=-1)  # (total_nodes, input_dim + lstm_hidden_dim)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.fc(x)
