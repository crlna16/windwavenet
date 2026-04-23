# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class VirtualNodeCenterLSTMGCNNet(nn.Module):
    """LSTM + GCN backbone with virtual-node temporal context distribution.

    Replaces the "scatter to center only" strategy of
    ``CenterLSTMGCNNet`` with a virtual node that injects the LSTM
    temporal embedding into **all** graph nodes at every GCN layer.

    **Stage 1 — Temporal encoding (LSTM):**
    An LSTM processes the center node's temporal history
    ``data.x_center_seq`` and produces a fixed-size embedding ``h_T``.

    **Stage 2 — Virtual-node GCN loop:**
    ``h_T`` is projected to initialise a per-graph *virtual node
    embedding* ``vn``.  Before each GCN layer the virtual embedding is
    added to every real node (injection), and after each layer the real
    node embeddings are aggregated back into the virtual node via
    ``global_mean_pool`` with a residual MLP update (collection).

    This gives every node — regardless of graph distance from the
    center — direct access to the temporal context at every layer,
    while still allowing the GCN to refine spatial relationships.

    **Stage 3 — Decoder:**
    A linear head projects each node's final embedding to the output.
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
        """Initialize the VirtualNodeCenterLSTMGCN network.

        Args:
            input_dim: Number of input features per node (current time step).
            output_dim: Number of output features per node.
            hidden_dim: Hidden dimension of GCN layers and virtual node.
            lstm_hidden_dim: Hidden dimension of the LSTM.
            num_lstm_layers: Number of stacked LSTM layers.
            num_gcn_layers: Number of GCN message-passing layers.
            dropout: Dropout probability after each GCN hidden layer.
        """
        super().__init__()

        self.dropout = dropout

        # Stage 1: LSTM for center node temporal encoding
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Project LSTM output to hidden_dim for virtual node initialisation
        self.vn_init = nn.Linear(lstm_hidden_dim, hidden_dim)

        # Node encoder: lift raw features into hidden_dim
        self.node_encoder = nn.Linear(input_dim, hidden_dim)

        # Stage 2: GCN layers + per-layer virtual node update MLPs
        self.convs = nn.ModuleList()
        self.vn_mlps = nn.ModuleList()
        for i in range(num_gcn_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            # Residual MLP: aggregated node info → virtual node update
            self.vn_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        # Stage 3: Decoder
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Forward pass through the virtual-node network.

        Args:
            data: PyG Batch object with attributes:
                - ``x``: ``(total_nodes, input_dim)`` current-time features.
                - ``x_center_seq``: ``(batch_size, seq_len, input_dim)``
                  temporal history for each graph's center node.
                - ``edge_index``: ``(2, n_edges)`` graph connectivity.
                - ``edge_weight``: optional ``(n_edges,)`` edge weights.
                - ``batch``: ``(total_nodes,)`` graph membership from PyG.

        Returns:
            Predictions of shape ``(total_nodes, output_dim)``.
        """
        x = data.x  # (total_nodes, input_dim)
        x_center_seq = data.x_center_seq  # (batch_size, seq_len, input_dim)
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        batch = data.batch  # (total_nodes,)

        # --- Stage 1: encode center node history ---
        lstm_out, _ = self.lstm(x_center_seq)  # (batch_size, seq_len, lstm_hidden_dim)
        h_T = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)

        # Initialise virtual node embedding from temporal encoding
        vn = self.vn_init(h_T)  # (batch_size, hidden_dim)

        # Encode real node features into hidden_dim
        x = self.node_encoder(x)  # (total_nodes, hidden_dim)

        # --- Stage 2: GCN layers with virtual node injection/collection ---
        for conv, vn_mlp in zip(self.convs, self.vn_mlps):
            # Inject: broadcast virtual node embedding to all real nodes
            x = x + vn[batch]

            # Message passing
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Collect: aggregate real node embeddings back into virtual node
            vn_agg = global_mean_pool(x, batch)  # (batch_size, hidden_dim)
            vn = vn + vn_mlp(vn_agg)  # residual update

        # --- Stage 3: decode ---
        return self.fc(x)
