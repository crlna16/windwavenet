# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class CrossAttnVirtualNodeGCN(nn.Module):
    """LSTM + GCN backbone with cross-attention virtual node injection and collection.

    Extends the virtual-node pattern of ``VirtualNodeCenterLSTMGCNNet``
    by replacing uniform additive injection with **gated cross-attention**
    and uniform mean-pool collection with **attention-weighted aggregation**.

    **Stage 1 — Temporal encoding (LSTM):**
    An LSTM processes the center node's temporal history and produces a
    fixed-size embedding that initialises the virtual node.

    **Stage 2 — Cross-attention GCN loop:**
    For each processor layer:

      a. *Gated injection* — each real node computes a per-feature
         sigmoid gate from its own embedding and the virtual node,
         controlling how much temporal context it absorbs.  Coastal
         nodes exposed to open ocean may learn to absorb more temporal
         context than sheltered interior nodes::

             gate_i = sigmoid(W_g [x_i ; vn])
             x_i   += gate_i * W_p(vn)

      b. *GCN message passing* — standard ``GCNConv`` for spatial
         diffusion with distance-based edge weights.

      c. *Attention-weighted collection* — the virtual node emits a
         learned query that dot-product attends over all real nodes
         (keys), producing a weighted aggregation that replaces uniform
         mean pooling.  This lets the model focus on the most
         informative nodes when updating temporal state::

             alpha_i = softmax_per_graph( q_vn . k_i / sqrt(d) )
             vn     += MLP( sum_i alpha_i * x_i )

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
        """Initialize the CrossAttnVirtualNodeGCN network.

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

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self._scale = 1.0 / math.sqrt(hidden_dim)

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

        # Stage 2: per-layer modules
        self.convs = nn.ModuleList()
        self.inject_gates = nn.ModuleList()
        self.inject_projs = nn.ModuleList()
        self.collect_queries = nn.ModuleList()
        self.collect_keys = nn.ModuleList()
        self.vn_mlps = nn.ModuleList()

        for _ in range(num_gcn_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

            # Gated injection: gate from [x; vn], projection of vn
            self.inject_gates.append(nn.Linear(2 * hidden_dim, hidden_dim))
            self.inject_projs.append(nn.Linear(hidden_dim, hidden_dim))

            # Attention-weighted collection: query from vn, key from nodes
            self.collect_queries.append(nn.Linear(hidden_dim, hidden_dim))
            self.collect_keys.append(nn.Linear(hidden_dim, hidden_dim))

            # Residual MLP for virtual node update
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
        """Forward pass: encode → cross-attention GCN loop → decode.

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

        # --- Stage 2: GCN layers with cross-attention virtual node ---
        for conv, gate_lin, proj_lin, q_lin, k_lin, vn_mlp in zip(
            self.convs,
            self.inject_gates,
            self.inject_projs,
            self.collect_queries,
            self.collect_keys,
            self.vn_mlps,
        ):
            # (a) Gated cross-attention injection
            vn_broadcast = vn[batch]  # (total_nodes, hidden_dim)
            gate = torch.sigmoid(gate_lin(torch.cat([x, vn_broadcast], dim=-1)))
            x = x + gate * proj_lin(vn_broadcast)

            # (b) GCN message passing
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # (c) Attention-weighted collection
            q = q_lin(vn)[batch]  # (total_nodes, hidden_dim)
            k = k_lin(x)  # (total_nodes, hidden_dim)
            scores = (q * k).sum(dim=-1) * self._scale  # (total_nodes,)
            attn = pyg_softmax(scores, batch)  # (total_nodes,) normalised per graph
            vn_agg = global_add_pool(x * attn.unsqueeze(-1), batch)  # (batch_size, hidden_dim)
            vn = vn + vn_mlp(vn_agg)  # residual update

        # --- Stage 3: decode ---
        return self.fc(x)
