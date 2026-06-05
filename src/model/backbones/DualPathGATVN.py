# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion for scalar edge features.

    Expands a scalar input (e.g. edge_weight in [0, 1]) into a vector of
    Gaussian basis responses.  centers are fixed and uniformly spaced; widths
    are fixed such that adjacent bases overlap at half-maximum.

    This gives downstream attention heads a rich representation of distance
    from which they can learn arbitrary (including non-monotonic) spatial
    attention profiles.
    """

    def __init__(self, n_rbf: int = 16, low: float = 0.0, high: float = 1.0):
        super().__init__()
        self.centers = torch.linspace(low, high, n_rbf)
        # Width so that adjacent bases cross at ~0.5 height
        self.gamma = 0.5 / ((high - low) / n_rbf) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand scalar x to RBF features.

        Args:
            x: (n_edges,) or (n_edges, 1) scalar values.

        Returns:
            (n_edges, n_rbf) Gaussian basis expansion.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return torch.exp(-self.gamma * (x - self.centers) ** 2)


class DualPathGATVN(nn.Module):
    """Dual-path backbone: local distance-conditioned GAT + global virtual node.

    Separates local spatial reasoning from global temporal context:

    **Local path** — Multi-head GATv2 attention over the mesh edge set with
    distance-conditioned bias.  Edge weights (inverse-distance) are expanded
    through Gaussian RBFs and passed as edge features to GATv2Conv, which
    internally projects them to per-head attention biases.  This allows each
    attention head to learn its own characteristic length scale — some heads
    can focus on immediate neighbors (short-range wave dynamics) while others
    attend to more distant nodes (swell propagation).

    **Global path** — Cross-attention virtual node initialized from the
    center node's LSTM temporal encoding.  At each layer:
      - *Gated injection*: each node receives temporal context from the VN
        through a learned sigmoid gate.
      - *Attention-weighted collection*: the VN queries attention over all
        nodes and aggregates their embeddings.
      - *Recurrent update*: a GRU cell updates the VN state, giving the VN
        memory across layers (unlike purely residual updates).

    **Fusion** — A learned gate per node combines the local and global
    representations before the next layer.

    **Decoder** — Linear head projects fused embeddings to predictions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        n_rbf: int = 16,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize the DualPathGATVN network.

        Args:
            input_dim: Number of input features per node (current time step).
            output_dim: Number of output features per node.
            hidden_dim: Hidden dimension shared across all components.
                Must be divisible by ``heads``.
            lstm_hidden_dim: Hidden dimension of the LSTM.
            num_lstm_layers: Number of stacked LSTM layers.
            num_layers: Number of dual-path processing layers.
            heads: Number of attention heads in the local GAT path.
            n_rbf: Number of Gaussian RBF centers for distance encoding.
            dropout: Dropout probability after each processing layer.
            attn_dropout: Dropout on GAT attention coefficients.
        """
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self._scale = 1.0 / math.sqrt(hidden_dim)

        # ---- Distance encoder ----
        self.rbf = GaussianRBF(n_rbf=n_rbf, low=0.0, high=1.0)

        # ---- Stage 1: Temporal encoding (LSTM) ----
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.vn_init = nn.Linear(lstm_hidden_dim, hidden_dim)

        # ---- Node encoder ----
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # ---- Stage 2: Dual-path processing layers ----
        head_dim = hidden_dim // heads

        # Local path: GATv2 with distance-conditioned attention
        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        # Global path: VN injection, collection, recurrent update
        self.inject_gates = nn.ModuleList()
        self.inject_projs = nn.ModuleList()
        self.collect_queries = nn.ModuleList()
        self.collect_keys = nn.ModuleList()
        self.vn_grus = nn.ModuleList()

        # Fusion gates
        self.fusion_gates = nn.ModuleList()

        for _ in range(num_layers):
            # Local: GATv2 with RBF edge features
            self.gat_convs.append(
                GATv2Conv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    concat=True,
                    add_self_loops=True,
                    edge_dim=n_rbf,
                    dropout=attn_dropout,
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

            # Global: gated injection
            self.inject_gates.append(nn.Linear(2 * hidden_dim, hidden_dim))
            self.inject_projs.append(nn.Linear(hidden_dim, hidden_dim))

            # Global: attention-weighted collection
            self.collect_queries.append(nn.Linear(hidden_dim, hidden_dim))
            self.collect_keys.append(nn.Linear(hidden_dim, hidden_dim))

            # Global: recurrent VN update (GRU cell)
            self.vn_grus.append(nn.GRUCell(hidden_dim, hidden_dim))

            # Fusion: gate to combine local + global per node
            self.fusion_gates.append(nn.Linear(2 * hidden_dim, hidden_dim))

        # ---- Stage 3: Decoder ----
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass: encode → dual-path processing → decode.

        Args:
            data: PyG Batch object with attributes:
                - ``x``: ``(total_nodes, input_dim)`` current-time features.
                - ``x_center_seq``: ``(batch_size, seq_len, input_dim)``
                  temporal history for each graph's center node.
                - ``edge_index``: ``(2, n_edges)`` graph connectivity.
                - ``edge_weight``: ``(n_edges,)`` inverse-distance weights
                  in (0, 1].
                - ``batch``: ``(total_nodes,)`` graph membership.

        Returns:
            Predictions of shape ``(total_nodes, output_dim)``.
        """
        x = data.x
        x_center_seq = data.x_center_seq
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        batch = data.batch

        # ---- Distance encoding ----
        if edge_weight is not None:
            edge_attr = self.rbf(edge_weight)  # (n_edges, n_rbf)
        else:
            edge_attr = None

        # ---- Stage 1: Temporal encoding ----
        lstm_out, _ = self.lstm(x_center_seq)
        h_T = lstm_out[:, -1, :]
        vn = self.vn_init(h_T)  # (batch_size, hidden_dim)

        # Encode node features
        x = self.node_encoder(x)  # (total_nodes, hidden_dim)

        # ---- Stage 2: Dual-path processing ----
        for i, (gat, norm, gate_lin, proj_lin, q_lin, k_lin, gru, fuse_lin) in enumerate(
            zip(
                self.gat_convs,
                self.gat_norms,
                self.inject_gates,
                self.inject_projs,
                self.collect_queries,
                self.collect_keys,
                self.vn_grus,
                self.fusion_gates,
            )
        ):
            x_residual = x

            # --- Local path: distance-conditioned GAT ---
            x_local = gat(x, edge_index, edge_attr=edge_attr)
            x_local = F.elu(x_local)
            x_local = F.dropout(x_local, p=self.dropout, training=self.training)
            x_local = norm(x_local + x_residual)  # residual + LayerNorm

            # --- Global path: VN injection ---
            vn_broadcast = vn[batch]
            gate = torch.sigmoid(gate_lin(torch.cat([x, vn_broadcast], dim=-1)))
            x_global = x + gate * proj_lin(vn_broadcast)

            # --- Fusion: gated combination of local + global ---
            fusion_input = torch.cat([x_local, x_global], dim=-1)
            alpha = torch.sigmoid(fuse_lin(fusion_input))
            x = alpha * x_local + (1.0 - alpha) * x_global

            # --- Global path: attention-weighted collection ---
            q = q_lin(vn)[batch]  # (total_nodes, hidden_dim)
            k = k_lin(x)  # (total_nodes, hidden_dim)
            scores = (q * k).sum(dim=-1) * self._scale
            attn = pyg_softmax(scores, batch)
            vn_agg = global_add_pool(x * attn.unsqueeze(-1), batch)  # (batch_size, hidden_dim)

            # --- Global path: recurrent VN update ---
            vn = gru(vn_agg, vn)

        # ---- Stage 3: Decode ----
        return self.node_decoder(x)
