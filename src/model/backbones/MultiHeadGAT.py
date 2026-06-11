# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .DualPathGATVN import GaussianRBF


class MultiHeadGAT(nn.Module):
    """Spatial graph backbone: local distance-conditioned GAT.

    **Local path** — Multi-head GATv2 attention over the mesh edge set with
    distance-conditioned bias.  Edge weights (inverse-distance) are expanded
    through Gaussian RBFs and passed as edge features to GATv2Conv, which
    internally projects them to per-head attention biases.  This allows each
    attention head to learn its own characteristic length scale — some heads
    can focus on immediate neighbors (short-range wave dynamics) while others
    attend to more distant nodes (swell propagation).

    **Decoder** — Linear head projects fused embeddings to predictions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        n_rbf: int = 16,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize the MultiHeadGAT network.

        Args:
            input_dim: Number of input features per node (current time step).
            output_dim: Number of output features per node.
            hidden_dim: Hidden dimension shared across all components.
                Must be divisible by ``heads``.
            num_layers: Number of dual-path processing layers.
            heads: Number of attention heads in the local GAT path.
            n_rbf: Number of Gaussian RBF centres for distance encoding.
            dropout: Dropout probability after each processing layer.
            attn_dropout: Dropout on GAT attention coefficients.
        """
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # ---- Distance encoder ----
        self.rbf = GaussianRBF(n_rbf=n_rbf, low=0.0, high=1.0)

        # ---- Node encoder ----
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # ---- Stage 2: Processing layers ----
        head_dim = hidden_dim // heads

        # Local path: GATv2 with distance-conditioned attention
        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

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

        # ---- Stage 3: Decoder ----
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass: encode → processing → decode.

        Args:
            data: PyG Batch object with attributes:
                - ``x``: ``(total_nodes, input_dim)`` current-time features.
                - ``edge_index``: ``(2, n_edges)`` graph connectivity.
                - ``edge_weight``: ``(n_edges,)`` inverse-distance weights
                  in (0, 1].

        Returns:
            Predictions of shape ``(total_nodes, output_dim)``.
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        # ---- Distance encoding ----
        if edge_weight is not None:
            edge_attr = self.rbf(edge_weight)  # (n_edges, n_rbf)
        else:
            edge_attr = None

        # ---- Stage 1: Encode node features ----
        x = self.node_encoder(x)  # (total_nodes, hidden_dim)

        # ---- Stage 2: Dual-path processing ----
        for gat, norm in zip(
            self.gat_convs,
            self.gat_norms,
        ):
            x_residual = x

            # --- Local path: distance-conditioned GAT ---
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = norm(x + x_residual)  # residual + LayerNorm

        # ---- Stage 3: Decode ----
        return self.node_decoder(x)
