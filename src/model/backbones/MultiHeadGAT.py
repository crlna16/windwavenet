# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .MultiHeadGATLSTMVN import GaussianRBF


class MultiHeadGAT(nn.Module):
    """Spatial graph backbone: multi-head GAT over mesh edges with edge-length encoding.

    Multi-head GATv2 attention over the mesh edge set.  Attention is strictly
    one-hop: in each layer a node attends only to its direct mesh neighbours
    (and itself), so the spatial receptive field grows with ``num_layers``.

    **Edge encoding** — The SCHISM mesh is unstructured: edge lengths vary by
    orders of magnitude between refined coastal areas and the open sea, so a
    hop along an edge covers very different physical distances in different
    parts of the domain.  Inverse-distance edge weights
    (``w = 1 / (1 + length / scale)``, see
    :func:`src.data.graph_utils.compute_edge_weights`) are expanded through
    Gaussian RBFs and passed as edge features to GATv2Conv, which projects
    them to per-head attention biases.  This tells each attention head how
    much physical space a hop covers, so it can weight neighbours by physical
    proximity rather than by hop count alone.  Note that the encoding is only
    informative if ``edge_weight_scale`` (data config) is comparable to
    typical mesh edge lengths, so that the weights spread over (0, 1); if the
    scale is much larger than the edges, all weights cluster near 1 and the
    RBF features carry no signal.

    **Decoder** — Linear head projects node embeddings to predictions.
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
            num_layers: Number of GAT layers (sets the receptive field in hops).
            heads: Number of attention heads.
            n_rbf: Number of Gaussian RBF centres for edge-length encoding.
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

        # ---- Stage 2: GAT layers ----
        head_dim = hidden_dim // heads

        # GATv2 with edge-length-conditioned attention
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

        # ---- Edge-length encoding ----
        if edge_weight is not None:
            edge_attr = self.rbf(edge_weight)  # (n_edges, n_rbf)
        else:
            edge_attr = None

        # ---- Stage 1: Encode node features ----
        x = self.node_encoder(x)  # (total_nodes, hidden_dim)

        # ---- Stage 2: GAT layers (one-hop attention each) ----
        for gat, norm in zip(
            self.gat_convs,
            self.gat_norms,
        ):
            x_residual = x

            x = gat(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = norm(x + x_residual)  # residual + LayerNorm

        # ---- Stage 3: Decode ----
        return self.node_decoder(x)
