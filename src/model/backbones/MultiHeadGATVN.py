# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

from .MultiHeadGATLSTMVN import GaussianRBF


class MultiHeadGATVN(nn.Module):
    """Dual-path backbone: local GAT over mesh edges + subgraph-wide virtual node.

    Combines the multi-head GAT spatial reasoning of :class:`MultiHeadGAT`
    with the virtual-node communication mechanism of
    :class:`MultiHeadGATLSTMVN`, using only the current time step's features
    (no temporal history or LSTM).

    **Local path** — Multi-head GATv2 attention over the mesh edge set.
    Attention is strictly one-hop (direct mesh neighbours plus self), so the
    local receptive field grows with ``num_layers``.  Edge lengths are encoded
    as inverse-distance weights expanded through Gaussian RBFs and passed as
    edge features to GATv2Conv; this tells attention how much physical space
    each hop covers on the unstructured mesh, whose edge lengths vary by
    orders of magnitude between coastal and offshore regions.  The encoding
    is only informative if ``edge_weight_scale`` is comparable to typical
    edge lengths (see :class:`MultiHeadGAT` and
    :func:`src.data.graph_utils.compute_edge_weights`).

    **Global path** — A virtual node (VN) provides subgraph-wide context in a
    single step, bypassing the one-hop limit of the local path.  Its scope is
    the sampled subgraph around the center node, not the full domain.  The VN
    is initialized from the mean pooling of encoded node features.  At each
    layer:
      - *Gated injection*: each node receives subgraph-wide context from the
        VN through a learned sigmoid gate.
      - *Attention-weighted collection*: the VN queries attention over all
        subgraph nodes and aggregates their embeddings.
      - *Recurrent update*: a GRU cell updates the VN state across layers.

    **Fusion** — A learned gate per node combines the local and global
    representations before the next layer.

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
        """Initialize the MultiHeadGATVN network.

        Args:
            input_dim: Number of input features per node (current time step).
            output_dim: Number of output features per node.
            hidden_dim: Hidden dimension shared across all components.
                Must be divisible by ``heads``.
            num_layers: Number of dual-path processing layers (sets the local
                receptive field in hops).
            heads: Number of attention heads in the local GAT path.
            n_rbf: Number of Gaussian RBF centres for edge-length encoding.
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

        # ---- Node encoder ----
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # ---- Virtual node initializer (from mean-pooled node embeddings) ----
        self.vn_init = nn.Linear(hidden_dim, hidden_dim)

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

        for i in range(num_layers):
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

            # Global: attention-weighted collection and recurrent update
            # Only needed for layers that feed a subsequent injection step.
            if i < num_layers - 1:
                self.collect_queries.append(nn.Linear(hidden_dim, hidden_dim))
                self.collect_keys.append(nn.Linear(hidden_dim, hidden_dim))
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
                - ``edge_index``: ``(2, n_edges)`` graph connectivity.
                - ``edge_weight``: ``(n_edges,)`` inverse-distance weights
                  in (0, 1].
                - ``batch``: ``(total_nodes,)`` graph membership.

        Returns:
            Predictions of shape ``(total_nodes, output_dim)``.
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        batch = data.batch

        # ---- Edge-length encoding ----
        if edge_weight is not None:
            edge_attr = self.rbf(edge_weight)  # (n_edges, n_rbf)
        else:
            edge_attr = None

        # ---- Stage 1: Encode node features ----
        x = self.node_encoder(x)  # (total_nodes, hidden_dim)

        # ---- Virtual node initialisation: mean pool over current embeddings ----
        vn = self.vn_init(global_mean_pool(x, batch))  # (batch_size, hidden_dim)

        # ---- Stage 2: Dual-path processing ----
        for i, (gat, norm, gate_lin, proj_lin, fuse_lin) in enumerate(
            zip(
                self.gat_convs,
                self.gat_norms,
                self.inject_gates,
                self.inject_projs,
                self.fusion_gates,
            )
        ):
            x_residual = x

            # --- Local path: one-hop GAT with edge-length-conditioned attention ---
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

            # --- Global path: attention-weighted collection + recurrent update ---
            # Skipped in the last layer: the updated VN would never be consumed.
            if i < len(self.collect_queries):
                q = self.collect_queries[i](vn)[batch]  # (total_nodes, hidden_dim)
                k = self.collect_keys[i](x)  # (total_nodes, hidden_dim)
                scores = (q * k).sum(dim=-1) * self._scale
                attn = pyg_softmax(scores, batch)
                vn_agg = global_add_pool(x * attn.unsqueeze(-1), batch)  # (batch_size, hidden_dim)
                vn = self.vn_grus[i](vn_agg, vn)

        # ---- Stage 3: Decode ----
        return self.node_decoder(x)
