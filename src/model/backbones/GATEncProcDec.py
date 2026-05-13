# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


def _build_mlp(dims: List[int]) -> nn.Sequential:
    """Build a simple MLP from a list of layer dimensions.

    Each pair of consecutive dimensions produces a Linear + ReLU block,
    except after the final Linear where no activation is applied.

    Args:
        dims: List of dimensions, e.g. [input, hidden, output].

    Returns:
        nn.Sequential module.
    """
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GATEncProcDec(nn.Module):
    """Encoder-Processor-Decoder backbone with GATv2 attention for full-graph prediction.

    Follows the Encoder-Processor-Decoder pattern (Pfaff et al., 2021) but
    replaces isotropic ``GCNConv`` message passing with multi-head
    ``GATv2Conv`` attention (Brody et al., 2022) in the processor.

    GATv2 computes *dynamic* attention: for each edge the attention score
    depends on both the source and target node features, enabling the model
    to learn which neighbours are most relevant in a given context.  This
    works naturally with varying graph topologies (dense vs. sparse) because
    softmax normalisation is per-node.

    **Encoder** — a node-local MLP that lifts input features into a uniform
    latent space (no message passing).

    **Processor** — a stack of multi-head ``GATv2Conv`` layers that perform
    attention-based message passing in latent space.  Optional residual
    connections add each layer's input to its output.

    **Decoder** — a node-local MLP that maps latent embeddings back to the
    physical output space (no message passing).

    When ``use_edge_features=True``, the existing distance-based edge weights
    are passed as edge attributes to ``GATv2Conv``, which linearly projects
    them and adds the result to the attention logits.  This combines learned
    feature-based attention with geometric distance information.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        encoder_layers: int = 2,
        num_mp_layers: int = 5,
        decoder_layers: int = 2,
        dropout: float = 0.0,
        use_residual: bool = True,
        heads: int = 4,
        use_edge_features: bool = True,
        attn_dropout: Optional[float] = None,
    ):
        """Initialize the GAT Encoder-Processor-Decoder network.

        Args:
            input_dim: Number of input features per node.
            output_dim: Number of output features per node.
            hidden_dim: Latent dimension shared across encoder, processor,
                and decoder.  Must be divisible by ``heads``.
            encoder_layers: Number of layers in the encoder MLP (>= 1).
            num_mp_layers: Number of GATv2Conv attention layers in the
                processor (>= 1).
            decoder_layers: Number of layers in the decoder MLP (>= 1).
            dropout: Dropout probability applied after each processor layer.
            use_residual: If True, add a residual (skip) connection around
                each processor attention layer.
            heads: Number of attention heads per processor layer.
            use_edge_features: If True, pass edge weights as edge attributes
                to GATv2Conv so that distance information biases attention.
            attn_dropout: Dropout probability on attention coefficients.
                Defaults to ``dropout`` if not specified.
        """
        super().__init__()

        if encoder_layers < 1:
            raise ValueError(f"encoder_layers must be >= 1, got {encoder_layers}")
        if num_mp_layers < 1:
            raise ValueError(f"num_mp_layers must be >= 1, got {num_mp_layers}")
        if decoder_layers < 1:
            raise ValueError(f"decoder_layers must be >= 1, got {decoder_layers}")
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")

        self.dropout = dropout
        self.use_residual = use_residual
        self.use_edge_features = use_edge_features

        if attn_dropout is None:
            attn_dropout = dropout

        # --- Encoder MLP ---
        enc_dims = [input_dim] + [hidden_dim] * encoder_layers
        self.encoder = _build_mlp(enc_dims)

        # --- Processor (attention layers) ---
        head_dim = hidden_dim // heads
        edge_dim = 1 if use_edge_features else None

        self.convs = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    edge_dim=edge_dim,
                    dropout=attn_dropout,
                )
            )

        # --- Decoder MLP ---
        dec_dims = [hidden_dim] * decoder_layers + [output_dim]
        self.decoder = _build_mlp(dec_dims)

    def forward(self, data) -> torch.Tensor:
        """Forward pass: encode → process (attention) → decode.

        Args:
            data: PyG Batch object with attributes ``x``, ``edge_index``, and
                optionally ``edge_weight``.

        Returns:
            Predictions of shape ``(total_nodes_in_batch, output_dim)``.
        """
        x, edge_index = data.x, data.edge_index

        # Prepare edge attributes for attention bias
        edge_attr = None
        if self.use_edge_features:
            edge_weight = getattr(data, "edge_weight", None)
            if edge_weight is not None:
                edge_attr = edge_weight.unsqueeze(-1)  # (n_edges,) → (n_edges, 1)

        # Encoder (node-local)
        x = self.encoder(x)

        # Processor (attention-based message passing with optional residual)
        for conv in self.convs:
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + h
            else:
                x = h

        # Decoder (node-local)
        return self.decoder(x)
