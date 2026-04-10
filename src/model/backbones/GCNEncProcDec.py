# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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


class GCNEncProcDec(nn.Module):
    """Encoder-Processor-Decoder GCN backbone for full-graph prediction.

    Follows the architecture from *Learning Mesh-Based Simulation with Graph
    Networks* (Pfaff et al., 2021).  The three stages are:

    **Encoder** — a node-local MLP that lifts heterogeneous input features
    into a uniform latent space (no message passing).

    **Processor** — a stack of GCNConv layers that perform message passing in
    latent space.  All layers operate at the same ``hidden_dim``.  Optional
    residual (skip) connections add each layer's input to its output, which
    helps combat oversmoothing.

    **Decoder** — a node-local MLP that maps latent embeddings back to the
    physical output space (no message passing).
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
    ):
        """Initialize the Encoder-Processor-Decoder network.

        Args:
            input_dim: Number of input features per node.
            output_dim: Number of output features per node.
            hidden_dim: Latent dimension shared across encoder, processor,
                and decoder.
            encoder_layers: Number of layers in the encoder MLP (>= 1).
                A single layer maps directly from input_dim to hidden_dim.
            num_mp_layers: Number of GCNConv message-passing layers in the
                processor (>= 1).
            decoder_layers: Number of layers in the decoder MLP (>= 1).
                A single layer maps directly from hidden_dim to output_dim.
            dropout: Dropout probability applied after each processor layer.
            use_residual: If True, add a residual (skip) connection around
                each processor GCNConv layer: ``x = x + conv(x, ...)``.
        """
        super().__init__()

        if encoder_layers < 1:
            raise ValueError(f"encoder_layers must be >= 1, got {encoder_layers}")
        if num_mp_layers < 1:
            raise ValueError(f"num_mp_layers must be >= 1, got {num_mp_layers}")
        if decoder_layers < 1:
            raise ValueError(f"decoder_layers must be >= 1, got {decoder_layers}")

        self.dropout = dropout
        self.use_residual = use_residual

        # --- Encoder MLP ---
        enc_dims = [input_dim] + [hidden_dim] * encoder_layers
        self.encoder = _build_mlp(enc_dims)

        # --- Processor (message-passing layers) ---
        self.convs = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))

        # --- Decoder MLP ---
        dec_dims = [hidden_dim] * decoder_layers + [output_dim]
        self.decoder = _build_mlp(dec_dims)

    def forward(self, data) -> torch.Tensor:
        """Forward pass: encode → process → decode.

        Args:
            data: PyG Batch object with attributes ``x``, ``edge_index``, and
                optionally ``edge_weight``.

        Returns:
            Predictions of shape ``(total_nodes_in_batch, output_dim)``.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        # Encoder (node-local)
        x = self.encoder(x)

        # Processor (message passing with optional residual)
        for conv in self.convs:
            h = conv(x, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + h
            else:
                x = h

        # Decoder (node-local)
        return self.decoder(x)
