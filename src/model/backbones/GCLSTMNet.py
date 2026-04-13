# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCLSTMCell(nn.Module):
    """Graph Convolutional LSTM cell.

    Replaces the linear transformations in a standard LSTM cell with graph
    convolutions (GCNConv), so that the forget, input, output gates and
    cell candidate all aggregate information from neighbouring nodes.

    Based on Si et al., *An Attention Enhanced Graph Convolutional LSTM
    Network for Skeleton-Based Action Recognition*, CVPR 2019.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """Initialize the GC-LSTM cell.

        Args:
            input_dim: Number of input features per node.
            hidden_dim: Number of hidden units per node.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Fused graph convolutions for all 4 gates (i, f, c̃, o)
        self.conv_x = GCNConv(input_dim, 4 * hidden_dim, add_self_loops=False)
        self.conv_h = GCNConv(hidden_dim, 4 * hidden_dim, add_self_loops=False)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> tuple:
        """Forward pass for one timestep.

        Args:
            x_t: Node input features ``(total_nodes, input_dim)``.
            h_prev: Previous hidden state ``(total_nodes, hidden_dim)``.
            c_prev: Previous cell state ``(total_nodes, hidden_dim)``.
            edge_index: Edge index ``(2, n_edges)``.
            edge_weight: Optional edge weights ``(n_edges,)``.

        Returns:
            Tuple of (h_t, c_t), each ``(total_nodes, hidden_dim)``.
        """
        gates = self.conv_x(x_t, edge_index, edge_weight=edge_weight) + self.conv_h(
            h_prev, edge_index, edge_weight=edge_weight
        )

        i, f, c_tilde, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        c_tilde = torch.tanh(c_tilde)
        o = torch.sigmoid(o)

        c_t = f * c_prev + i * c_tilde
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class GCLSTMNet(nn.Module):
    """Graph Convolutional LSTM backbone for full-graph sequence prediction.

    Uses ``GCLSTMCell`` layers where graph convolutions replace the linear
    transforms inside the LSTM gates.  This tightly couples spatial message
    passing with temporal gating at every timestep, unlike
    ``GraphConvLSTMNet`` which runs GCN and LSTM sequentially.

    Designed for ``SequenceGraphWWMDataset`` with ``full_graph_target=True``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """Initialize the GC-LSTM network.

        Args:
            input_dim: Number of input features per node per timestep.
            output_dim: Number of output features per node.
            hidden_dim: Hidden dimension for all GC-LSTM layers (default: 64).
            num_layers: Number of stacked GC-LSTM layers (default: 1).
            dropout: Dropout probability applied between stacked layers
                at each timestep (default: 0.0).  Not applied after the
                last layer.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.cells = nn.ModuleList()
        self.cells.append(GCLSTMCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.cells.append(GCLSTMCell(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data) -> torch.Tensor:
        """Forward pass: unroll GC-LSTM cells over the temporal sequence.

        Args:
            data: PyG Batch object with attributes:
                - ``x``: ``(total_nodes, seq_len, n_features)``
                - ``edge_index``: ``(2, n_edges)``
                - ``edge_weight``: optional ``(n_edges,)``

        Returns:
            Predictions of shape ``(total_nodes, output_dim)`` for all nodes.
        """
        x_seq, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        total_nodes, seq_len, _ = x_seq.shape
        device = x_seq.device

        # Initialise hidden and cell states for each layer
        h = [torch.zeros(total_nodes, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(total_nodes, self.hidden_dim, device=device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(x_t, h[layer_idx], c[layer_idx], edge_index, edge_weight)
                x_t = h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    x_t = F.dropout(x_t, p=self.dropout, training=self.training)

        # Project the final hidden state of the last layer
        return self.fc(h[-1])
