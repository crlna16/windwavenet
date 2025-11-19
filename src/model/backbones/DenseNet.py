# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List

import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """Flexible feed-forward neural network with configurable hidden layers and dropout.

    A dense network architecture with ReLU activation and optional dropout between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.0,
    ):
        """Initialize the DenseNet.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions. For example, [128, 64, 32] creates
                         three hidden layers with 128, 64, and 32 neurons respectively.
            output_dim: Number of output features.
            dropout_rate: Dropout probability applied after each hidden layer (default: 0.0).
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers with ReLU activation and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
