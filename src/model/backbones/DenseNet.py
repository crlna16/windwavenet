# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """Simple feed-forward neural network with one hidden layer.

    A basic dense network architecture with ReLU activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize the DenseNet.

        Args:
            input_dim: Number of input features.
            hidden_dim: Number of neurons in the hidden layer.
            output_dim: Number of output features.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)
