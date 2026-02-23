# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import Literal

import torch
import torch.nn as nn


class DenseNet(nn.Module):
    """Flexible feed-forward neural network with configurable architecture.

    A dense network with configurable number of layers, neurons, and scaling mode.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 3,
        first_layer_neurons: int = 128,
        layer_scaling: Literal["constant", "halving"] = "halving",
        dropout_rate: float = 0.0,
    ):
        """Initialize the DenseNet.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output features.
            num_hidden_layers: Number of hidden layers (default: 3).
            first_layer_neurons: Number of neurons in the first hidden layer (default: 128).
            layer_scaling: Mode for neuron count across layers (default: "halving").
                          - "constant": All layers have the same number of neurons.
                          - "halving": Each layer has half the neurons of the previous layer.
            dropout_rate: Dropout probability applied after each hidden layer (default: 0.0).

        Examples:
            - num_hidden_layers=3, first_layer_neurons=128, layer_scaling="constant" → [128, 128, 128]
            - num_hidden_layers=3, first_layer_neurons=128, layer_scaling="halving" → [128, 64, 32]
        """
        super().__init__()

        # Build the hidden layer dimensions based on scaling mode
        hidden_dims = []
        current_neurons = first_layer_neurons

        for _ in range(num_hidden_layers):
            hidden_dims.append(current_neurons)
            if layer_scaling == "halving":
                current_neurons = max(1, current_neurons // 2)  # Ensure at least 1 neuron

        # Build the network layers
        layers = []
        prev_dim = input_dim

        # Build hidden layers with ReLU activation and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
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
