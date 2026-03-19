# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """LSTM-based network for sequence-to-one prediction.

    Takes a sequence of features and predicts the next time step.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """Initialize the LSTM network.

        Args:
            input_dim: Number of input features per time step.
            output_dim: Number of output features.
            hidden_dim: Number of hidden units in LSTM (default: 128).
            num_layers: Number of stacked LSTM layers (default: 2).
            dropout: Dropout probability between LSTM layers (default: 0.1).
                     Only applied if num_layers > 1.
            bidirectional: If True, use bidirectional LSTM (default: False).
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output projection: map from LSTM hidden state to output
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # LSTM forward pass
        # output: (batch_size, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the output at the last time step
        # lstm_out[:, -1, :] has shape (batch_size, hidden_dim * num_directions)
        last_output = lstm_out[:, -1, :]

        # Project to output dimension
        out = self.fc(last_output)

        return out
