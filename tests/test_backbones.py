# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
from torch_geometric.data import Data as GData

from src.model.backbones.DenseNet import DenseNet
from src.model.backbones.GCLSTMNet import GCLSTMNet
from src.model.backbones.LSTMNet import LSTMNet


class TestDenseNet:
    """Test suite for DenseNet backbone."""

    def test_forward_basic(self):
        """Test basic forward pass with default parameters."""
        model = DenseNet(input_dim=10, output_dim=3)
        x = torch.randn(8, 10)  # batch_size=8, input_dim=10
        output = model(x)
        assert output.shape == (8, 3)

    def test_forward_constant_scaling(self):
        """Test forward pass with constant layer scaling."""
        model = DenseNet(
            input_dim=20,
            output_dim=5,
            num_hidden_layers=3,
            first_layer_neurons=64,
            layer_scaling="constant",
        )
        x = torch.randn(4, 20)
        output = model(x)
        assert output.shape == (4, 5)

    def test_forward_halving_scaling(self):
        """Test forward pass with halving layer scaling."""
        model = DenseNet(
            input_dim=20,
            output_dim=5,
            num_hidden_layers=3,
            first_layer_neurons=64,
            layer_scaling="halving",
        )
        x = torch.randn(4, 20)
        output = model(x)
        assert output.shape == (4, 5)

    def test_forward_with_dropout(self):
        """Test forward pass with dropout enabled."""
        model = DenseNet(input_dim=10, output_dim=2, dropout_rate=0.5)
        x = torch.randn(8, 10)
        # Test in training mode (dropout active)
        model.train()
        output_train = model(x)
        # Test in eval mode (dropout inactive)
        model.eval()
        output_eval = model(x)
        assert output_train.shape == (8, 2)
        assert output_eval.shape == (8, 2)

    def test_different_layer_configurations(self):
        """Test various layer configurations."""
        configs = [
            {"num_hidden_layers": 1, "first_layer_neurons": 32},
            {"num_hidden_layers": 5, "first_layer_neurons": 256},
            {"num_hidden_layers": 2, "first_layer_neurons": 16},
        ]
        for cfg in configs:
            model = DenseNet(input_dim=10, output_dim=2, **cfg)
            x = torch.randn(2, 10)
            output = model(x)
            assert output.shape == (2, 2)

    def test_single_sample_batch(self):
        """Test forward pass with a single sample."""
        model = DenseNet(input_dim=10, output_dim=3)
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 3)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = DenseNet(input_dim=10, output_dim=3)
        x = torch.randn(4, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestLSTMNet:
    """Test suite for LSTMNet backbone."""

    def test_forward_basic(self):
        """Test basic forward pass with default parameters."""
        model = LSTMNet(input_dim=10, output_dim=3)
        x = torch.randn(8, 5, 10)  # batch=8, seq_len=5, input_dim=10
        output = model(x)
        assert output.shape == (8, 3)

    def test_forward_bidirectional(self):
        """Test forward pass with bidirectional LSTM."""
        model = LSTMNet(input_dim=10, output_dim=3, bidirectional=True)
        x = torch.randn(8, 5, 10)
        output = model(x)
        assert output.shape == (8, 3)

    def test_forward_multiple_layers(self):
        """Test forward pass with multiple LSTM layers."""
        model = LSTMNet(input_dim=10, output_dim=3, num_layers=4, dropout=0.2)
        x = torch.randn(8, 5, 10)
        output = model(x)
        assert output.shape == (8, 3)

    def test_forward_single_layer_no_dropout(self):
        """Test single layer LSTM (dropout should be ignored)."""
        model = LSTMNet(input_dim=10, output_dim=3, num_layers=1, dropout=0.5)
        x = torch.randn(4, 6, 10)
        model.eval()
        output = model(x)
        assert output.shape == (4, 3)

    def test_forward_varying_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        model = LSTMNet(input_dim=10, output_dim=3)
        for seq_len in [1, 5, 10, 50]:
            x = torch.randn(4, seq_len, 10)
            output = model(x)
            assert output.shape == (4, 3)

    def test_single_sample_batch(self):
        """Test forward pass with a single sample."""
        model = LSTMNet(input_dim=10, output_dim=3)
        x = torch.randn(1, 5, 10)
        output = model(x)
        assert output.shape == (1, 3)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = LSTMNet(input_dim=10, output_dim=3)
        x = torch.randn(4, 5, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_hidden_dims(self):
        """Test various hidden dimension configurations."""
        for hidden_dim in [32, 64, 256]:
            model = LSTMNet(input_dim=10, output_dim=3, hidden_dim=hidden_dim)
            x = torch.randn(2, 5, 10)
            output = model(x)
            assert output.shape == (2, 3)


def _make_graph_data(n_nodes=20, seq_len=5, n_features=10, n_edges=40):
    """Create a mock PyG Data object with sequential node features."""
    x = torch.randn(n_nodes, seq_len, n_features)
    # Random undirected edges (no self-loops needed; backbone uses add_self_loops=False)
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = torch.rand(n_edges)
    return GData(x=x, edge_index=edge_index, edge_weight=edge_weight)


class TestGCLSTMNet:
    """Test suite for GCLSTMNet backbone (graph convolutions inside LSTM gates)."""

    def test_forward_basic(self):
        """Test basic forward pass with default parameters."""
        model = GCLSTMNet(input_dim=10, output_dim=3)
        data = _make_graph_data(n_nodes=20, seq_len=5, n_features=10)
        output = model(data)
        assert output.shape == (20, 3)

    def test_forward_custom_dims(self):
        """Test forward pass with custom hidden dimension and multiple layers."""
        model = GCLSTMNet(
            input_dim=6,
            output_dim=2,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
        )
        data = _make_graph_data(n_nodes=15, seq_len=8, n_features=6)
        output = model(data)
        assert output.shape == (15, 2)

    def test_forward_single_timestep(self):
        """Test with seq_len=1 (degenerate temporal case)."""
        model = GCLSTMNet(input_dim=5, output_dim=2)
        data = _make_graph_data(n_nodes=12, seq_len=1, n_features=5)
        output = model(data)
        assert output.shape == (12, 2)

    def test_forward_single_node(self):
        """Test with a single-node graph."""
        model = GCLSTMNet(input_dim=3, output_dim=1)
        data = _make_graph_data(n_nodes=1, seq_len=4, n_features=3, n_edges=0)
        data.edge_index = torch.zeros(2, 0, dtype=torch.long)
        data.edge_weight = torch.zeros(0)
        output = model(data)
        assert output.shape == (1, 1)

    def test_dropout_train_vs_eval(self):
        """Test that dropout behaves differently in train vs eval mode."""
        model = GCLSTMNet(input_dim=5, output_dim=2, num_layers=2, dropout=0.5)
        data = _make_graph_data(n_nodes=10, seq_len=4, n_features=5)
        model.train()
        out_train = model(data)
        model.eval()
        out_eval = model(data)
        assert out_train.shape == (10, 2)
        assert out_eval.shape == (10, 2)

    def test_gradient_flow(self):
        """Test that gradients flow through the entire network."""
        model = GCLSTMNet(input_dim=5, output_dim=2)
        data = _make_graph_data(n_nodes=10, seq_len=4, n_features=5)
        data.x.requires_grad_(True)
        output = model(data)
        loss = output.sum()
        loss.backward()
        assert data.x.grad is not None
        assert data.x.grad.shape == data.x.shape

    def test_without_edge_weight(self):
        """Test forward pass when edge_weight is absent."""
        model = GCLSTMNet(input_dim=5, output_dim=2)
        data = _make_graph_data(n_nodes=10, seq_len=4, n_features=5)
        del data.edge_weight
        output = model(data)
        assert output.shape == (10, 2)

    def test_stacked_layers(self):
        """Test with multiple stacked GC-LSTM layers."""
        model = GCLSTMNet(input_dim=4, output_dim=1, hidden_dim=16, num_layers=3)
        data = _make_graph_data(n_nodes=8, seq_len=6, n_features=4)
        output = model(data)
        assert output.shape == (8, 1)
