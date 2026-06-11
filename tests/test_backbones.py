# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import torch
from torch_geometric.data import Batch, Data as GData

from src.model.backbones.DenseNet import DenseNet
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


def _make_center_history_batch(n_graphs=2, n_nodes_per_graph=10, n_features=3, n_edges_per_graph=20, seq_len=5):
    """Create a mock PyG Batch mimicking CenterHistoryData batching.

    Each sub-graph has ``n_nodes_per_graph`` nodes with random edges and a
    random center node.  ``x_center_seq`` is stacked along a new batch
    dimension (matching ``CenterHistoryData.__cat_dim__`` behaviour).
    """
    data_list = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes_per_graph, n_features)
        src = torch.randint(0, n_nodes_per_graph, (n_edges_per_graph,))
        dst = torch.randint(0, n_nodes_per_graph, (n_edges_per_graph,))
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = torch.rand(n_edges_per_graph)
        center_idx = torch.randint(0, n_nodes_per_graph, ())
        x_center_seq = torch.randn(seq_len, n_features)
        data_list.append(
            GData(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                center_idx=center_idx,
                x_center_seq=x_center_seq,
            )
        )
    # Batch.from_data_list stacks x_center_seq along a new dim when
    # __cat_dim__ returns None; we simulate that here by stacking manually.
    batch = Batch.from_data_list(data_list)
    # x_center_seq needs to be (n_graphs, seq_len, n_features)
    batch.x_center_seq = torch.stack([d.x_center_seq for d in data_list], dim=0)
    return batch
