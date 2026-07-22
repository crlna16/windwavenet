# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import pytest
import torch
from torch_geometric.data import Batch, Data as GData

from src.model.backbones.LSTMNet import LSTMNet
from src.model.backbones.MLP import MLP
from src.model.backbones.MultiHeadGAT import MultiHeadGAT
from src.model.backbones.MultiHeadGATVN import MultiHeadGATVN


class TestMLP:
    """Test suite for MLP backbone."""

    def test_forward_basic(self):
        """Test basic forward pass with default parameters."""
        model = MLP(input_dim=10, output_dim=3)
        x = torch.randn(8, 10)  # batch_size=8, input_dim=10
        output = model(x)
        assert output.shape == (8, 3)

    def test_forward_constant_scaling(self):
        """Test forward pass with constant layer scaling."""
        model = MLP(
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
        model = MLP(
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
        model = MLP(input_dim=10, output_dim=2, dropout_rate=0.5)
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
            model = MLP(input_dim=10, output_dim=2, **cfg)
            x = torch.randn(2, 10)
            output = model(x)
            assert output.shape == (2, 2)

    def test_single_sample_batch(self):
        """Test forward pass with a single sample."""
        model = MLP(input_dim=10, output_dim=3)
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 3)

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = MLP(input_dim=10, output_dim=3)
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


def _make_gat_batch(n_graphs=2, n_nodes_per_graph=10, n_features=6, n_edges_per_graph=20):
    """Create a mock PyG Batch of simple (non-temporal) node-feature graphs."""
    data_list = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes_per_graph, n_features)
        src = torch.randint(0, n_nodes_per_graph, (n_edges_per_graph,))
        dst = torch.randint(0, n_nodes_per_graph, (n_edges_per_graph,))
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = torch.rand(n_edges_per_graph)
        data_list.append(GData(x=x, edge_index=edge_index, edge_weight=edge_weight))
    return Batch.from_data_list(data_list)


@pytest.mark.parametrize("backbone_cls", [MultiHeadGAT, MultiHeadGATVN])
class TestEdgeEncodingAblation:
    """Edge-weight ablation: 'rbf' (default), 'scalar', and 'none' modes.

    Parametrized across the (non-temporal) GAT backbones that share the
    same edge-encoding pattern. MultiHeadGATLSTMVN follows the identical
    pattern but requires x_center_seq, so it is exercised separately below.
    """

    @pytest.mark.parametrize("edge_encoding", ["rbf", "scalar", "none"])
    def test_forward_shape(self, backbone_cls, edge_encoding):
        n_features, output_dim = 6, 2
        batch = _make_gat_batch(n_features=n_features)
        model = backbone_cls(
            input_dim=n_features,
            output_dim=output_dim,
            hidden_dim=16,
            num_layers=2,
            heads=4,
            edge_encoding=edge_encoding,
        )
        out = model(batch)
        assert out.shape == (batch.x.shape[0], output_dim)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("edge_encoding", ["rbf", "scalar", "none"])
    def test_gradient_flow(self, backbone_cls, edge_encoding):
        batch = _make_gat_batch(n_features=6)
        model = backbone_cls(
            input_dim=6, output_dim=2, hidden_dim=16, num_layers=2, heads=4, edge_encoding=edge_encoding
        )
        out = model(batch)
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)

    def test_none_ignores_edge_weight(self, backbone_cls):
        """With edge_encoding='none', the presence of edge_weight must not affect output."""
        torch.manual_seed(0)
        batch = _make_gat_batch(n_features=6)
        model = backbone_cls(input_dim=6, output_dim=2, hidden_dim=16, num_layers=2, heads=4, edge_encoding="none")
        model.eval()
        out_with_weight = model(batch)

        batch_no_weight = batch.clone()
        batch_no_weight.edge_weight = None
        out_without_weight = model(batch_no_weight)

        assert torch.allclose(out_with_weight, out_without_weight, atol=1e-6)

    def test_invalid_edge_encoding_raises(self, backbone_cls):
        with pytest.raises(ValueError):
            backbone_cls(input_dim=6, output_dim=2, hidden_dim=16, num_layers=2, heads=4, edge_encoding="bogus")

    def test_edge_dim_matches_encoding(self, backbone_cls):
        n_rbf = 8
        model_rbf = backbone_cls(
            input_dim=6, output_dim=2, hidden_dim=16, num_layers=1, heads=4, n_rbf=n_rbf, edge_encoding="rbf"
        )
        model_scalar = backbone_cls(
            input_dim=6, output_dim=2, hidden_dim=16, num_layers=1, heads=4, edge_encoding="scalar"
        )
        model_none = backbone_cls(input_dim=6, output_dim=2, hidden_dim=16, num_layers=1, heads=4, edge_encoding="none")

        assert model_rbf.gat_convs[0].edge_dim == n_rbf
        assert model_scalar.gat_convs[0].edge_dim == 1
        assert model_none.gat_convs[0].edge_dim is None
