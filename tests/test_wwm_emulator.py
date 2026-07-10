# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock

import pytest
import torch
import numpy as np

from src.model.WWMEmulator import WWMEmulator
from src.model.backbones.MLP import MLP
from src.model.backbones.LSTMNet import LSTMNet


@pytest.fixture
def simple_backbone():
    """Create a simple MLP backbone."""
    return MLP(input_dim=10, output_dim=2)


@pytest.fixture
def lstm_backbone():
    """Create a simple LSTM backbone."""
    return LSTMNet(input_dim=10, output_dim=2)


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer partial."""
    return lambda params: torch.optim.Adam(params, lr=0.001)


@pytest.fixture
def target_names():
    """Return sample target names."""
    return ["sigWaveHeight", "peakPeriod"]


@pytest.fixture
def emulator(simple_backbone, target_names, mock_optimizer):
    """Create a WWMEmulator instance."""
    return WWMEmulator(
        backbone=simple_backbone,
        target_names=target_names,
        optimizer=mock_optimizer,
    )


class TestWWMEmulatorInit:
    """Test WWMEmulator initialization."""

    def test_init_basic(self, simple_backbone, target_names, mock_optimizer):
        """Test basic initialization."""
        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
        )
        assert emulator.backbone is simple_backbone
        assert emulator.target_names == target_names
        assert emulator.num_targets == 2

    def test_init_with_loss_weights(self, simple_backbone, target_names, mock_optimizer):
        """Test initialization with custom loss weights."""
        loss_weights = {"sigWaveHeight": 2.0, "peakPeriod": 0.5}
        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
            loss_weights=loss_weights,
        )
        assert emulator.loss_weights["sigWaveHeight"] == 2.0
        assert emulator.loss_weights["peakPeriod"] == 0.5

    def test_init_default_loss_weights(self, simple_backbone, target_names, mock_optimizer):
        """Test that default loss weights are 1.0."""
        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
        )
        assert emulator.loss_weights["sigWaveHeight"] == 1.0
        assert emulator.loss_weights["peakPeriod"] == 1.0

    def test_init_is_delta_target(self, simple_backbone, mock_optimizer):
        """Test is_delta_target detection."""
        target_names = ["delta_sigWaveHeight", "peakPeriod"]
        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
        )
        assert emulator.is_delta_target["delta_sigWaveHeight"] is True
        assert emulator.is_delta_target["peakPeriod"] is False

    def test_init_with_target_bins(self, simple_backbone, target_names, mock_optimizer):
        """Test initialization with target bins."""
        target_bins = {"sigWaveHeight": [-1.0, 0.0, 0.5, 1.0]}
        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
            target_bins=target_bins,
        )
        assert "sigWaveHeight" in emulator.target_bins
        assert len(emulator.val_rmse_binned) == 5  # 4 bins + 1


class TestWWMEmulatorForward:
    """Test WWMEmulator forward pass."""

    def test_forward_basic(self, emulator):
        """Test basic forward pass."""
        x = torch.randn(8, 10)
        output = emulator(x)
        assert output.shape == (8, 2)

    def test_forward_preserves_dtype(self, emulator):
        """Test that forward preserves input dtype."""
        x = torch.randn(8, 10, dtype=torch.float32)
        output = emulator(x)
        assert output.dtype == torch.float32


class TestWWMEmulatorModelStep:
    """Test WWMEmulator model_step method."""

    def test_model_step_without_weights(self, emulator):
        """Test model_step without sample weights."""
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        batch = (x, y)

        total_loss, per_var_losses, preds, targets = emulator.model_step(batch)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # scalar
        assert len(per_var_losses) == 2
        assert "sigWaveHeight" in per_var_losses
        assert "peakPeriod" in per_var_losses
        assert preds.shape == (8, 2)
        assert targets.shape == (8, 2)

    def test_model_step_with_weights(self, emulator):
        """Test model_step with sample weights."""
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        weights = torch.ones(8, 2) * 2.0
        batch = (x, y, weights)

        total_loss, per_var_losses, preds, targets = emulator.model_step(batch)

        assert isinstance(total_loss, torch.Tensor)
        assert preds.shape == (8, 2)

    def test_model_step_loss_weights_applied(self, simple_backbone, mock_optimizer):
        """Test that loss weights are applied correctly."""
        target_names = ["var1", "var2"]
        loss_weights = {"var1": 2.0, "var2": 0.5}
        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
            loss_weights=loss_weights,
        )

        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        batch = (x, y)

        total_loss, per_var_losses, _, _ = emulator.model_step(batch)

        # Total loss should be weighted sum
        expected_total = per_var_losses["var1"] * 2.0 + per_var_losses["var2"] * 0.5
        assert torch.allclose(total_loss, expected_total)


class TestWWMEmulatorTraining:
    """Test WWMEmulator training functionality."""

    def test_training_step_runs(self, emulator):
        """Test that training_step runs without error."""
        # Mock trainer for logging
        emulator.trainer = MagicMock()
        emulator.trainer.callback_metrics = {}

        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        batch = (x, y)

        loss = emulator.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_validation_step_runs(self, emulator):
        """Test that validation_step runs without error."""
        emulator.trainer = MagicMock()
        emulator.trainer.callback_metrics = {}

        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        batch = (x, y)

        # Should not raise
        emulator.validation_step(batch, batch_idx=0)

    def test_test_step_runs(self, emulator):
        """Test that test_step runs without error."""
        emulator.trainer = MagicMock()
        emulator.trainer.callback_metrics = {}

        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        batch = (x, y)

        # Should not raise
        emulator.test_step(batch, batch_idx=0)

    def test_predict_step_returns_predictions(self, emulator):
        """Test that predict_step returns predictions."""
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)
        batch = (x, y)

        preds = emulator.predict_step(batch, batch_idx=0)
        assert preds.shape == (8, 2)


class TestWWMEmulatorOptimizer:
    """Test WWMEmulator optimizer configuration."""

    def test_configure_optimizers_without_scheduler(self, simple_backbone, target_names, mock_optimizer):
        """Test optimizer configuration without scheduler."""
        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
        )
        # Mock trainer
        emulator.trainer = MagicMock()
        emulator.trainer.model.parameters.return_value = simple_backbone.parameters()

        opt_config = emulator.configure_optimizers()
        assert "optimizer" in opt_config
        assert isinstance(opt_config["optimizer"], torch.optim.Adam)

    def test_configure_optimizers_with_scheduler(self, simple_backbone, target_names, mock_optimizer):
        """Test optimizer configuration with scheduler."""
        scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)  # noqa: E731

        emulator = WWMEmulator(
            backbone=simple_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
            scheduler=scheduler,
        )
        # Mock trainer
        emulator.trainer = MagicMock()
        emulator.trainer.model.parameters.return_value = simple_backbone.parameters()

        opt_config = emulator.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        assert opt_config["lr_scheduler"]["monitor"] == "val/loss"


class TestWWMEmulatorWithLSTM:
    """Test WWMEmulator with LSTM backbone."""

    def test_forward_with_lstm(self, lstm_backbone, target_names, mock_optimizer):
        """Test forward pass with LSTM backbone."""
        emulator = WWMEmulator(
            backbone=lstm_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
        )
        x = torch.randn(8, 5, 10)  # batch, seq_len, input_dim
        output = emulator(x)
        assert output.shape == (8, 2)

    def test_model_step_with_lstm(self, lstm_backbone, target_names, mock_optimizer):
        """Test model_step with LSTM backbone."""
        emulator = WWMEmulator(
            backbone=lstm_backbone,
            target_names=target_names,
            optimizer=mock_optimizer,
        )
        x = torch.randn(8, 5, 10)
        y = torch.randn(8, 2)
        batch = (x, y)

        total_loss, per_var_losses, preds, targets = emulator.model_step(batch)
        assert preds.shape == (8, 2)
        assert total_loss.ndim == 0


class TestUnitCircleConstraint:
    """Test suite for the unit_circle paired output constraint."""

    @pytest.fixture
    def mock_optimizer(self):
        return lambda params: torch.optim.Adam(params, lr=0.001)

    def test_unit_circle_normalizes_to_unit_circle(self, mock_optimizer):
        """sin/cos pair is projected onto the unit circle."""
        backbone = MLP(input_dim=2, output_dim=2)
        emulator = WWMEmulator(
            backbone=backbone,
            target_names=["sin_meanWaveDirection", "cos_meanWaveDirection"],
            optimizer=mock_optimizer,
            output_constraints={
                "sin_meanWaveDirection": "unit_circle",
                "cos_meanWaveDirection": "unit_circle",
            },
        )
        # Feed a non-unit-norm pair (3, 4) — norm = 5, expected output (0.6, 0.8)
        preds = torch.tensor([[3.0, 4.0]])
        constrained = emulator._apply_output_constraints(preds)
        np.testing.assert_allclose(constrained[0, 0].item(), 0.6, atol=1e-5)
        np.testing.assert_allclose(constrained[0, 1].item(), 0.8, atol=1e-5)

    def test_unit_circle_output_has_unit_norm(self, mock_optimizer):
        """For arbitrary inputs, constrained sin²+cos² == 1."""
        backbone = MLP(input_dim=2, output_dim=2)
        emulator = WWMEmulator(
            backbone=backbone,
            target_names=["sin_meanWaveDirection", "cos_meanWaveDirection"],
            optimizer=mock_optimizer,
            output_constraints={
                "sin_meanWaveDirection": "unit_circle",
                "cos_meanWaveDirection": "unit_circle",
            },
        )
        preds = torch.tensor([[3.0, 4.0], [-1.0, 2.0], [0.0, 5.0]])
        constrained = emulator._apply_output_constraints(preds)
        norms = (constrained[:, 0] ** 2 + constrained[:, 1] ** 2).sqrt()
        np.testing.assert_allclose(norms.detach().numpy(), np.ones(3), atol=1e-5)

    def test_unit_circle_with_mixed_constraints(self, mock_optimizer):
        """semipositive constraint on sigWaveHeight is unaffected by unit_circle logic."""
        backbone = MLP(input_dim=3, output_dim=3)
        emulator = WWMEmulator(
            backbone=backbone,
            target_names=["sigWaveHeight", "sin_meanWaveDirection", "cos_meanWaveDirection"],
            optimizer=mock_optimizer,
            output_constraints={
                "sigWaveHeight": "semipositive",
                "sin_meanWaveDirection": "unit_circle",
                "cos_meanWaveDirection": "unit_circle",
            },
        )
        preds = torch.tensor([[-2.0, 3.0, 4.0]])  # sigWaveHeight<0 should clamp to 0
        constrained = emulator._apply_output_constraints(preds)
        # sigWaveHeight: relu(-2) = 0
        assert constrained[0, 0].item() == pytest.approx(0.0)
        # sin/cos: (3,4) -> (0.6, 0.8)
        np.testing.assert_allclose(constrained[0, 1].item(), 0.6, atol=1e-5)
        np.testing.assert_allclose(constrained[0, 2].item(), 0.8, atol=1e-5)

    def test_unit_circle_no_partner_leaves_unchanged(self, mock_optimizer):
        """If only sin_ has unit_circle but no matching cos_, channel is left as-is."""
        backbone = MLP(input_dim=1, output_dim=1)
        emulator = WWMEmulator(
            backbone=backbone,
            target_names=["sin_meanWaveDirection"],
            optimizer=mock_optimizer,
            output_constraints={"sin_meanWaveDirection": "unit_circle"},
        )
        preds = torch.tensor([[3.0]])
        constrained = emulator._apply_output_constraints(preds)
        assert constrained[0, 0].item() == pytest.approx(3.0)
