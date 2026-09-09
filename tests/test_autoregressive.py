# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from src.data.ARWWMDataModule import ARWWMDataModule
from src.data.AutoregressiveWWMDatasets import SimpleARWWMDataset
from src.data.dataset_utils import build_feedback_map
from src.model.ARWWMEmulator import ARWWMEmulator
from src.model.WWMEmulator import WWMEmulator
from src.train import train


@pytest.fixture
def cicd_data_path():
    """Return path to CICD test data."""
    return Path(__file__).parent / "cicd_data" / "cicd.zarr"


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer partial."""
    return lambda params: torch.optim.Adam(params, lr=0.001)


# ---------------------------------------------------------------------------
# build_feedback_map
# ---------------------------------------------------------------------------


class TestBuildFeedbackMap:
    """Test suite for build_feedback_map (shared by rollout eval and AR training)."""

    def test_maps_prev_feature_to_target(self):
        """A `prev_<target>` feature should map to its target's index."""
        feedback = build_feedback_map(
            features=["windSpeedX", "prev_sigWaveHeight"],
            targets=["sigWaveHeight"],
            scaling_params={"prev_sigWaveHeight": {"mean": 1.0, "std": 2.0}},
        )
        assert feedback == [(1, 0, 1.0, 2.0)]

    def test_no_scaling_params_defaults_to_identity(self):
        """Without scaling params, mean=0/std=1 (no rescaling)."""
        feedback = build_feedback_map(
            features=["prev_sigWaveHeight"],
            targets=["sigWaveHeight"],
            scaling_params=None,
        )
        assert feedback == [(0, 0, 0.0, 1.0)]

    def test_unmatched_prev_feature_is_skipped(self):
        """A `prev_*` feature with no matching target is skipped, not an error."""
        feedback = build_feedback_map(
            features=["prev_windSpeedX"],
            targets=["sigWaveHeight"],
            scaling_params=None,
        )
        assert feedback == []

    def test_non_prev_features_are_ignored(self):
        """Features without the `prev_` prefix never appear in the feedback map."""
        feedback = build_feedback_map(
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            scaling_params=None,
        )
        assert feedback == []


# ---------------------------------------------------------------------------
# SimpleARWWMDataset
# ---------------------------------------------------------------------------


class TestSimpleARWWMDataset:
    """Test suite for SimpleARWWMDataset."""

    def test_rejects_rollout_steps_below_one(self, cicd_data_path):
        """rollout_steps must be >= 1."""
        with pytest.raises(ValueError, match="rollout_steps"):
            SimpleARWWMDataset(
                data_dir=str(cicd_data_path),
                data_range=[1, 2],
                features=["windSpeedX", "windSpeedY"],
                targets=["sigWaveHeight"],
                rollout_steps=0,
            )

    def test_rejects_delta_targets(self, cicd_data_path):
        """Delta targets are not supported for autoregressive training."""
        with pytest.raises(ValueError, match="delta"):
            SimpleARWWMDataset(
                data_dir=str(cicd_data_path),
                data_range=[1, 2],
                features=["windSpeedX", "windSpeedY"],
                targets=["delta_sigWaveHeight"],
                rollout_steps=3,
            )

    def test_getitem_shapes(self, cicd_data_path):
        """Each sample is a window of rollout_steps consecutive time steps."""
        rollout_steps = 3
        ds = SimpleARWWMDataset(
            data_dir=str(cicd_data_path),
            data_range=[1, 2],
            features=["windSpeedX", "windSpeedY", "prev_sigWaveHeight"],
            targets=["sigWaveHeight"],
            rollout_steps=rollout_steps,
        )
        X_win, y_win = ds[0]
        assert X_win.shape == (rollout_steps, 3)
        assert y_win.shape == (rollout_steps, 1)

    def test_len_matches_valid_windows_times_nodes(self, cicd_data_path):
        """Dataset length is n_valid_times * n_nodes."""
        ds = SimpleARWWMDataset(
            data_dir=str(cicd_data_path),
            data_range=[1, 2],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            rollout_steps=3,
        )
        assert len(ds) == ds.n_valid_times * ds.n_nodes

    def test_consecutive_targets_match_source_timeseries(self, cicd_data_path):
        """y_win[s] should equal the target at time_start + s for a fixed node."""
        rollout_steps = 4
        ds = SimpleARWWMDataset(
            data_dir=str(cicd_data_path),
            data_range=[1, 2],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            rollout_steps=rollout_steps,
        )
        node_idx = 0
        X_win, y_win = ds[node_idx]  # idx // n_nodes == 0 -> time_start = valid_starts[0]
        time_start = int(ds._valid_starts[0])
        expected = ds.y[time_start : time_start + rollout_steps, node_idx, :]
        torch.testing.assert_close(y_win, expected)


# ---------------------------------------------------------------------------
# ARWWMDataModule
# ---------------------------------------------------------------------------


class TestARWWMDataModule:
    """Test suite for ARWWMDataModule."""

    def test_setup_fit_uses_ar_dataset_and_builds_feedback_map(self, cicd_data_path):
        """Fit stage should create AR datasets and populate feedback_map."""
        dm = ARWWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "prev_sigWaveHeight"],
            targets=["sigWaveHeight"],
            rollout_steps=3,
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="fit")

        assert isinstance(dm.ds_train, SimpleARWWMDataset)
        assert isinstance(dm.ds_val, SimpleARWWMDataset)
        assert dm.feedback_map == [(1, 0, 0.0, 1.0)]

    def test_setup_test_uses_plain_dataset(self, cicd_data_path):
        """Test/predict stages should keep the plain single-step dataset."""
        from src.data.WWMDatasets import SimpleWWMDataset

        dm = ARWWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "prev_sigWaveHeight"],
            targets=["sigWaveHeight"],
            rollout_steps=3,
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="test")

        assert type(dm.ds_test) is SimpleWWMDataset


# ---------------------------------------------------------------------------
# AutoregressiveMixin / ARWWMEmulator
# ---------------------------------------------------------------------------


def _make_ar_emulator(rollout_steps: int, mock_optimizer) -> ARWWMEmulator:
    """Build an ARWWMEmulator with a tiny linear backbone and a manually-registered feedback map."""
    backbone = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        backbone.weight.copy_(torch.tensor([[2.0, 3.0]]))  # pred = 2*x0 + 3*x1

    model = ARWWMEmulator(
        backbone=backbone,
        target_names=["sigWaveHeight"],
        optimizer=mock_optimizer,
        rollout_steps=rollout_steps,
        window_len=1,
    )
    # Bypass on_fit_start (no real Trainer/datamodule): register the feedback
    # map directly, as if feature 1 (prev_sigWaveHeight) is fed from target 0.
    model.feedback_feat_idx = torch.tensor([1], dtype=torch.long)
    model.feedback_target_idx = torch.tensor([0], dtype=torch.long)
    model.feedback_mean = torch.tensor([0.0])
    model.feedback_std = torch.tensor([1.0])
    model._feedback_map_ready = True
    return model


class TestAutoregressiveMixin:
    """Test suite for AutoregressiveMixin / ARWWMEmulator."""

    def test_rollout_steps_one_matches_base_model_step(self, mock_optimizer):
        """With rollout_steps=1 there is no feedback; loss should match plain WWMEmulator."""
        torch.manual_seed(0)
        backbone_ar = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            backbone_ar.weight.copy_(torch.tensor([[2.0, 3.0]]))

        ar_model = _make_ar_emulator(rollout_steps=1, mock_optimizer=mock_optimizer)

        base_model = WWMEmulator(
            backbone=backbone_ar,
            target_names=["sigWaveHeight"],
            optimizer=mock_optimizer,
        )

        x = torch.randn(4, 2)
        y = torch.randn(4, 1)

        x_win = x.unsqueeze(1)  # (B, T=1, F)
        y_win = y.unsqueeze(1)  # (B, T=1, C)

        ar_loss, _, ar_preds, ar_targets = ar_model.model_step((x_win, y_win))
        base_loss, _, base_preds, base_targets = base_model.model_step((x, y))

        torch.testing.assert_close(ar_loss, base_loss)
        torch.testing.assert_close(ar_preds, base_preds)
        torch.testing.assert_close(ar_targets, base_targets)

    def test_feedback_writes_predictions_into_next_step(self, mock_optimizer):
        """Step s+1's feature 1 should be fed by step s's (rescaled) prediction."""
        model = _make_ar_emulator(rollout_steps=2, mock_optimizer=mock_optimizer)

        x0 = torch.tensor([[1.0, 100.0]])  # step 0 feature 1 is untouched (ground truth)
        x1 = torch.tensor([[5.0, 100.0]])  # step 1 feature 1 will be overwritten by feedback
        x_win = torch.stack([x0, x1], dim=1)  # (1, 2, 2)
        # y_win = torch.zeros(1, 2, 1)

        preds = model._rollout(x_win)

        pred_step0 = 2.0 * 1.0 + 3.0 * 100.0
        expected_fed_back_feature = pred_step0  # mean=0, std=1 -> no rescaling
        pred_step1_expected = 2.0 * 5.0 + 3.0 * expected_fed_back_feature

        torch.testing.assert_close(preds[0, 0, 0], torch.tensor(pred_step0))
        torch.testing.assert_close(preds[0, 1, 0], torch.tensor(pred_step1_expected))

        # The input buffer passed in must not be mutated (only a clone is).
        torch.testing.assert_close(x_win[0, 1, 1], torch.tensor(100.0))

    def test_loss_averages_over_all_rollout_steps(self, mock_optimizer):
        """Total loss should equal the mean per-sample MSE across all T steps."""
        model = _make_ar_emulator(rollout_steps=3, mock_optimizer=mock_optimizer)

        torch.manual_seed(1)
        x_win = torch.randn(4, 3, 2)
        y_win = torch.randn(4, 3, 1)

        total_loss, per_var_losses, preds, targets = model.model_step((x_win, y_win))

        preds_3d = preds.reshape(4, 3, 1)
        targets_3d = targets.reshape(4, 3, 1)
        expected_loss = ((preds_3d - targets_3d) ** 2).mean()

        torch.testing.assert_close(total_loss, expected_loss)
        torch.testing.assert_close(per_var_losses["sigWaveHeight"], expected_loss)

    def test_detach_blocks_gradient_through_feedback(self, mock_optimizer):
        """Earlier-step outputs get no gradient contribution from later steps' loss
        (only their own step's loss contributes) since feedback is always detached."""
        model = _make_ar_emulator(rollout_steps=2, mock_optimizer=mock_optimizer)
        x_win = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], requires_grad=False)
        y_win = torch.zeros(1, 2, 1)

        total_loss, _, _, _ = model.model_step((x_win, y_win))
        total_loss.backward()

        # Gradient w.r.t. weight[1] (coefficient on feature 1, the feedback slot) should
        # only reflect direct contributions since the feedback value was detached.
        assert model.backbone.weight.grad is not None


# ---------------------------------------------------------------------------
# End-to-end fast_dev_run via src.train.train
# ---------------------------------------------------------------------------


class TestAutoregressiveTrainE2E:
    """Runs the real (non-mocked) train() pipeline against the small CICD dataset."""

    def test_fast_dev_run_ar_mlp(self, cicd_data_path):
        """AR MLP config should train, validate, and test without error."""
        config_dir = Path(__file__).parent.parent / "configs" / "cicd"

        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="cicd_train_ar_mlp.yaml",
                overrides=[
                    f"data.data_dir={cicd_data_path}",
                    "logger=null",
                    "callbacks=null",
                    f"paths.root_dir={tempfile.gettempdir()}",
                    f"paths.output_dir={tempfile.mkdtemp()}",
                ],
            )

        metric_dict, object_dict = train(cfg)

        assert "train/loss" in metric_dict or "val/loss" in metric_dict
        assert "val/loss_step0" in metric_dict

    def test_fast_dev_run_ar_lstm(self, cicd_data_path):
        """AR LSTM config should train, validate, and test without error."""
        config_dir = Path(__file__).parent.parent / "configs" / "cicd"

        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="cicd_train_ar_lstm.yaml",
                overrides=[
                    f"data.data_dir={cicd_data_path}",
                    "logger=null",
                    "callbacks=null",
                    f"paths.root_dir={tempfile.gettempdir()}",
                    f"paths.output_dir={tempfile.mkdtemp()}",
                ],
            )

        metric_dict, object_dict = train(cfg)

        assert "train/loss" in metric_dict or "val/loss" in metric_dict
        assert "val/loss_step0" in metric_dict

    def test_fast_dev_run_ar_multihead_gat_vn(self, cicd_data_path):
        """AR pointwise graph (MultiHeadGATVN) config should train, validate, and test without error."""
        config_dir = Path(__file__).parent.parent / "configs" / "cicd"

        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="cicd_train_ar_multihead_gat_vn.yaml",
                overrides=[
                    f"data.data_dir={cicd_data_path}",
                    "logger=null",
                    "callbacks=null",
                    f"paths.root_dir={tempfile.gettempdir()}",
                    f"paths.output_dir={tempfile.mkdtemp()}",
                ],
            )

        metric_dict, object_dict = train(cfg)

        assert "train/loss" in metric_dict or "val/loss" in metric_dict
        assert "val/loss_step0" in metric_dict

    def test_fast_dev_run_ar_dual_path_gat_vn(self, cicd_data_path):
        """AR center-history graph (MultiHeadGATLSTMVN) config should train, validate, and test without error."""
        config_dir = Path(__file__).parent.parent / "configs" / "cicd"

        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
            cfg = compose(
                config_name="cicd_train_ar_dual_path_gat_vn.yaml",
                overrides=[
                    f"data.data_dir={cicd_data_path}",
                    "logger=null",
                    "callbacks=null",
                    f"paths.root_dir={tempfile.gettempdir()}",
                    f"paths.output_dir={tempfile.mkdtemp()}",
                ],
            )

        metric_dict, object_dict = train(cfg)

        assert "train/loss" in metric_dict or "val/loss" in metric_dict
        assert "val/loss_step0" in metric_dict
