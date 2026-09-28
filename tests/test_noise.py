# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from src.model.noise import PrevNoiseInjector, build_prev_noise_injector

FEATURES = ["windSpeedX", "windSpeedY", "prev_sigWaveHeight"]
# (feature_idx, target_idx, mean, std) — prev_sigWaveHeight is feature 2, std 0.5
FEEDBACK_MAP = [(2, 0, 1.0, 0.5)]


class _FakeDataModule:
    def __init__(self, feedback_map=FEEDBACK_MAP, scaling_config="scale.yaml"):
        self.features = FEATURES
        self.targets = ["sigWaveHeight"]
        self.feedback_map = feedback_map
        self.scaling_config = scaling_config


class TestPrevNoiseInjector:
    def test_sigma_is_converted_to_feature_space(self):
        """Sigma is physical; the feature is z-scored, so it must be divided by std."""
        injector = PrevNoiseInjector(FEATURES, {"sigWaveHeight": 0.1}, FEEDBACK_MAP)
        assert injector.feat_idx.tolist() == [2]
        assert injector.sigma.item() == pytest.approx(0.1 / 0.5)

    def test_only_prev_channels_are_touched(self):
        torch.manual_seed(0)
        injector = PrevNoiseInjector(FEATURES, {"sigWaveHeight": 0.1}, FEEDBACK_MAP)
        x = torch.randn(64, 3)
        out = injector(x)
        assert torch.equal(out[:, :2], x[:, :2])
        assert not torch.equal(out[:, 2], x[:, 2])

    def test_input_is_not_modified_in_place(self):
        torch.manual_seed(0)
        injector = PrevNoiseInjector(FEATURES, {"sigWaveHeight": 0.1}, FEEDBACK_MAP)
        x = torch.randn(32, 3)
        before = x.clone()
        injector(x)
        assert torch.equal(x, before)

    def test_perturbation_has_the_requested_scale(self):
        torch.manual_seed(0)
        injector = PrevNoiseInjector(FEATURES, {"sigWaveHeight": 0.1}, FEEDBACK_MAP)
        x = torch.zeros(200_000, 3)
        delta = injector(x)[:, 2]
        assert delta.std().item() == pytest.approx(0.2, rel=0.02)  # 0.1 / 0.5
        assert delta.mean().item() == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("shape", [(16, 3), (16, 5, 3)])
    def test_shapes_pointwise_and_sequence(self, shape):
        injector = PrevNoiseInjector(FEATURES, {"sigWaveHeight": 0.1}, FEEDBACK_MAP)
        out = injector(torch.randn(*shape))
        assert out.shape == shape

    def test_sequence_steps_get_independent_noise(self):
        """Each step of a sequence window is perturbed independently."""
        torch.manual_seed(0)
        injector = PrevNoiseInjector(FEATURES, {"sigWaveHeight": 0.1}, FEEDBACK_MAP)
        x = torch.zeros(4, 6, 3)
        delta = injector(x)[:, :, 2]
        assert not torch.allclose(delta[:, 0], delta[:, 1])

    def test_zero_sigma_is_inactive(self):
        injector = PrevNoiseInjector(FEATURES, {"sigWaveHeight": 0.0}, FEEDBACK_MAP)
        assert not injector
        x = torch.randn(8, 3)
        assert torch.equal(injector(x), x)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            PrevNoiseInjector(FEATURES, {"sigWaveHeight": -0.1}, FEEDBACK_MAP)

    def test_unknown_target_is_skipped(self):
        """A name with no matching prev_ feature is reported, not fatal."""
        injector = PrevNoiseInjector(FEATURES, {"meanWavePeriod": 0.2}, FEEDBACK_MAP)
        assert not injector


class TestBuildPrevNoiseInjector:
    def test_none_when_disabled(self):
        assert build_prev_noise_injector(_FakeDataModule(), None) is None
        assert build_prev_noise_injector(_FakeDataModule(), {}) is None

    def test_builds_from_datamodule(self):
        injector = build_prev_noise_injector(_FakeDataModule(), {"sigWaveHeight": 0.1})
        assert injector is not None
        assert injector.feat_idx.tolist() == [2]

    def test_missing_scaling_config_raises(self):
        """Without scaling_config the std is unknown and sigma would be silently wrong."""
        with pytest.raises(RuntimeError, match="scaling_config"):
            build_prev_noise_injector(_FakeDataModule(scaling_config=None), {"sigWaveHeight": 0.1})

    def test_empty_feedback_map_raises(self):
        with pytest.raises(RuntimeError, match="feedback_map"):
            build_prev_noise_injector(_FakeDataModule(feedback_map=[]), {"sigWaveHeight": 0.1})

    def test_resolves_to_nothing_returns_none(self):
        assert build_prev_noise_injector(_FakeDataModule(), {"meanWavePeriod": 0.2}) is None


class _StubTrainer:
    def __init__(self, training: bool):
        self.training = training
        self.datamodule = _FakeDataModule()


def _make_emulator(prev_noise_std=None):
    import torch.nn as nn

    from src.model.WWMEmulator import WWMEmulator

    return WWMEmulator(
        backbone=nn.Linear(3, 1),
        target_names=["sigWaveHeight"],
        optimizer=None,
        prev_noise_std=prev_noise_std,
    )


class TestEmulatorNoiseHook:
    """The hook must be inert by default and never fire outside training."""

    def test_disabled_by_default(self):
        model = _make_emulator()
        assert model._prev_noise is None
        batch = (torch.randn(8, 3), torch.randn(8, 1))
        assert model.on_after_batch_transfer(batch, 0) is batch

    def test_no_trainer_attached_is_safe(self):
        """_add_prev_noise must not blow up when the module has no Trainer."""
        model = _make_emulator({"sigWaveHeight": 0.1})
        model._prev_noise = build_prev_noise_injector(_FakeDataModule(), {"sigWaveHeight": 0.1})
        x = torch.randn(8, 3)
        assert torch.equal(model._add_prev_noise(x), x)

    def test_applied_during_training_only(self):
        torch.manual_seed(0)
        model = _make_emulator({"sigWaveHeight": 0.1})
        model._prev_noise = build_prev_noise_injector(_FakeDataModule(), {"sigWaveHeight": 0.1})
        x, y = torch.randn(64, 3), torch.randn(64, 1)

        model._trainer = _StubTrainer(training=True)
        noised_x, noised_y = model.on_after_batch_transfer((x, y), 0)
        assert not torch.equal(noised_x[:, 2], x[:, 2])
        assert torch.equal(noised_y, y)  # targets stay clean

        model._trainer = _StubTrainer(training=False)
        val_x, _ = model.on_after_batch_transfer((x, y), 0)
        assert torch.equal(val_x, x)

    def test_sample_weights_are_passed_through(self):
        model = _make_emulator({"sigWaveHeight": 0.1})
        model._prev_noise = build_prev_noise_injector(_FakeDataModule(), {"sigWaveHeight": 0.1})
        model._trainer = _StubTrainer(training=True)
        batch = (torch.randn(8, 3), torch.randn(8, 1), torch.rand(8, 1))
        out = model.on_after_batch_transfer(batch, 0)
        assert len(out) == 3
        assert torch.equal(out[2], batch[2])
