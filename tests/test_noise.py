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


CIRCLE_FEATURES = ["windSpeedX", "prev_sigWaveHeight", "prev_sin_meanWaveDirection", "prev_cos_meanWaveDirection"]
# prev_sigWaveHeight: mean 1.0, std 0.5; sin/cos unscaled (mean 0, std 1)
CIRCLE_FEEDBACK_MAP = [(1, 0, 1.0, 0.5), (2, 1, 0.0, 1.0), (3, 2, 0.0, 1.0)]
CONSTRAINTS = {
    "sigWaveHeight": "positive",
    "sin_meanWaveDirection": "unit_circle",
    "cos_meanWaveDirection": "unit_circle",
}


class TestConstrainedNoise:
    def _injector(self, constraints=CONSTRAINTS, noise_std=None):
        noise_std = noise_std or {"sigWaveHeight": 2.0, "sin_meanWaveDirection": 0.5, "cos_meanWaveDirection": 0.5}
        return PrevNoiseInjector(CIRCLE_FEATURES, noise_std, CIRCLE_FEEDBACK_MAP, constraints)

    def _clean_batch(self, n=10_000):
        x = torch.zeros(n, 4)
        x[:, 1] = (0.2 - 1.0) / 0.5  # 0.2 m of wave height, z-scored
        x[:, 2], x[:, 3] = 0.6, 0.8
        return x

    def test_positive_channel_is_clamped_at_zero(self):
        """Large noise on a small wave height must not produce negative physical values."""
        torch.manual_seed(0)
        out = self._injector()(self._clean_batch())
        swh = out[:, 1] * 0.5 + 1.0
        assert swh.min().item() == pytest.approx(0.0, abs=1e-6)
        assert (swh > 0).any()

    def test_negative_channel_is_clamped_at_zero(self):
        torch.manual_seed(0)
        injector = self._injector(constraints={"sigWaveHeight": "negative"})
        swh = injector(self._clean_batch())[:, 1] * 0.5 + 1.0
        assert swh.max().item() == pytest.approx(0.0, abs=1e-6)

    def test_sin_cos_pair_stays_on_unit_circle(self):
        torch.manual_seed(0)
        out = self._injector()(self._clean_batch())
        norm = (out[:, 2] ** 2 + out[:, 3] ** 2).sqrt()
        torch.testing.assert_close(norm, torch.ones_like(norm))
        assert not torch.allclose(out[:, 2], torch.full_like(out[:, 2], 0.6))

    def test_pair_is_projected_when_only_one_channel_is_noised(self):
        torch.manual_seed(0)
        out = self._injector(noise_std={"sin_meanWaveDirection": 0.5})(self._clean_batch())
        norm = (out[:, 2] ** 2 + out[:, 3] ** 2).sqrt()
        torch.testing.assert_close(norm, torch.ones_like(norm))

    def test_sequence_input(self):
        torch.manual_seed(0)
        x = self._clean_batch(64).unsqueeze(1).repeat(1, 6, 1)
        out = self._injector()(x)
        assert out.shape == x.shape
        assert (out[..., 1] * 0.5 + 1.0).min().item() >= -1e-6

    def test_without_constraints_noise_is_unconstrained(self):
        """Default behaviour of earlier runs: no clamping, no projection."""
        torch.manual_seed(0)
        injector = self._injector(constraints=None)
        assert not injector.circle_pairs
        out = injector(self._clean_batch())
        assert (out[:, 1] * 0.5 + 1.0).min().item() < 0
        norm = (out[:, 2] ** 2 + out[:, 3] ** 2).sqrt()
        assert not torch.allclose(norm, torch.ones_like(norm))

    def test_clean_channels_are_untouched(self):
        torch.manual_seed(0)
        x = self._clean_batch()
        x[:, 0] = torch.randn(x.shape[0])
        out = self._injector()(x)
        assert torch.equal(out[:, 0], x[:, 0])


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

    def test_output_constraints_are_passed_through(self):
        injector = build_prev_noise_injector(_FakeDataModule(), {"sigWaveHeight": 0.1}, {"sigWaveHeight": "positive"})
        assert injector.lower.item() == pytest.approx((0.0 - 1.0) / 0.5)


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
