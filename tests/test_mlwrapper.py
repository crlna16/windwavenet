# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from src.model.MLWrapper import MLModelWrapper


class DummyEstimator:
    """Small sklearn-like estimator for wrapper tests."""

    def __init__(self, n_outputs: int = 1, return_value: float = 0.0):
        self.n_outputs = n_outputs
        self.return_value = return_value
        self.fit_calls = 0
        self.last_fit_kwargs = None

    def fit(self, X, y, sample_weight=None, eval_set=None, verbose=True, callbacks=None):
        self.fit_calls += 1
        self.last_fit_kwargs = {
            "sample_weight": sample_weight,
            "eval_set": eval_set,
            "verbose": verbose,
            "callbacks": callbacks,
        }
        return self

    def predict(self, X):
        return np.full((len(X), self.n_outputs), self.return_value, dtype=np.float64)


class ImportanceEstimator(DummyEstimator):
    def __init__(self, feature_importances):
        super().__init__(n_outputs=1, return_value=0.0)
        self.feature_importances_ = np.asarray(feature_importances, dtype=np.float64)


@pytest.fixture
def two_target_data():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    return X, y


def test_init_sets_expected_attributes():
    model = DummyEstimator(n_outputs=2)
    wrapper = MLModelWrapper(model=model, target_names=["a", "b"])

    assert wrapper.model is model
    assert wrapper.target_names == ["a", "b"]
    assert wrapper.num_targets == 2
    assert wrapper.target_bins == {}
    assert wrapper.callbacks == []


def test_fit_passes_eval_set_and_sample_weight(two_target_data):
    X_train, y_train = two_target_data
    X_val = X_train[:2]
    y_val = y_train[:2]
    sample_weight = np.array([1.0, 2.0, 1.0, 2.0])

    model = DummyEstimator(n_outputs=2)
    wrapper = MLModelWrapper(model=model, target_names=["a", "b"])

    metrics = wrapper.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight=sample_weight,
    )

    assert model.fit_calls == 1
    assert model.last_fit_kwargs["sample_weight"] is sample_weight
    assert model.last_fit_kwargs["eval_set"] == [(X_val, y_val)]
    assert model.last_fit_kwargs["verbose"] is True
    assert model.last_fit_kwargs["callbacks"] is None
    assert "train/loss" in metrics
    assert "val/loss" in metrics
    assert "train/rmse_a" in metrics
    assert "val/rmse_b" in metrics


def test_fit_without_validation_sets_eval_set_none(two_target_data):
    X_train, y_train = two_target_data

    model = DummyEstimator(n_outputs=2)
    wrapper = MLModelWrapper(model=model, target_names=["a", "b"])

    metrics = wrapper.fit(X_train=X_train, y_train=y_train)

    assert model.last_fit_kwargs["eval_set"] is None
    assert model.last_fit_kwargs["callbacks"] is None
    assert "train/loss" in metrics
    assert "val/loss" not in metrics


def test_fit_passes_callbacks_to_model(two_target_data):
    X_train, y_train = two_target_data
    sentinel = object()

    model = DummyEstimator(n_outputs=2)
    wrapper = MLModelWrapper(model=model, target_names=["a", "b"], callbacks=[sentinel])

    wrapper.fit(X_train=X_train, y_train=y_train)

    assert model.last_fit_kwargs["callbacks"] == [sentinel]


def test_predict_delegates_to_model(two_target_data):
    X, _ = two_target_data
    model = DummyEstimator(n_outputs=2, return_value=7.5)
    wrapper = MLModelWrapper(model=model, target_names=["a", "b"])

    preds = wrapper.predict(X)

    assert preds.shape == (len(X), 2)
    assert np.allclose(preds, 7.5)


def test_evaluate_uses_prefix_and_returns_metrics(two_target_data):
    X, y = two_target_data
    model = DummyEstimator(n_outputs=2, return_value=0.0)
    wrapper = MLModelWrapper(model=model, target_names=["a", "b"])

    metrics = wrapper.evaluate(X, y, prefix="holdout")

    assert "holdout/loss" in metrics
    assert "holdout/rmse" in metrics
    assert "holdout/loss_a" in metrics
    assert "holdout/rmse_b" in metrics


def test_compute_metrics_handles_1d_arrays():
    model = DummyEstimator(n_outputs=1, return_value=0.0)
    wrapper = MLModelWrapper(model=model, target_names=["target"])

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])

    metrics = wrapper._compute_metrics(y_true, y_pred, prefix="t")

    expected_mse = ((0.0**2) + (0.0**2) + (2.0**2)) / 3.0
    assert metrics["t/loss"] == pytest.approx(expected_mse)
    assert metrics["t/loss_target"] == pytest.approx(expected_mse)
    assert metrics["t/rmse"] == pytest.approx(np.sqrt(expected_mse))


def test_compute_metrics_adds_binned_metrics():
    model = DummyEstimator(n_outputs=1)
    wrapper = MLModelWrapper(
        model=model,
        target_names=["sigWaveHeight"],
        target_bins={"sigWaveHeight": [1.0, 2.0]},
    )

    y_true = np.array([[0.5], [1.5], [2.5]])
    y_pred = np.array([[0.0], [1.0], [2.0]])

    metrics = wrapper._compute_metrics(y_true, y_pred, prefix="test")

    assert "test/rmse_sigWaveHeight_<1.0" in metrics
    assert "test/rmse_sigWaveHeight_[1.0,2.0)" in metrics
    assert "test/rmse_sigWaveHeight_>=2.0" in metrics


def test_save_and_load_roundtrip(tmp_path):
    model = DummyEstimator(n_outputs=1, return_value=3.14)
    wrapper = MLModelWrapper(model=model, target_names=["target"])

    path = tmp_path / "models" / "mlwrapper.joblib"
    wrapper.save(str(path))
    assert path.exists()

    wrapper.model = DummyEstimator(n_outputs=1, return_value=-99.0)
    wrapper.load(str(path))

    preds = wrapper.predict(np.zeros((2, 3)))
    assert np.allclose(preds, 3.14)


def test_print_feature_importance_paths_do_not_raise():
    wrapper_no_attr = MLModelWrapper(model=DummyEstimator(), target_names=["target"])
    wrapper_no_attr.print_feature_importance(["f1", "f2"])

    wrapper_mismatch = MLModelWrapper(model=ImportanceEstimator([0.7, 0.3]), target_names=["target"])
    wrapper_mismatch.print_feature_importance(["f1"])

    wrapper_ok = MLModelWrapper(model=ImportanceEstimator([0.2, 0.8]), target_names=["target"])
    wrapper_ok.print_feature_importance(["f1", "f2"])
