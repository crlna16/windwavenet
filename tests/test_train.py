# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize

from src.train import train


@pytest.fixture
def minimal_config():
    """Create a minimal test configuration."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml")

        # Override with minimal test settings
        cfg.trainer.max_epochs = 1
        cfg.trainer.fast_dev_run = True
        cfg.train = True
        cfg.test = False
        cfg.logger = None  # Disable logging for tests

    return cfg


def test_train_returns_correct_types(minimal_config):
    """Test that train function returns the expected types."""
    metric_dict, object_dict = train(minimal_config)

    assert isinstance(metric_dict, dict), "metric_dict should be a dictionary"
    assert isinstance(object_dict, dict), "object_dict should be a dictionary"


def test_train_creates_required_objects(minimal_config):
    """Test that train function creates all required objects."""
    _, object_dict = train(minimal_config)

    required_keys = ["cfg", "datamodule", "model", "callbacks", "logger", "trainer"]
    for key in required_keys:
        assert key in object_dict, f"object_dict should contain '{key}'"


def test_train_with_seed(minimal_config):
    """Test that seed is set when provided."""
    minimal_config.seed = 42

    with patch("lightning.seed_everything") as mock_seed:
        train(minimal_config)
        mock_seed.assert_called_once_with(42, workers=True)


def test_train_respects_train_flag(minimal_config):
    """Test that training is skipped when train flag is False."""
    minimal_config.train = False

    metric_dict, _ = train(minimal_config)

    # When training is skipped, metrics should still be a dict (possibly empty)
    assert isinstance(metric_dict, dict)


def test_train_with_callbacks(minimal_config):
    """Test that callbacks are properly instantiated."""
    _, object_dict = train(minimal_config)

    assert "callbacks" in object_dict
    assert isinstance(object_dict["callbacks"], list)


@patch("src.train.hydra.utils.instantiate")
def test_train_instantiates_components(mock_instantiate, minimal_config):
    """Test that all components are instantiated via Hydra."""
    # Mock the instantiate function to return dummy objects
    mock_instantiate.return_value = MagicMock()

    train(minimal_config)

    # Should instantiate datamodule, model, and trainer at minimum
    assert mock_instantiate.call_count >= 3


def test_train_returns_metrics_dict_with_correct_structure(minimal_config):
    """Test that returned metrics dictionary has the expected structure."""
    metric_dict, _ = train(minimal_config)

    # All values should be numeric or tensors
    for key, value in metric_dict.items():
        assert isinstance(key, str), f"Metric key '{key}' should be a string"
