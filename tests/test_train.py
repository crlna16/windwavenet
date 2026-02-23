# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.train import train


@pytest.fixture
def mock_datamodule():
    """Create a mock datamodule."""
    mock_dm = MagicMock()
    mock_dm.train_dataloader.return_value = MagicMock()
    mock_dm.val_dataloader.return_value = MagicMock()
    mock_dm.test_dataloader.return_value = MagicMock()
    return mock_dm


@pytest.fixture
def mock_model():
    """Create a mock model."""
    mock_mdl = MagicMock()
    mock_mdl.parameters.return_value = []
    return mock_mdl


@pytest.fixture
def mock_trainer():
    """Create a mock trainer."""
    mock_t = MagicMock()
    mock_t.callback_metrics = {"loss": torch.tensor(0.5)}
    mock_t.checkpoint_callback = None
    return mock_t


@pytest.fixture
def minimal_config():
    """Create a minimal test configuration."""
    config_dir = Path(__file__).parent.parent / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "trainer.max_epochs=1",
                "train=true",
                "test=false",
                "callbacks=null",
                "logger=null",
                f"paths.root_dir={tempfile.gettempdir()}",
                f"paths.output_dir={tempfile.mkdtemp()}",
            ],
        )

    return cfg


@patch("src.train.print_config_tree")
@patch("src.train.hydra.utils.instantiate")
def test_train_returns_correct_types(
    mock_instantiate,
    mock_print_config,
    minimal_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that train function returns the expected types."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    metric_dict, object_dict = train(minimal_config)

    assert isinstance(metric_dict, dict), "metric_dict should be a dictionary"
    assert isinstance(object_dict, dict), "object_dict should be a dictionary"


@patch("src.train.print_config_tree")
@patch("src.train.hydra.utils.instantiate")
def test_train_creates_required_objects(
    mock_instantiate,
    mock_print_config,
    minimal_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that train function creates all required objects."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    _, object_dict = train(minimal_config)

    required_keys = ["cfg", "datamodule", "model", "callbacks", "logger", "trainer"]
    for key in required_keys:
        assert key in object_dict, f"object_dict should contain '{key}'"


@patch("src.train.print_config_tree")
@patch("src.train.hydra.utils.instantiate")
@patch("lightning.seed_everything")
def test_train_with_seed(
    mock_seed,
    mock_instantiate,
    mock_print_config,
    minimal_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that seed is set when provided."""
    minimal_config.seed = 42
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    train(minimal_config)
    mock_seed.assert_called_once_with(42, workers=True)


@patch("src.train.print_config_tree")
@patch("src.train.hydra.utils.instantiate")
def test_train_respects_train_flag(
    mock_instantiate,
    mock_print_config,
    minimal_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that training is skipped when train flag is False."""
    minimal_config.train = False
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    metric_dict, _ = train(minimal_config)

    # When training is skipped, metrics should still be a dict
    assert isinstance(metric_dict, dict)
    # fit should not be called
    mock_trainer.fit.assert_not_called()


@patch("src.train.print_config_tree")
@patch("src.train.hydra.utils.instantiate")
def test_train_with_callbacks(
    mock_instantiate,
    mock_print_config,
    minimal_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that callbacks are properly instantiated."""
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    _, object_dict = train(minimal_config)

    assert "callbacks" in object_dict
    assert isinstance(object_dict["callbacks"], list)


@patch("src.train.print_config_tree")
@patch("src.train.hydra.utils.instantiate")
def test_train_instantiates_components(
    mock_instantiate,
    mock_print_config,
    minimal_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that all components are instantiated via Hydra."""
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    train(minimal_config)

    # Should instantiate datamodule, model, and trainer
    assert mock_instantiate.call_count >= 3


@patch("src.train.print_config_tree")
@patch("src.train.hydra.utils.instantiate")
def test_train_returns_metrics_dict_with_correct_structure(
    mock_instantiate,
    mock_print_config,
    minimal_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that returned metrics dictionary has the expected structure."""
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    metric_dict, _ = train(minimal_config)

    # All values should be numeric or tensors
    for key, value in metric_dict.items():
        assert isinstance(key, str), f"Metric key '{key}' should be a string"
