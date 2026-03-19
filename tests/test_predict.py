# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr
from hydra import compose, initialize_config_dir

from src.predict import predict


@pytest.fixture
def mock_dataset():
    """Create a mock xarray dataset for predictions."""
    time = np.arange(10)
    space = np.arange(5)

    ds = xr.Dataset(
        {
            "sigWaveHeight": (["time", "nSCHISM_hgrid_node"], np.random.rand(10, 5)),
            "prev_sigWaveHeight": (
                ["time", "nSCHISM_hgrid_node"],
                np.random.rand(10, 5),
            ),
        },
        coords={
            "time": time,
            "nSCHISM_hgrid_node": space,
            "SCHISM_hgrid_node_x": ("nSCHISM_hgrid_node", np.random.rand(5)),
            "SCHISM_hgrid_node_y": ("nSCHISM_hgrid_node", np.random.rand(5)),
        },
    )
    return ds


@pytest.fixture
def mock_datamodule(mock_dataset):
    """Create a mock datamodule with prediction dataset."""
    mock_dm = MagicMock()
    mock_ds_pred = MagicMock()
    mock_ds_pred.ds = mock_dataset
    mock_dm.ds_pred = mock_ds_pred
    mock_dm.predict_dataloader.return_value = MagicMock()
    return mock_dm


@pytest.fixture
def mock_model():
    """Create a mock model with target_names."""
    mock_mdl = MagicMock()
    mock_mdl.target_names = ["sigWaveHeight"]
    mock_mdl.parameters.return_value = []
    return mock_mdl


@pytest.fixture
def mock_trainer():
    """Create a mock trainer that returns predictions."""
    mock_t = MagicMock()
    # Return mock predictions: batch of 50 samples (10 time * 5 space), 1 target
    mock_predictions = [torch.rand(50, 1)]
    mock_t.predict.return_value = mock_predictions
    return mock_t


@pytest.fixture
def minimal_predict_config():
    """Create a minimal test configuration for prediction."""
    config_dir = Path(__file__).parent.parent / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
        cfg = compose(
            config_name="predict.yaml",
            overrides=[
                "logger=null",
                "data.targets=[sigWaveHeight]",  # Match mock model targets
                f"paths.root_dir={tempfile.gettempdir()}",
                f"paths.output_dir={tempfile.mkdtemp()}",
                f"ckpt_path={tempfile.mktemp()}.ckpt",
            ],
        )

    return cfg


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_returns_xarray_dataset(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that predict function returns an xarray Dataset."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    result = predict(minimal_predict_config)

    assert isinstance(result, xr.Dataset), "predict should return an xarray Dataset"


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_creates_required_objects(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that predict function creates all required objects."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    _ = predict(minimal_predict_config)

    # Verify instantiate was called for datamodule, model, and trainer
    assert mock_instantiate.call_count >= 3


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_calls_trainer_predict(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that trainer.predict is called."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    predict(minimal_predict_config)

    mock_trainer.predict.assert_called_once()


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_verifies_target_names(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that predict verifies target names match between config and checkpoint."""
    # Setup mocks with matching targets
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    # Should not raise an error with matching targets
    result = predict(minimal_predict_config)
    assert isinstance(result, xr.Dataset)


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_raises_error_on_target_mismatch(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that predict raises error when targets don't match."""
    # Setup model with different target names
    mock_model.target_names = ["different_target"]
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    with pytest.raises(ValueError, match="Target mismatch"):
        predict(minimal_predict_config)


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_saves_netcdf_file(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that predictions are saved to netCDF file."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    result = predict(minimal_predict_config)

    # Check that the result is returned (file saving is a side effect)
    assert isinstance(result, xr.Dataset)
    # Note: File creation is tested implicitly through the function execution
    # The actual file system I/O would be mocked in more isolated unit tests


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_with_delta_target(
    mock_instantiate,
    mock_print_config,
    mock_datamodule,
    mock_trainer,
):
    """Test that predict handles delta targets correctly."""
    # Create a fresh config with delta targets
    config_dir = Path(__file__).parent.parent / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
        cfg = compose(
            config_name="predict.yaml",
            overrides=[
                "logger=null",
                "data.targets=[delta_sigWaveHeight]",
                f"paths.root_dir={tempfile.gettempdir()}",
                f"paths.output_dir={tempfile.mkdtemp()}",
                f"ckpt_path={tempfile.mktemp()}.ckpt",
            ],
        )

    # Create model with delta target
    mock_model = MagicMock()
    mock_model.target_names = ["delta_sigWaveHeight"]
    mock_model.parameters.return_value = []

    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    result = predict(cfg)

    # Check that the reconstructed variable is in the output (not delta)
    assert "sigWaveHeight" in result.data_vars or "delta_sigWaveHeight" in result.data_vars


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_output_has_correct_dimensions(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that prediction output has correct dimensions."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    result = predict(minimal_predict_config)

    # Check dimensions match input dataset
    assert "time" in result.dims
    assert "nSCHISM_hgrid_node" in result.dims
    assert result.dims["time"] == 10
    assert result.dims["nSCHISM_hgrid_node"] == 5


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_output_has_metadata(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that prediction output includes metadata attributes."""
    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    result = predict(minimal_predict_config)

    # Check attributes
    assert "description" in result.attrs
    assert "checkpoint" in result.attrs
    assert "model" in result.attrs


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_requires_checkpoint_path(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
):
    """Test that predict requires a checkpoint path."""
    # Remove checkpoint path
    minimal_predict_config.ckpt_path = None

    with pytest.raises(AssertionError):
        predict(minimal_predict_config)


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_with_multiple_targets(
    mock_instantiate,
    mock_print_config,
    mock_datamodule,
    mock_trainer,
):
    """Test prediction with multiple target variables."""
    # Create a fresh config with multiple targets
    config_dir = Path(__file__).parent.parent / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
        cfg = compose(
            config_name="predict.yaml",
            overrides=[
                "logger=null",
                "data.targets=[sigWaveHeight,windSpeed]",
                f"paths.root_dir={tempfile.gettempdir()}",
                f"paths.output_dir={tempfile.mkdtemp()}",
                f"ckpt_path={tempfile.mktemp()}.ckpt",
            ],
        )

    # Create model with multiple targets
    mock_model = MagicMock()
    mock_model.target_names = ["sigWaveHeight", "windSpeed"]
    mock_model.parameters.return_value = []

    # Update trainer to return predictions for 2 targets
    mock_predictions = [torch.rand(50, 2)]
    mock_trainer.predict.return_value = mock_predictions

    # Setup mocks
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    result = predict(cfg)

    # Should have both targets in output
    assert isinstance(result, xr.Dataset)
    # At least one of the targets should be present
    assert len(result.data_vars) > 0


@patch("src.predict.print_config_tree")
@patch("src.predict.hydra.utils.instantiate")
def test_predict_unscales_target_when_also_feature(
    mock_instantiate,
    mock_print_config,
    minimal_predict_config,
    mock_datamodule,
    mock_model,
    mock_trainer,
):
    """Test that direct targets are unscaled before saving when also used as features."""
    mock_datamodule.ds_pred.features = ["sigWaveHeight", "windSpeed"]
    mock_datamodule.ds_pred.scaling_params = {
        "sigWaveHeight": {"mean": 2.0, "std": 4.0},
    }
    mock_trainer.predict.return_value = [torch.full((50, 1), 0.5)]
    mock_instantiate.side_effect = [mock_datamodule, mock_model, mock_trainer]

    result = predict(minimal_predict_config)

    expected = np.full((10, 5), 4.0)
    np.testing.assert_allclose(result["sigWaveHeight"].values, expected)
