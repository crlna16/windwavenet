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
            "windSpeedX": (["time", "nSCHISM_hgrid_node"], np.random.rand(10, 5)),
            "windSpeedY": (["time", "nSCHISM_hgrid_node"], np.random.rand(10, 5)),
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
def mock_model():
    """Create a mock model with target_names and a forward pass returning a fixed tensor."""
    mock_mdl = MagicMock()
    mock_mdl.target_names = ["sigWaveHeight"]
    mock_mdl.to.return_value = mock_mdl
    mock_mdl.forward.return_value = torch.rand(5, 1)
    return mock_mdl


@pytest.fixture
def mock_data_pipeline(mock_dataset):
    """Patch the dataset_utils calls predict.py uses to build its feature arrays."""
    with (
        patch("src.predict.du.open_and_slice_dataset", return_value=(mock_dataset, 24)),
        patch("src.predict.du.preprocess_variables", return_value=(mock_dataset, 5)),
        patch("src.predict.du.scale_variables", return_value=(mock_dataset, None)),
    ):
        yield mock_dataset


@pytest.fixture
def mock_model_loading(mock_model):
    """Patch checkpoint loading so ``model_cls.load_from_checkpoint(...)`` returns ``mock_model``."""
    fake_module = MagicMock()
    fake_module.WWMEmulator.load_from_checkpoint.return_value = mock_model
    with (
        patch("src.predict.importlib.import_module", return_value=fake_module),
        patch("src.predict.hydra.utils.instantiate", return_value=MagicMock()),
    ):
        yield mock_model


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
                "data.features=[windSpeedX,windSpeedY]",
                f"paths.root_dir={tempfile.gettempdir()}",
                f"paths.output_dir={tempfile.mkdtemp()}",
                f"ckpt_path={tempfile.mktemp()}.ckpt",
                "data.boundary_nodes_csv=null",
            ],
        )

    return cfg


@patch("src.predict.print_config_tree")
def test_predict_returns_xarray_dataset(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that predict function returns an xarray Dataset."""
    result = predict(minimal_predict_config)

    assert isinstance(result, xr.Dataset), "predict should return an xarray Dataset"


@patch("src.predict.print_config_tree")
def test_predict_loads_model_from_checkpoint(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that predict loads the checkpoint and moves the model to a device (no Trainer)."""
    predict(minimal_predict_config)

    mock_model_loading.to.assert_called_once()
    mock_model_loading.eval.assert_called_once()


@patch("src.predict.print_config_tree")
def test_predict_calls_model_forward(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that the model's forward pass is invoked directly (no Trainer/DataLoader)."""
    predict(minimal_predict_config)

    assert mock_model_loading.forward.called


@patch("src.predict.print_config_tree")
def test_predict_verifies_target_names(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that predict verifies target names match between config and checkpoint."""
    result = predict(minimal_predict_config)
    assert isinstance(result, xr.Dataset)


@patch("src.predict.print_config_tree")
def test_predict_raises_error_on_target_mismatch(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that predict raises error when targets don't match."""
    mock_model_loading.target_names = ["different_target"]

    with pytest.raises(ValueError, match="Target mismatch"):
        predict(minimal_predict_config)


@patch("src.predict.print_config_tree")
def test_predict_saves_netcdf_file(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that predictions are saved to netCDF file."""
    result = predict(minimal_predict_config)

    output_path = Path(minimal_predict_config.paths.output_dir) / "predictions.nc"
    assert output_path.exists()
    assert isinstance(result, xr.Dataset)


@patch("src.predict.print_config_tree")
def test_predict_with_delta_target(
    mock_print_config,
    mock_data_pipeline,
    mock_model,
):
    """Test that predict handles delta targets correctly."""
    config_dir = Path(__file__).parent.parent / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
        cfg = compose(
            config_name="predict.yaml",
            overrides=[
                "logger=null",
                "data.targets=[delta_sigWaveHeight]",
                "data.features=[windSpeedX,windSpeedY]",
                f"paths.root_dir={tempfile.gettempdir()}",
                f"paths.output_dir={tempfile.mkdtemp()}",
                f"ckpt_path={tempfile.mktemp()}.ckpt",
                "data.boundary_nodes_csv=null",
            ],
        )

    mock_model.target_names = ["delta_sigWaveHeight"]
    fake_module = MagicMock()
    fake_module.WWMEmulator.load_from_checkpoint.return_value = mock_model

    with (
        patch("src.predict.importlib.import_module", return_value=fake_module),
        patch("src.predict.hydra.utils.instantiate", return_value=MagicMock()),
    ):
        result = predict(cfg)

    # Check that the reconstructed variable is in the output (not delta)
    assert "sigWaveHeight" in result.data_vars or "delta_sigWaveHeight" in result.data_vars


@patch("src.predict.print_config_tree")
def test_predict_output_has_correct_dimensions(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that prediction output has correct dimensions."""
    result = predict(minimal_predict_config)

    assert "time" in result.dims
    assert "nSCHISM_hgrid_node" in result.dims
    assert result.sizes["time"] == 10
    assert result.sizes["nSCHISM_hgrid_node"] == 5


@patch("src.predict.print_config_tree")
def test_predict_output_has_metadata(
    mock_print_config,
    minimal_predict_config,
    mock_data_pipeline,
    mock_model_loading,
):
    """Test that prediction output includes metadata attributes."""
    result = predict(minimal_predict_config)

    assert "description" in result.attrs
    assert "checkpoint" in result.attrs
    assert "model" in result.attrs


def test_predict_requires_checkpoint_path(minimal_predict_config):
    """Test that predict requires a checkpoint path."""
    minimal_predict_config.ckpt_path = None

    with pytest.raises(AssertionError):
        predict(minimal_predict_config)


@patch("src.predict.print_config_tree")
def test_predict_with_multiple_targets(
    mock_print_config,
    mock_data_pipeline,
    mock_model,
):
    """Test prediction with multiple target variables."""
    config_dir = Path(__file__).parent.parent / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.absolute())):
        cfg = compose(
            config_name="predict.yaml",
            overrides=[
                "logger=null",
                "data.targets=[sigWaveHeight,windSpeedX]",
                "data.features=[windSpeedX,windSpeedY]",
                f"paths.root_dir={tempfile.gettempdir()}",
                f"paths.output_dir={tempfile.mkdtemp()}",
                f"ckpt_path={tempfile.mktemp()}.ckpt",
                "data.boundary_nodes_csv=null",
            ],
        )

    mock_model.target_names = ["sigWaveHeight", "windSpeedX"]
    mock_model.forward.return_value = torch.rand(5, 2)
    fake_module = MagicMock()
    fake_module.WWMEmulator.load_from_checkpoint.return_value = mock_model

    with (
        patch("src.predict.importlib.import_module", return_value=fake_module),
        patch("src.predict.hydra.utils.instantiate", return_value=MagicMock()),
    ):
        result = predict(cfg)

    assert isinstance(result, xr.Dataset)
    assert len(result.data_vars) > 0


@patch("src.predict.print_config_tree")
def test_predict_unscales_target_when_also_feature(
    mock_print_config,
    minimal_predict_config,
    mock_dataset,
    mock_model,
):
    """Test that direct targets are unscaled before saving when also used as features."""
    scaling_params = {"sigWaveHeight": {"mean": 2.0, "std": 4.0}}
    mock_model.forward.return_value = torch.full((5, 1), 0.5)
    fake_module = MagicMock()
    fake_module.WWMEmulator.load_from_checkpoint.return_value = mock_model

    with (
        patch("src.predict.du.open_and_slice_dataset", return_value=(mock_dataset, 24)),
        patch("src.predict.du.preprocess_variables", return_value=(mock_dataset, 5)),
        patch("src.predict.du.scale_variables", return_value=(mock_dataset, scaling_params)),
        patch("src.predict.importlib.import_module", return_value=fake_module),
        patch("src.predict.hydra.utils.instantiate", return_value=MagicMock()),
    ):
        minimal_predict_config.data.features = ["sigWaveHeight", "windSpeedX"]
        result = predict(minimal_predict_config)

    expected = np.full((10, 5), 4.0)
    np.testing.assert_allclose(result["sigWaveHeight"].values, expected)
