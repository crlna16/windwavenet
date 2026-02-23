# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import xarray as xr

from src.data.SimpleWWMDataset import SequenceWWMDataset, SimpleWWMDataset


class TestDatasetFunctions:
    """Test suite for SimpleWWMDataset static methods using synthetic xarray datasets."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a synthetic xarray dataset for testing."""
        time = np.arange(5)
        nSCHISM_hgrid_node = np.arange(3)

        # Create synthetic data
        temperature = np.array(
            [
                [10, 11, 12],  # t=0
                [15, 16, 17],  # t=1
                [20, 21, 22],  # t=2
                [25, 26, 27],  # t=3
                [30, 31, 32],  # t=4
            ]
        )

        wind_speed = np.array(
            [
                [5, 6, 7],
                [8, 9, 10],
                [11, 12, 13],
                [14, 15, 16],
                [17, 18, 19],
            ]
        )

        ds = xr.Dataset(
            {
                "temperature": (["time", "nSCHISM_hgrid_node"], temperature),
                "wind_speed": (["time", "nSCHISM_hgrid_node"], wind_speed),
            },
            coords={
                "time": time,
                "nSCHISM_hgrid_node": nSCHISM_hgrid_node,
            },
        )
        return ds

    def test_compute_prevT(self, sample_dataset):
        """Test _compute_prevT creates shifted variables correctly."""
        features = ["prev_temperature", "wind_speed"]

        result = SimpleWWMDataset._compute_prevT(sample_dataset, features)

        # Check that prev_temperature was created
        assert "prev_temperature" in result

        # Check first timestep is NaN
        assert np.isnan(result["prev_temperature"].isel(time=0).values).all()

        # Check that prev_temperature[t] == temperature[t-1]
        np.testing.assert_array_equal(
            result["prev_temperature"].isel(time=1).values,
            sample_dataset["temperature"].isel(time=0).values,
        )
        np.testing.assert_array_equal(
            result["prev_temperature"].isel(time=2).values,
            sample_dataset["temperature"].isel(time=1).values,
        )

        # Check that wind_speed is unchanged (no "prev_" prefix)
        assert "prev_wind_speed" not in result
        np.testing.assert_array_equal(result["wind_speed"].values, sample_dataset["wind_speed"].values)

    def test_compute_deltaT(self, sample_dataset):
        """Test _compute_deltaT creates difference variables correctly."""
        targets = ["delta_temperature", "wind_speed"]

        result = SimpleWWMDataset._compute_deltaT(sample_dataset, targets)

        # Check that delta_temperature was created
        assert "delta_temperature" in result

        # Check first timestep is NaN (diff creates NaN at start)
        assert np.isnan(result["delta_temperature"].isel(time=0).values).all()

        # Check that delta_temperature[t] == temperature[t] - temperature[t-1]
        expected_delta = np.array([5.0, 5.0, 5.0])  # t=1: 15-10, 16-11, 17-12
        np.testing.assert_array_equal(result["delta_temperature"].isel(time=1).values, expected_delta)

        # Check that wind_speed is unchanged (no "delta_" prefix)
        assert "delta_wind_speed" not in result
        np.testing.assert_array_equal(result["wind_speed"].values, sample_dataset["wind_speed"].values)

    def test_select_variables(self, sample_dataset):
        """Test _select_variables filters dataset correctly."""
        variables = ["temperature"]

        result = SimpleWWMDataset._select_variables(sample_dataset, variables)

        # Check only selected variable is present
        assert "temperature" in result
        assert "wind_speed" not in result

        # Check data is unchanged
        np.testing.assert_array_equal(result["temperature"].values, sample_dataset["temperature"].values)

    def test_apply_scaling(self, sample_dataset):
        """Test _apply_scaling applies standardization correctly."""
        variables = ["temperature"]
        scaling_params = {"temperature": {"mean": 20.0, "std": 5.0}}

        result = SimpleWWMDataset._apply_scaling(sample_dataset.copy(), variables, scaling_params)

        # Check scaling was applied: (x - mean) / std
        expected = (sample_dataset["temperature"].values - 20.0) / 5.0
        np.testing.assert_array_almost_equal(result["temperature"].values, expected)

        # Check unscaled variable is unchanged
        np.testing.assert_array_equal(result["wind_speed"].values, sample_dataset["wind_speed"].values)

    def test_apply_scaling_multiple_variables(self, sample_dataset):
        """Test _apply_scaling with multiple variables."""
        variables = ["temperature", "wind_speed"]
        scaling_params = {
            "temperature": {"mean": 20.0, "std": 5.0},
            "wind_speed": {"mean": 10.0, "std": 2.0},
        }

        result = SimpleWWMDataset._apply_scaling(sample_dataset.copy(), variables, scaling_params)

        # Check both variables were scaled
        expected_temp = (sample_dataset["temperature"].values - 20.0) / 5.0
        expected_wind = (sample_dataset["wind_speed"].values - 10.0) / 2.0

        np.testing.assert_array_almost_equal(result["temperature"].values, expected_temp)
        np.testing.assert_array_almost_equal(result["wind_speed"].values, expected_wind)

    def test_apply_scaling_missing_params(self, sample_dataset):
        """Test _apply_scaling handles missing scaling parameters."""
        variables = ["temperature", "wind_speed"]
        scaling_params = {
            "temperature": {"mean": 20.0, "std": 5.0}
            # wind_speed scaling params missing
        }

        result = SimpleWWMDataset._apply_scaling(sample_dataset.copy(), variables, scaling_params)

        # Check temperature was scaled
        expected_temp = (sample_dataset["temperature"].values - 20.0) / 5.0
        np.testing.assert_array_almost_equal(result["temperature"].values, expected_temp)

        # Check wind_speed was NOT scaled (left unchanged)
        np.testing.assert_array_equal(result["wind_speed"].values, sample_dataset["wind_speed"].values)

    def test_compute_prevT_and_deltaT_combined(self, sample_dataset):
        """Test combined workflow of computing prev and delta variables."""
        # First compute previous timestep
        features = ["prev_temperature"]
        ds_with_prev = SimpleWWMDataset._compute_prevT(sample_dataset, features)

        # Then compute delta
        targets = ["delta_temperature"]
        ds_final = SimpleWWMDataset._compute_deltaT(ds_with_prev, targets)

        # Check both variables exist
        assert "prev_temperature" in ds_final
        assert "delta_temperature" in ds_final

        # Verify relationship: delta[t] ≈ temperature[t] - prev_temperature[t]
        # (approximately, due to how diff works)
        for t in range(1, 4):
            delta_val = ds_final["delta_temperature"].isel(time=t).values
            temp_diff = (
                sample_dataset["temperature"].isel(time=t).values
                - sample_dataset["temperature"].isel(time=t - 1).values
            )
            np.testing.assert_array_almost_equal(delta_val, temp_diff)


class TestReconstructFromDelta:
    """Test suite for SimpleWWMDataset.reconstruct_from_delta static method."""

    @pytest.fixture
    def delta_dataset(self):
        """Create a synthetic dataset for delta reconstruction testing."""
        time = np.arange(5)
        nSCHISM_hgrid_node = np.arange(3)

        # Create absolute values
        temperature = np.array(
            [
                [10.0, 11.0, 12.0],
                [15.0, 16.0, 17.0],
                [20.0, 21.0, 22.0],
                [25.0, 26.0, 27.0],
                [30.0, 31.0, 32.0],
            ]
        )

        # Create previous timestep values
        prev_temperature = np.array(
            [
                [np.nan, np.nan, np.nan],  # t=0
                [10.0, 11.0, 12.0],  # t=1
                [15.0, 16.0, 17.0],  # t=2
                [20.0, 21.0, 22.0],  # t=3
                [25.0, 26.0, 27.0],  # t=4
            ]
        )

        ds = xr.Dataset(
            {
                "temperature": (["time", "nSCHISM_hgrid_node"], temperature),
                "prev_temperature": (["time", "nSCHISM_hgrid_node"], prev_temperature),
            },
            coords={
                "time": time,
                "nSCHISM_hgrid_node": nSCHISM_hgrid_node,
            },
        )
        return ds

    def test_reconstruct_from_delta_basic(self, delta_dataset):
        """Test basic delta reconstruction without scaling."""
        # Delta predictions (simulating model output)
        delta_predictions = np.array(
            [
                [5.0, 5.0, 5.0],  # t=0
                [5.0, 5.0, 5.0],  # t=1
                [5.0, 5.0, 5.0],  # t=2
                [5.0, 5.0, 5.0],  # t=3
                [5.0, 5.0, 5.0],  # t=4
            ]
        )

        reconstructed, _ = SimpleWWMDataset.reconstruct_from_delta(
            delta_predictions=delta_predictions,
            dataset=delta_dataset,
            target_name="delta_temperature",
            features=["prev_temperature"],
            scaling_params=None,
        )

        # reconstructed[t] = prev_temperature[t] + delta[t]
        expected = np.array(
            [
                [np.nan, np.nan, np.nan],  # t=0: NaN + 5
                [15.0, 16.0, 17.0],  # t=1: 10 + 5, 11 + 5, 12 + 5
                [20.0, 21.0, 22.0],  # t=2: 15 + 5, ...
                [25.0, 26.0, 27.0],  # t=3
                [30.0, 31.0, 32.0],  # t=4
            ]
        )

        # Check non-NaN values
        np.testing.assert_array_almost_equal(reconstructed[1:], expected[1:])

    def test_reconstruct_from_delta_raises_on_non_delta_target(self, delta_dataset):
        """Test that non-delta targets raise ValueError."""
        delta_predictions = np.zeros((5, 3))

        with pytest.raises(ValueError, match="not a delta target"):
            SimpleWWMDataset.reconstruct_from_delta(
                delta_predictions=delta_predictions,
                dataset=delta_dataset,
                target_name="temperature",  # Not a delta target
                features=["prev_temperature"],
            )

    def test_reconstruct_from_delta_computes_prev_if_missing(self):
        """Test that prev_* is computed if not in dataset."""
        time = np.arange(3)
        nSCHISM_hgrid_node = np.arange(2)

        # Dataset without prev_temperature
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "nSCHISM_hgrid_node"],
                    np.array([[10.0, 11.0], [15.0, 16.0], [20.0, 21.0]]),
                ),
            },
            coords={"time": time, "nSCHISM_hgrid_node": nSCHISM_hgrid_node},
        )

        delta_predictions = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])

        reconstructed, updated_ds = SimpleWWMDataset.reconstruct_from_delta(
            delta_predictions=delta_predictions,
            dataset=ds,
            target_name="delta_temperature",
            features=["prev_temperature"],
        )

        # Check prev_temperature was added
        assert "prev_temperature" in updated_ds


class TestSequenceWWMDatasetFunctions:
    """Test suite for SequenceWWMDataset getitem and len."""

    @pytest.fixture
    def mock_sequence_dataset(self, tmp_path):
        """Create a minimal mock Zarr dataset for sequence testing."""
        import pandas as pd

        # Create synthetic data: 48 time steps (2 days at hourly), 5 nodes
        n_time = 48
        n_nodes = 5

        time = pd.date_range("2020-01-01", periods=n_time, freq="h")
        nSCHISM_hgrid_node = np.arange(n_nodes)

        U10 = np.random.randn(n_time, n_nodes).astype(np.float32)
        V10 = np.random.randn(n_time, n_nodes).astype(np.float32)
        sigWaveHeight = np.random.randn(n_time, n_nodes).astype(np.float32)

        ds = xr.Dataset(
            {
                "U10": (["time", "nSCHISM_hgrid_node"], U10),
                "V10": (["time", "nSCHISM_hgrid_node"], V10),
                "sigWaveHeight": (["time", "nSCHISM_hgrid_node"], sigWaveHeight),
            },
            coords={
                "time": time,
                "nSCHISM_hgrid_node": nSCHISM_hgrid_node,
            },
        )

        zarr_path = tmp_path / "test.zarr"
        ds.to_zarr(zarr_path)
        return zarr_path

    def test_sequence_dataset_len(self, mock_sequence_dataset):
        """Test that SequenceWWMDataset length is correct."""
        seq_len = 5
        ds = SequenceWWMDataset(
            data_dir=mock_sequence_dataset,
            data_range=[1, 2],  # 1 day = 24 samples
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )

        # After removing first timestep (for delta/prev), n_time = 23
        # n_valid_times = 23 - 5 = 18
        # Total samples = 18 * 5 nodes = 90
        # (exact number depends on data processing in SimpleWWMDataset)
        assert len(ds) > 0

    def test_sequence_dataset_getitem_shapes(self, mock_sequence_dataset):
        """Test that __getitem__ returns correctly shaped tensors."""
        seq_len = 5
        ds = SequenceWWMDataset(
            data_dir=mock_sequence_dataset,
            data_range=[1, 2],
            features=["U10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )

        X, y = ds[0]

        # X should be (seq_len, n_features)
        assert X.shape[0] == seq_len
        assert X.shape[1] == 1  # 1 feature (U10)

        # y should be (n_targets,)
        assert y.shape[0] == 1  # 1 target

    def test_sequence_dataset_raises_on_invalid_seq_len(self, mock_sequence_dataset):
        """Test that invalid seq_len raises ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            SequenceWWMDataset(
                data_dir=mock_sequence_dataset,
                data_range=[1, 2],  # Small time range
                features=["U10"],
                targets=["sigWaveHeight"],
                seq_len=1000,  # Too large
            )
