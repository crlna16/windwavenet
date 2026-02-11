# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import xarray as xr

from src.data.SimpleWWMDataset import SimpleWWMDataset


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
        np.testing.assert_array_equal(
            result["wind_speed"].values, sample_dataset["wind_speed"].values
        )

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
        np.testing.assert_array_equal(
            result["delta_temperature"].isel(time=1).values, expected_delta
        )

        # Check that wind_speed is unchanged (no "delta_" prefix)
        assert "delta_wind_speed" not in result
        np.testing.assert_array_equal(
            result["wind_speed"].values, sample_dataset["wind_speed"].values
        )

    def test_select_variables(self, sample_dataset):
        """Test _select_variables filters dataset correctly."""
        variables = ["temperature"]

        result = SimpleWWMDataset._select_variables(sample_dataset, variables)

        # Check only selected variable is present
        assert "temperature" in result
        assert "wind_speed" not in result

        # Check data is unchanged
        np.testing.assert_array_equal(
            result["temperature"].values, sample_dataset["temperature"].values
        )

    def test_select_variables_multiple(self, sample_dataset):
        """Test _select_variables with multiple variables."""
        variables = ["temperature", "wind_speed"]

        result = SimpleWWMDataset._select_variables(sample_dataset, variables)

        # Check both variables are present
        assert "temperature" in result
        assert "wind_speed" in result

        # Check data is unchanged
        np.testing.assert_array_equal(
            result["temperature"].values, sample_dataset["temperature"].values
        )
        np.testing.assert_array_equal(
            result["wind_speed"].values, sample_dataset["wind_speed"].values
        )

    def test_remove_first_timestep_if_nan_with_nans(self):
        """Test _remove_first_timestep_if_nan removes first timestep when it has NaN."""
        time = np.arange(4)
        nSCHISM_hgrid_node = np.arange(2)

        # Create dataset with NaN in first timestep
        data = np.array(
            [
                [np.nan, np.nan],  # t=0
                [10, 11],  # t=1
                [20, 21],  # t=2
                [30, 31],  # t=3
            ]
        )

        ds = xr.Dataset(
            {"var": (["time", "nSCHISM_hgrid_node"], data)},
            coords={"time": time, "nSCHISM_hgrid_node": nSCHISM_hgrid_node},
        )

        result = SimpleWWMDataset._remove_first_timestep_if_nan(ds)

        # Check first timestep was removed
        assert len(result["time"]) == 3
        assert result["time"][0].values == 1

        # Check remaining data is correct
        np.testing.assert_array_equal(
            result["var"].isel(time=0).values, np.array([10, 11])
        )

    def test_remove_first_timestep_if_nan_without_nans(self, sample_dataset):
        """Test _remove_first_timestep_if_nan keeps data when no NaN present."""
        result = SimpleWWMDataset._remove_first_timestep_if_nan(sample_dataset)

        # Check no timesteps were removed
        assert len(result["time"]) == len(sample_dataset["time"])

        # Check data is unchanged
        np.testing.assert_array_equal(
            result["temperature"].values, sample_dataset["temperature"].values
        )

    def test_remove_first_timestep_if_nan_partial_nans(self):
        """Test _remove_first_timestep_if_nan removes first timestep with partial NaN."""
        time = np.arange(3)
        nSCHISM_hgrid_node = np.arange(2)

        # Create dataset with some NaN in first timestep
        data = np.array(
            [
                [10, np.nan],  # t=0 (partial NaN)
                [20, 21],  # t=1
                [30, 31],  # t=2
            ]
        )

        ds = xr.Dataset(
            {"var": (["time", "nSCHISM_hgrid_node"], data)},
            coords={"time": time, "nSCHISM_hgrid_node": nSCHISM_hgrid_node},
        )

        result = SimpleWWMDataset._remove_first_timestep_if_nan(ds)

        # Check first timestep was removed (even with partial NaN)
        assert len(result["time"]) == 2
        assert result["time"][0].values == 1

    def test_apply_scaling(self, sample_dataset):
        """Test _apply_scaling applies standardization correctly."""
        variables = ["temperature"]
        scaling_params = {"temperature": {"mean": 20.0, "std": 5.0}}

        result = SimpleWWMDataset._apply_scaling(
            sample_dataset.copy(), variables, scaling_params
        )

        # Check scaling was applied: (x - mean) / std
        expected = (sample_dataset["temperature"].values - 20.0) / 5.0
        np.testing.assert_array_almost_equal(result["temperature"].values, expected)

        # Check unscaled variable is unchanged
        np.testing.assert_array_equal(
            result["wind_speed"].values, sample_dataset["wind_speed"].values
        )

    def test_apply_scaling_multiple_variables(self, sample_dataset):
        """Test _apply_scaling with multiple variables."""
        variables = ["temperature", "wind_speed"]
        scaling_params = {
            "temperature": {"mean": 20.0, "std": 5.0},
            "wind_speed": {"mean": 10.0, "std": 2.0},
        }

        result = SimpleWWMDataset._apply_scaling(
            sample_dataset.copy(), variables, scaling_params
        )

        # Check both variables were scaled
        expected_temp = (sample_dataset["temperature"].values - 20.0) / 5.0
        expected_wind = (sample_dataset["wind_speed"].values - 10.0) / 2.0

        np.testing.assert_array_almost_equal(
            result["temperature"].values, expected_temp
        )
        np.testing.assert_array_almost_equal(result["wind_speed"].values, expected_wind)

    def test_apply_scaling_missing_params(self, sample_dataset):
        """Test _apply_scaling handles missing scaling parameters."""
        variables = ["temperature", "wind_speed"]
        scaling_params = {
            "temperature": {"mean": 20.0, "std": 5.0}
            # wind_speed scaling params missing
        }

        result = SimpleWWMDataset._apply_scaling(
            sample_dataset.copy(), variables, scaling_params
        )

        # Check temperature was scaled
        expected_temp = (sample_dataset["temperature"].values - 20.0) / 5.0
        np.testing.assert_array_almost_equal(
            result["temperature"].values, expected_temp
        )

        # Check wind_speed was NOT scaled (left unchanged)
        np.testing.assert_array_equal(
            result["wind_speed"].values, sample_dataset["wind_speed"].values
        )

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
