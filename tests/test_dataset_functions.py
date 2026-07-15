# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import xarray as xr

from src.data import dataset_utils as du
from src.data.WWMDatasets import CenterHistoryGraphWWMDataset, GraphWWMDataset, SequenceWWMDataset, SimpleWWMDataset


class TestDatasetFunctions:
    """Test suite for dataset_utils functions using synthetic xarray datasets."""

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

        result = du.compute_prevT(sample_dataset, features)

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

        result = du.compute_deltaT(sample_dataset, targets)

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

        result = du.select_variables(sample_dataset, variables)

        # Check only selected variable is present
        assert "temperature" in result
        assert "wind_speed" not in result

        # Check data is unchanged
        np.testing.assert_array_equal(result["temperature"].values, sample_dataset["temperature"].values)

    def test_apply_scaling(self, sample_dataset):
        """Test _apply_scaling applies standardization correctly."""
        variables = ["temperature"]
        scaling_params = {"temperature": {"mean": 20.0, "std": 5.0}}

        result = du.apply_scaling(sample_dataset.copy(), variables, scaling_params)

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

        result = du.apply_scaling(sample_dataset.copy(), variables, scaling_params)

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

        result = du.apply_scaling(sample_dataset.copy(), variables, scaling_params)

        # Check temperature was scaled
        expected_temp = (sample_dataset["temperature"].values - 20.0) / 5.0
        np.testing.assert_array_almost_equal(result["temperature"].values, expected_temp)

        # Check wind_speed was NOT scaled (left unchanged)
        np.testing.assert_array_equal(result["wind_speed"].values, sample_dataset["wind_speed"].values)

    def test_compute_prevT_and_deltaT_combined(self, sample_dataset):
        """Test combined workflow of computing prev and delta variables."""
        # First compute previous timestep
        features = ["prev_temperature"]
        ds_with_prev = du.compute_prevT(sample_dataset, features)

        # Then compute delta
        targets = ["delta_temperature"]
        ds_final = du.compute_deltaT(ds_with_prev, targets)

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
    """Test suite for du.reconstruct_from_delta function."""

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

        reconstructed, _ = du.reconstruct_from_delta(
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
            du.reconstruct_from_delta(
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

        reconstructed, updated_ds = du.reconstruct_from_delta(
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
            data_range=[1, 2],  # days 1-2 inclusive = 48 samples
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )

        # After removing first timestep (for delta/prev), n_time = 47
        # n_valid_times = 47 - 5 + 1 = 43
        # Total samples = 43 * 5 nodes = 215
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
        with pytest.raises(ValueError, match="has only"):
            SequenceWWMDataset(
                data_dir=mock_sequence_dataset,
                data_range=[1, 2],  # Small time range
                features=["U10"],
                targets=["sigWaveHeight"],
                seq_len=1000,  # Too large
            )

    @pytest.fixture
    def deterministic_sequence_dataset(self, tmp_path):
        """Create deterministic Zarr data so sequence indexing can be validated exactly."""
        import pandas as pd

        n_time = 30
        n_nodes = 4

        time = pd.date_range("2020-01-01", periods=n_time, freq="h")
        nSCHISM_hgrid_node = np.arange(n_nodes)

        # Feature values encode time/node to make index mapping assertions unambiguous.
        t = np.arange(n_time)[:, None]
        n = np.arange(n_nodes)[None, :]
        U10 = (100.0 * t + n).astype(np.float32)
        V10 = (1000.0 * t + n).astype(np.float32)
        sigWaveHeight = (10.0 * t + n).astype(np.float32)

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

        zarr_path = tmp_path / "deterministic_sequence.zarr"
        ds.to_zarr(zarr_path)
        return zarr_path

    def test_sequence_dataset_len_is_exact(self, deterministic_sequence_dataset):
        """Test that len and n_valid_times follow exact sequence arithmetic."""
        seq_len = 3
        ds = SequenceWWMDataset(
            data_dir=deterministic_sequence_dataset,
            data_range=[1, 1],
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )

        # data_range=[1,1] keeps day 1 inclusive (24 samples); dataset drops first timestep -> 23
        assert ds.n_time == 23
        assert ds.n_valid_times == 21
        assert len(ds) == 84  # 21 valid starts * 4 nodes

    def test_sequence_dataset_preload_item_values(self, deterministic_sequence_dataset):
        """Test preloaded sequence indexing returns the expected timesteps and target."""
        seq_len = 3
        ds = SequenceWWMDataset(
            data_dir=deterministic_sequence_dataset,
            data_range=[1, 1],
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )

        # idx=0 -> time_start=0, node=0; after first timestep removal this maps to original t=1
        X_seq, y_target = ds[0]

        expected_u10 = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        expected_v10 = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)

        assert tuple(X_seq.shape) == (3, 2)
        np.testing.assert_allclose(X_seq[:, 0].numpy(), expected_u10)
        np.testing.assert_allclose(X_seq[:, 1].numpy(), expected_v10)
        # Target is last step in the sequence: original t=3 at node 0
        np.testing.assert_allclose(y_target.numpy(), np.array([30.0], dtype=np.float32))

    def test_sequence_dataset_lazy_matches_preload(self, deterministic_sequence_dataset):
        """Test lazy-loading sequence access matches preloaded access for the same index."""
        seq_len = 4
        preload_ds = SequenceWWMDataset(
            data_dir=deterministic_sequence_dataset,
            data_range=[1, 1],
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )
        lazy_ds = SequenceWWMDataset(
            data_dir=deterministic_sequence_dataset,
            data_range=[1, 1],
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )

        # Choose index mapping to non-zero time and node.
        sample_idx = 13
        X_preload, y_preload = preload_ds[sample_idx]
        X_lazy, y_lazy = lazy_ds[sample_idx]

        assert tuple(X_lazy.shape) == (4, 2)
        assert tuple(y_lazy.shape) == (1,)
        np.testing.assert_allclose(X_lazy.numpy(), X_preload.numpy())
        np.testing.assert_allclose(y_lazy.numpy(), y_preload.numpy())

    def test_sequence_dataset_multi_range_no_straddle(self, deterministic_sequence_dataset):
        """Windows must not straddle the gap between non-contiguous day ranges."""
        seq_len = 3
        ds = SequenceWWMDataset(
            data_dir=deterministic_sequence_dataset,
            data_range=[[1, 1], [2, 2]],
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            seq_len=seq_len,
        )

        # Range 1: 24 raw steps -> 23 after first-step drop -> 21 window starts.
        # Range 2: fixture has 30 steps, so day 2 is clamped to 6 raw steps
        # -> 5 after first-step drop -> 3 window starts.
        assert ds.n_time == 28
        assert ds.n_valid_times == 24
        assert len(ds) == 24 * 4

        # First sample of the second range (seq_idx 21, node 0): the window
        # starts at concatenated index 23, which is raw t=25 (range 2 drops
        # its own first step, raw t=24) — NOT a window into range 1's tail.
        X_seq, y_target = ds[21 * 4]
        np.testing.assert_allclose(X_seq[:, 0].numpy(), np.array([2500.0, 2600.0, 2700.0], dtype=np.float32))
        np.testing.assert_allclose(y_target.numpy(), np.array([270.0], dtype=np.float32))


class TestGraphWWMDatasetFunctions:
    """Graph dataset coverage using a small synthetic SCHISM-like mesh."""

    @pytest.fixture
    def deterministic_graph_dataset(self, tmp_path):
        """Create deterministic Zarr data with enough mesh variables for graph datasets."""
        import pandas as pd

        n_time = 30
        n_nodes = 4
        n_edges = 5

        time = pd.date_range("2020-01-01", periods=n_time, freq="h")
        nSCHISM_hgrid_node = np.arange(n_nodes)
        nSCHISM_hgrid_edge = np.arange(n_edges)
        two = np.arange(2)

        t = np.arange(n_time)[:, None]
        n = np.arange(n_nodes)[None, :]
        U10 = (100.0 * t + n).astype(np.float32)
        V10 = (1000.0 * t + n).astype(np.float32)
        sigWaveHeight = (10.0 * t + n).astype(np.float32)

        edge_nodes = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 1],
                [1, 3],
            ],
            dtype=np.int64,
        )
        edge_nodes = np.broadcast_to(edge_nodes, (n_time, n_edges, 2)).copy()

        ds = xr.Dataset(
            {
                "U10": (["time", "nSCHISM_hgrid_node"], U10),
                "V10": (["time", "nSCHISM_hgrid_node"], V10),
                "sigWaveHeight": (["time", "nSCHISM_hgrid_node"], sigWaveHeight),
                "SCHISM_hgrid_edge_nodes": (
                    ["time", "nSCHISM_hgrid_edge", "two"],
                    edge_nodes,
                ),
            },
            coords={
                "time": time,
                "nSCHISM_hgrid_node": nSCHISM_hgrid_node,
                "nSCHISM_hgrid_edge": nSCHISM_hgrid_edge,
                "two": two,
                "SCHISM_hgrid_node_x": (["nSCHISM_hgrid_node"], np.array([0.0, 1.0, 1.0, 0.0])),
                "SCHISM_hgrid_node_y": (["nSCHISM_hgrid_node"], np.array([0.0, 0.0, 1.0, 1.0])),
            },
        )

        zarr_path = tmp_path / "deterministic_graph.zarr"
        ds.to_zarr(zarr_path)
        return zarr_path

    def test_graph_dataset_cache_mode_false_uses_on_demand_reads(self, deterministic_graph_dataset):
        """Graph samples should work without target/feature RAM caches."""
        ds = GraphWWMDataset(
            data_dir=deterministic_graph_dataset,
            data_range=[1, 1],
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            graph_features=["SCHISM_hgrid_edge_nodes"],
            max_nodes=3,
            edge_weight_scale=1.0,
            cache_mode=False,
        )

        graph = ds.get(0)

        assert len(ds) == 23 * 4
        assert not hasattr(ds, "_features_ram")
        assert not hasattr(ds, "_targets_ram")
        assert tuple(graph.x.shape) == (3, 2)
        assert tuple(graph.y.shape) == (3, 1)
        assert graph.edge_weight.shape[0] == graph.edge_index.shape[1]

    @pytest.mark.parametrize("cache_mode", [True, False])
    def test_center_history_graph_uses_expected_sequence(self, deterministic_graph_dataset, cache_mode):
        """Center-history samples should return the same sequence with and without RAM cache."""
        ds = CenterHistoryGraphWWMDataset(
            data_dir=deterministic_graph_dataset,
            data_range=[1, 1],
            features=["U10", "V10"],
            targets=["sigWaveHeight"],
            graph_features=["SCHISM_hgrid_edge_nodes"],
            seq_len=3,
            max_nodes=3,
            edge_weight_scale=1.0,
            cache_mode=cache_mode,
        )

        graph = ds.get(0)

        assert hasattr(ds, "_features_ram") is cache_mode
        assert hasattr(ds, "_targets_ram") is cache_mode
        assert tuple(graph.x.shape) == (3, 2)
        assert tuple(graph.x_center_seq.shape) == (3, 2)
        np.testing.assert_allclose(
            graph.x_center_seq.numpy(),
            np.array(
                [
                    [100.0, 1000.0],
                    [200.0, 2000.0],
                    [300.0, 3000.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(graph.y[:, 0].numpy(), np.array([30.0, 31.0, 32.0], dtype=np.float32))


class TestSimpleWWMDatasetCoverage:
    """Additional branch coverage tests for SimpleWWMDataset and dataset_utils."""

    @pytest.fixture
    def simple_dataset_zarr(self, tmp_path):
        """Create deterministic Zarr data for SimpleWWMDataset integration tests."""
        import pandas as pd

        n_time = 48
        n_nodes = 4

        time = pd.date_range("2020-01-01", periods=n_time, freq="h")
        nodes = np.arange(n_nodes)

        t = np.arange(n_time, dtype=np.float32)[:, None]
        n = np.arange(n_nodes, dtype=np.float32)[None, :]

        ds = xr.Dataset(
            {
                "U10": (["time", "nSCHISM_hgrid_node"], 100.0 * t + n),
                "V10": (["time", "nSCHISM_hgrid_node"], 200.0 * t + n),
                "sigWaveHeight": (["time", "nSCHISM_hgrid_node"], 10.0 * t + n),
            },
            coords={"time": time, "nSCHISM_hgrid_node": nodes},
        )

        zarr_path = tmp_path / "simple_dataset.zarr"
        ds.to_zarr(zarr_path)
        return zarr_path

    def test_load_scaling_config_and_revert_scaling(self, tmp_path):
        """Cover YAML scaling load and revert scaling happy/error paths."""
        sample_dataset = xr.Dataset(
            {
                "temperature": (
                    ["time", "nSCHISM_hgrid_node"],
                    np.array([[10.0, 11.0], [15.0, 16.0], [20.0, 21.0]], dtype=np.float32),
                ),
                "wind_speed": (
                    ["time", "nSCHISM_hgrid_node"],
                    np.array([[5.0, 6.0], [8.0, 9.0], [11.0, 12.0]], dtype=np.float32),
                ),
            },
            coords={"time": np.arange(3), "nSCHISM_hgrid_node": np.arange(2)},
        )

        scaling_yaml = tmp_path / "scaling.yaml"
        scaling_yaml.write_text("temperature:\n  mean: 20.0\n  std: 5.0\n")

        params = du.load_scaling_config(str(scaling_yaml))
        assert params["temperature"]["mean"] == 20.0
        assert params["temperature"]["std"] == 5.0

        scaled = du.apply_scaling(sample_dataset.copy(), ["temperature"], params)
        reverted = du.revert_scaling(scaled.copy(), ["temperature"], params)
        np.testing.assert_allclose(reverted["temperature"].values, sample_dataset["temperature"].values)

        with pytest.raises(ValueError, match="No parameters found"):
            du.revert_scaling(sample_dataset.copy(), ["wind_speed"], params)

    def test_reconstruct_from_delta_with_scaling_reverts_prev(self):
        """Cover branch where reconstruction reverts scaling of prev_* feature."""
        ds = xr.Dataset(
            {
                "temperature": (["time", "nSCHISM_hgrid_node"], np.array([[10.0], [20.0], [30.0]])),
                # Scaled prev_temperature with mean=10, std=2 => raw 10 and 20 become 0 and 5
                "prev_temperature": (["time", "nSCHISM_hgrid_node"], np.array([[np.nan], [0.0], [5.0]])),
            },
            coords={"time": np.arange(3), "nSCHISM_hgrid_node": np.array([0])},
        )

        delta_predictions = np.array([[0.0], [1.0], [2.0]])
        reconstructed, _ = du.reconstruct_from_delta(
            delta_predictions=delta_predictions,
            dataset=ds,
            target_name="delta_temperature",
            features=["prev_temperature"],
            scaling_params={"prev_temperature": {"mean": 10.0, "std": 2.0}},
        )

        np.testing.assert_allclose(reconstructed[1:, 0], np.array([11.0, 22.0]))

    def test_init_raises_when_node_idx_csv_missing(self, simple_dataset_zarr, tmp_path):
        """Cover FileNotFoundError path for missing node index CSV."""
        missing_csv = tmp_path / "missing_nodes.csv"

        with pytest.raises(FileNotFoundError, match="Node index CSV not found"):
            SimpleWWMDataset(
                data_dir=simple_dataset_zarr,
                data_range=[1, 2],
                features=["U10"],
                targets=["sigWaveHeight"],
                node_idx_csv=str(missing_csv),
            )

    def test_init_raises_when_node_idx_column_invalid(self, simple_dataset_zarr, tmp_path):
        """Cover ValueError path when CSV has wrong column name."""
        bad_csv = tmp_path / "bad_nodes.csv"
        bad_csv.write_text("node_id\n0\n1\n")

        with pytest.raises(ValueError, match="must have a 'nSCHISM_hgrid_node' column"):
            SimpleWWMDataset(
                data_dir=simple_dataset_zarr,
                data_range=[1, 2],
                features=["U10"],
                targets=["sigWaveHeight"],
                node_idx_csv=str(bad_csv),
            )

    def test_compute_weights_unknown_strategy_raises(self, simple_dataset_zarr):
        """Cover error branch for unsupported sample_weight_strategy."""
        ds = SimpleWWMDataset(
            data_dir=simple_dataset_zarr,
            data_range=[1, 2],
            features=["U10"],
            targets=["sigWaveHeight"],
            sample_weight_strategy=None,
        )

        with pytest.raises(ValueError, match="Unknown sample_weight_strategy"):
            du.compute_weights(ds.y[:, 0], "invalid_strategy", 1.0)

    def test_simple_dataset_multi_range_len(self, simple_dataset_zarr):
        """Non-contiguous day ranges: each range drops its own first time step."""
        ds = SimpleWWMDataset(
            data_dir=simple_dataset_zarr,
            data_range=[[1, 1], [2, 2]],
            features=["U10"],
            targets=["sigWaveHeight"],
        )

        # Each day: 24 steps minus the per-range dropped first step -> 23.
        assert ds.range_lengths == [23, 23]
        assert len(ds) == 46 * 4


class TestMultiRangeHelpers:
    """Test suite for day-range normalization and gap-aware sequence starts."""

    def test_normalize_single_range(self):
        assert du.normalize_day_ranges([1, 2]) == [[1, 2]]

    def test_normalize_multi_range(self):
        assert du.normalize_day_ranges([[1, 1], [3, 4]]) == [[1, 1], [3, 4]]

    def test_normalize_rejects_overlap(self):
        with pytest.raises(ValueError, match="non-overlapping"):
            du.normalize_day_ranges([[1, 3], [3, 4]])

    def test_normalize_rejects_descending_order(self):
        with pytest.raises(ValueError, match="ascending"):
            du.normalize_day_ranges([[3, 4], [1, 2]])

    def test_normalize_rejects_malformed_range(self):
        with pytest.raises(ValueError, match="start <= end"):
            du.normalize_day_ranges([5, 2])

    def test_valid_sequence_starts_skip_gaps(self):
        starts = du.compute_valid_sequence_starts([5, 4], seq_len=3)
        # Range 1 (indices 0-4): starts 0, 1, 2. Range 2 (indices 5-8): starts 5, 6.
        np.testing.assert_array_equal(starts, np.array([0, 1, 2, 5, 6]))

    def test_valid_sequence_starts_raise_on_short_range(self):
        with pytest.raises(ValueError, match="seq_len=3 does not fit"):
            du.compute_valid_sequence_starts([5, 2], seq_len=3)


class TestComputeSincos:
    """Test suite for compute_sincos and reconstruct_from_sincos."""

    @pytest.fixture
    def angle_dataset(self):
        """Synthetic dataset with a meanWaveDirection variable."""
        angles_deg = np.array(
            [
                [0.0, 90.0, 180.0],  # t=0
                [270.0, 45.0, 360.0],  # t=1
            ]
        )
        return xr.Dataset(
            {"meanWaveDirection": (["time", "nSCHISM_hgrid_node"], angles_deg)},
            coords={"time": [0, 1], "nSCHISM_hgrid_node": [0, 1, 2]},
        )

    def test_sin_variable_created(self, angle_dataset):
        """sin_meanWaveDirection is added to the dataset."""
        ds = du.compute_sincos(angle_dataset, ["sin_meanWaveDirection"])
        assert "sin_meanWaveDirection" in ds

    def test_cos_variable_created(self, angle_dataset):
        """cos_meanWaveDirection is added to the dataset."""
        ds = du.compute_sincos(angle_dataset, ["cos_meanWaveDirection"])
        assert "cos_meanWaveDirection" in ds

    def test_sin_values_correct(self, angle_dataset):
        """sin(0°)=0, sin(90°)=1, sin(180°)=0, sin(270°)=-1."""
        ds = du.compute_sincos(angle_dataset, ["sin_meanWaveDirection"])
        sin_vals = ds["sin_meanWaveDirection"].values
        expected = np.sin(np.radians(angle_dataset["meanWaveDirection"].values))
        np.testing.assert_allclose(sin_vals, expected, atol=1e-6)

    def test_cos_values_correct(self, angle_dataset):
        """cos(0°)=1, cos(90°)=0, cos(180°)=-1, cos(270°)=0."""
        ds = du.compute_sincos(angle_dataset, ["cos_meanWaveDirection"])
        cos_vals = ds["cos_meanWaveDirection"].values
        expected = np.cos(np.radians(angle_dataset["meanWaveDirection"].values))
        np.testing.assert_allclose(cos_vals, expected, atol=1e-6)

    def test_0_and_360_give_same_sincos(self, angle_dataset):
        """0° and 360° should produce the same sin/cos values."""
        ds = du.compute_sincos(angle_dataset, ["sin_meanWaveDirection", "cos_meanWaveDirection"])
        # t=0, node=0 is 0°; t=1, node=2 is 360°
        np.testing.assert_allclose(
            ds["sin_meanWaveDirection"].values[0, 0],
            ds["sin_meanWaveDirection"].values[1, 2],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            ds["cos_meanWaveDirection"].values[0, 0],
            ds["cos_meanWaveDirection"].values[1, 2],
            atol=1e-6,
        )

    def test_non_sincos_vars_ignored(self, angle_dataset):
        """Variables not prefixed with sin_/cos_ are left unchanged."""
        ds = du.compute_sincos(angle_dataset, ["meanWaveDirection", "windSpeedX"])
        # meanWaveDirection was already there; nothing new should be added
        assert set(ds.data_vars) == {"meanWaveDirection"}

    def test_original_var_preserved(self, angle_dataset):
        """The original angle variable is not removed by compute_sincos."""
        ds = du.compute_sincos(angle_dataset, ["sin_meanWaveDirection", "cos_meanWaveDirection"])
        assert "meanWaveDirection" in ds

    def test_prev_sincos_created_by_compute_prevT(self, angle_dataset):
        """After compute_sincos, compute_prevT correctly shifts sin_/cos_ variables."""
        ds = du.compute_sincos(angle_dataset, ["sin_meanWaveDirection", "prev_sin_meanWaveDirection"])
        ds = du.compute_prevT(ds, ["prev_sin_meanWaveDirection"])
        assert "prev_sin_meanWaveDirection" in ds
        # t=1 prev should equal t=0 sin
        np.testing.assert_allclose(
            ds["prev_sin_meanWaveDirection"].values[1],
            ds["sin_meanWaveDirection"].values[0],
            atol=1e-6,
        )


class TestReconstructFromSincos:
    """Test suite for reconstruct_from_sincos."""

    def test_cardinal_angles(self):
        """Roundtrip for 0°, 90°, 180°, 270°."""
        angles = np.array([0.0, 90.0, 180.0, 270.0])
        sin_vals = np.sin(np.radians(angles))
        cos_vals = np.cos(np.radians(angles))
        reconstructed = du.reconstruct_from_sincos(sin_vals, cos_vals)
        np.testing.assert_allclose(reconstructed, angles, atol=1e-5)

    def test_arbitrary_angles_roundtrip(self):
        """Roundtrip for a range of angles."""
        angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 359.0])
        sin_vals = np.sin(np.radians(angles))
        cos_vals = np.cos(np.radians(angles))
        reconstructed = du.reconstruct_from_sincos(sin_vals, cos_vals)
        np.testing.assert_allclose(reconstructed, angles, atol=1e-4)

    def test_output_in_0_360(self):
        """Output is always in [0, 360)."""
        sin_vals = np.array([-1.0, 0.0, 1.0, 0.0])
        cos_vals = np.array([0.0, 1.0, 0.0, -1.0])
        reconstructed = du.reconstruct_from_sincos(sin_vals, cos_vals)
        assert np.all(reconstructed >= 0.0)
        assert np.all(reconstructed < 360.0)

    def test_360_maps_to_0(self):
        """360° and 0° give the same sin/cos, so reconstruction returns 0°."""
        sin_360 = np.sin(np.radians(360.0))
        cos_360 = np.cos(np.radians(360.0))
        reconstructed = du.reconstruct_from_sincos(np.array([sin_360]), np.array([cos_360]))
        # Accept either 0.0 or 360.0 (they're equivalent), but prefer 0.0
        assert np.isclose(reconstructed[0], 0.0, atol=1e-6) or np.isclose(reconstructed[0], 360.0, atol=1e-6)
