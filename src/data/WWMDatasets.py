# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
import yaml
import time
import pandas as pd
from torch.utils.data import Dataset

from torch_geometric.data import Dataset as GDataset
from torch_geometric.data import Data as GData

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class SimpleWWMDataset(Dataset):
    """Dataset that loads a Zarr dataset and returns features and targets per grid point and time.

    Note that preprocessing the zarr files is handled externally (CreateZarr.ipynb).
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with a column named 'nSCHISM_hgrid_node'
                containing node indices to keep. If None, keep all nodes.
            sample_weight_strategy: Strategy for computing per-sample weights.
                Options: None (no weighting), 'z_score' (weight by z-score of targets).
            sample_weight_power: Exponent to control the intensity of sample weighting (default: 1.0).
            subsample_prob: Optional probability threshold for subsampling based on z_score.
                If provided, samples are retained with probability derived from their z_score.
                Higher z_score (outliers) have higher retention probability.
            preload_data: If True, preload X/y in memory. If False, load X/y lazily in __getitem__.
                With multiple workers, each worker slices the time axis to avoid shared Zarr chunks.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.features = features
        self.targets = targets
        self.node_idx_csv = node_idx_csv
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power
        self.subsample_prob = subsample_prob
        self.preload_data = preload_data

        # account for omegaconf list
        self.features = [str(var) for var in list(self.features)]
        self.targets = [str(var) for var in list(self.targets)]

        # Store parameters for worker initialization
        self.worker_id = None
        self.num_workers = None
        self.worker_time_len = None

        self.ds = xr.open_dataset(
            self.data_dir,
            engine="zarr",
        )
        # get samples per day for indexing
        time_delta = self.ds.time.values[1] - self.ds.time.values[0]
        self.samples_per_day = int(pd.Timedelta(days=1) / time_delta)
        log.info(f"Dataset has {self.samples_per_day} samples per day.")
        start_idx = (self.data_range[0] - 1) * self.samples_per_day
        end_idx = (self.data_range[1] - 1) * self.samples_per_day
        self.ds = self.ds.isel(time=slice(start_idx, end_idx))
        log.info(f"Loaded dataset with times ranging from {self.ds.time.values[0]} to {self.ds.time.values[-1]}.")

        # Select nodes based on node_idx_csv
        if self.node_idx_csv:
            node_idx_path = Path(self.node_idx_csv)
            if not node_idx_path.exists():
                raise FileNotFoundError(f"Node index CSV not found: {node_idx_path}")
            node_idx_df = pd.read_csv(node_idx_path)
            if "nSCHISM_hgrid_node" not in node_idx_df.columns:
                raise ValueError("Node index CSV must have a 'nSCHISM_hgrid_node' column")
            node_idx = node_idx_df["nSCHISM_hgrid_node"].to_numpy(dtype=int)
            self.ds = self.ds.sel(nSCHISM_hgrid_node=node_idx)
            log.info(f"Selected {len(node_idx)} nodes from {node_idx_path}.")

        # now this is the same dataset
        self.ds = self._compute_prevT(self.ds, self.features)
        self.ds = self._compute_deltaT(self.ds, self.targets)
        # keep original variables always for reconstruction
        selected_targets = self.targets + [tt.replace("delta_", "") for tt in self.targets]
        self.ds = self._select_variables(self.ds, self.features + selected_targets)

        # clear nans from computing difference with respect to time
        self.ds = self._remove_first_timestep_if_nan(self.ds)

        self.n_nodes = len(self.ds.nSCHISM_hgrid_node)

        # scale all variables
        self.scaling_params = self._load_scaling_config(scaling_config) if scaling_config else None

        if self.scaling_params:
            self.ds = self._apply_scaling(self.ds, self.features, self.scaling_params)

        # Preload data based on preload_data flag
        if self.preload_data:
            self.X = self._retrieve_data(self.ds, self.features)
            self.y = self._retrieve_data(self.ds, self.targets)
        else:
            self.X = None
            self.y = None
            log.info("Lazy loading enabled: features and targets will be loaded on-demand")

        # Apply subsampling if requested (only works with preloaded data)
        if self.subsample_prob is not None:
            if self.preload_data:
                self._subsample_by_zscore()
            else:
                log.warning("Subsampling (subsample_prob) requires preload_data=True. Skipping subsampling.")

        # Compute weights (only works with preloaded data)
        if self.sample_weight_strategy is not None:
            if self.preload_data:
                self.compute_weights()
            else:
                log.warning("Sample weighting requires preload_data=True. Skipping weight computation.")
                self.weights = None
        else:
            self.compute_weights()

        num_samples = len(self.ds.time) * len(self.ds.nSCHISM_hgrid_node)
        log.info(f"Dataset initialized with {num_samples} samples. Preload data: {self.preload_data}.")

    @staticmethod
    def _compute_deltaT(dataset: xr.Dataset, targets: List[str]) -> xr.Dataset:
        """Compute delta_target = target(t) - target(t-1).

        For targets starting with 'delta_', computes the difference with respect to the
        previous time step and stores that value in a new variable.

                Example for sigWaveHeight and time=t:
                    ds["delta_sigWaveHeight"][t] = ds["sigWaveHeight"][t] - ds["sigWaveHeight"][t-1]

        Args:
            dataset: xarray Dataset containing the data.
            targets: List of target variable names.

        Returns:
            Dataset with per-target differences with respect to time added.
        """
        for target in targets:
            if target.startswith("delta_"):
                original_var = target.replace("delta_", "")
                dataset[target] = dataset[original_var].diff("time")
                log.info(f"Added variable {target}")

        return dataset

    @staticmethod
    def _compute_prevT(dataset: xr.Dataset, features: List[str]) -> xr.Dataset:
        """Compute prev_feature = feature(t-1).

                Example for sigWaveHeight and time=t:
                    ds["prev_sigWaveHeight"][t] = ds["sigWaveHeight"][t-1]

        Args:
            dataset (xr.Dataset): xarray Dataset containing the data.
            features (List[str]): List of feature variable names.

        Returns:
            xr.Dataset: Dataset with variables added from previous time step.
        """
        for feature in features:
            if feature.startswith("prev_"):
                original_var = feature.replace("prev_", "")
                dataset[feature] = dataset[original_var].shift(time=1)
                log.info(f"Added variable {feature}")

        return dataset

    @staticmethod
    def _select_variables(dataset: xr.Dataset, variables: List[str]) -> xr.Dataset:
        """Select only specified variables from dataset.

        Args:
            dataset: xarray Dataset containing the data.
            variables: List of variable names to select.

        Returns:
            xarray Dataset containing only specified variables.
        """
        return dataset[[v for v in variables]]

    @staticmethod
    def _remove_first_timestep_if_nan(dataset: xr.Dataset) -> xr.Dataset:
        """Remove first time step if it contains NaN values.

        When computing shifted variables (prev_* or delta_*), the first time step
        will contain NaN values. This method removes it unconditionally to avoid
        triggering computation with .any() check.

        Args:
            dataset: xarray Dataset containing the data.

        Returns:
            Dataset with first time step removed.
        """
        # Unconditionally remove first timestep to avoid triggering .any() computation
        # We know it will have NaN values from shift/diff operations
        log.info("Removing first time step (NaN from shift/diff operations)")
        return dataset.isel(time=slice(1, None))

    @staticmethod
    def _retrieve_data(dataset: xr.Dataset, variables: List[str]) -> torch.Tensor:
        """Extract variable data from an xarray Dataset and convert to a PyTorch tensor.

        Args:
            dataset: xarray Dataset containing the data.
            variables: List of variable names to extract.

        Returns:
            PyTorch tensor of shape (n_samples, n_variables) with time as the slow axis
            and nSCHISM_hgrid_node as the fast axis.
        """
        start_time = time.time()
        # stack in xarray
        da = dataset[variables].to_array(dim="variable")
        da = da.stack(sample=[d for d in da.dims if d != "variable"])
        da = da.transpose("sample", "variable")

        # convert to torch
        tensor = torch.from_numpy(np.asarray(da.data))

        log.info(f"Finished retrieving {len(variables)} variables in {time.time() - start_time:.0f} seconds.")
        return tensor

    @staticmethod
    def _load_scaling_config(config_path: str) -> Dict[str, Dict[str, float]]:
        """Load scaling parameters from YAML file.

        Args:
            config_path: Path to a YAML file containing mean and std for each variable.

        Returns:
            Dictionary mapping variable names to their mean and std values.
        """
        with open(config_path, "r") as f:
            scaling_params = yaml.safe_load(f)
        return scaling_params

    @staticmethod
    def _apply_scaling(
        dataset: xr.Dataset,
        variables: List[str],
        scaling_params: Dict[str, Dict[str, float]],
    ) -> xr.Dataset:
        """Apply standardization scaling to variables in the dataset.

        Args:
            dataset: xarray Dataset containing the variables.
            variables: List of variable names to scale.
            scaling_params: Dictionary mapping variable names to their mean and std values.

        Returns:
            Dataset with scaled variables (modified in place).
        """
        for var in variables:
            if var in scaling_params:
                mean = scaling_params[var]["mean"]
                std = scaling_params[var]["std"]
                dataset[var] = (dataset[var] - mean) / std
                log.info(f"Scaled variable '{var}' with mean={mean:.4f}, std={std:.4f}")
            else:
                log.warning(f"No scaling parameters found for variable '{var}', leaving unscaled")
        return dataset

    @staticmethod
    def _revert_scaling(
        dataset: xr.Dataset,
        variables: List[str],
        scaling_params: Dict[str, Dict[str, float]],
    ) -> xr.Dataset:
        """Revert standardization scaling of variables in the dataset.

        Args:
            dataset (xr.Dataset): xarray Dataset containing the variables.
            variables (List[str]): List of variable names to un-scale.
            scaling_params (Dict[str, Dict[str, float]]): Dictionary mapping variable names to
              their mean and std values.

        Returns:
            xr.Dataset: Dataset with un-scaled variables (modified in place).
        """
        for var in variables:
            if var in scaling_params:
                mean = scaling_params[var]["mean"]
                std = scaling_params[var]["std"]
                dataset[var] = dataset[var] * std + mean
                log.info(f"Reverted scaling for variable {var}.")
            else:
                raise ValueError(f"No parameters found to revert scaling for {var}.")

        return dataset

    @staticmethod
    def reconstruct_from_delta(
        delta_predictions: np.ndarray,
        dataset: xr.Dataset,
        target_name: str,
        features: List[str],
        scaling_params: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, xr.Dataset]:
        """Reconstruct absolute values from delta predictions.

        For targets like 'delta_sigWaveHeight', this computes:
            reconstructed[t] = prev_value[t] + delta_prediction[t]

        Args:
            delta_predictions: Array of delta predictions, shape (n_time, n_nodes).
            dataset: xarray Dataset containing the source data.
            target_name: Name of the delta target (e.g., 'delta_sigWaveHeight').
            features: List of feature names used by the model.
            scaling_params: Optional scaling parameters for reverting scaling.

        Returns:
            Tuple of (reconstructed_values, updated_dataset).
            The dataset is updated with prev_* variable if it wasn't present.
        """
        if not target_name.startswith("delta_"):
            raise ValueError(f"Target '{target_name}' is not a delta target")

        original_var = target_name.replace("delta_", "")
        prev_var = f"prev_{original_var}"

        # Compute prev_* variable if not present
        if prev_var not in dataset:
            log.info(f"Computing {prev_var} for reconstruction")
            dataset = SimpleWWMDataset._compute_prevT(dataset, [prev_var])

        # Revert scaling if prev_var was used as a scaled feature
        if scaling_params and prev_var in features:
            dataset = SimpleWWMDataset._revert_scaling(dataset, [prev_var], scaling_params)

        prev_values = dataset[prev_var].values
        reconstructed = prev_values + delta_predictions

        return reconstructed, dataset

    def compute_weights(self) -> None:
        """Compute and normalize sample weights.

        Weights are computed based on the sample_weight_strategy and normalized so that
        their sum equals the number of samples (mean = 1.0).
        """
        if self.sample_weight_strategy is None:
            self.weights = None
            log.info("No sample weight strategy specified, skipping weight computation")
            return

        log.info(f"Computing sample weights with strategy '{self.sample_weight_strategy}'...")

        n_samples = len(self.y)
        n_targets = len(self.targets)
        weights = np.ones((n_samples, n_targets), dtype=np.float32)

        if self.sample_weight_strategy == "z_score":
            z_scores = (self.y - torch.mean(self.y)) / torch.std(self.y)
            weights = 1.0 - torch.exp(-(z_scores**2) / torch.tensor(self.sample_weight_power))

        else:
            raise ValueError(
                f"Unknown sample_weight_strategy: '{self.sample_weight_strategy}'. Supported: None, 'z_score'"
            )

        # Normalize weights per target so that mean = 1.0
        for i in range(n_targets):
            weight_sum = weights[:, i].sum()
            weights[:, i] = weights[:, i] * (n_samples / weight_sum)
            log.info(
                f"Target {self.targets[i]}: normalized weights, "
                f"(mean={weights[:, i].mean():.4f}, std={weights[:, i].std():.4f}, "
                f"min={weights[:, i].min():.4f}, max={weights[:, i].max():.4f})"
            )

        self.weights = weights.float()
        log.info(f"Sample weights computed and normalized for {n_samples} samples")

    def _subsample_by_zscore(self) -> None:
        """Subsample the dataset based on z_score-derived probabilities.

        Samples with higher z_scores (outliers) have higher retention probability.
        The retention probability is computed as: p = 1 - exp(-z^2 / 2)
        This ensures that:
        - Samples near the mean (z=0) have low retention probability (~0)
        - Samples far from mean (high |z|) have high retention probability (~1)

        The actual retention is determined by comparing these probabilities against
        a uniform random draw. Only supported when data is preloaded.
        """
        log.info(f"Subsampling dataset based on z-score with target rate {self.subsample_prob}...")

        n_samples_before = len(self.y)

        # Compute z-scores for all targets (average across targets)
        z_scores = (self.y - torch.mean(self.y, dim=0)) / torch.std(self.y, dim=0)
        # Take the maximum absolute z-score across all targets for each sample
        max_abs_z = torch.abs(z_scores).max(dim=1)[0]

        # Compute retention probability: p = 1 - exp(-z^2 / 2)
        retention_probs = 1.0 - torch.exp(-(max_abs_z**2) / 2.0)

        # Scale probabilities so that the expected (mean) retention rate equals subsample_prob
        # This ensures that approximately subsample_prob fraction of samples are retained
        if retention_probs.mean() > 0:
            retention_probs = retention_probs * (self.subsample_prob / retention_probs.mean())
            # Clip to [0, 1] range in case scaling pushes values above 1
            retention_probs = torch.clamp(retention_probs, 0.0, 1.0)

        # Generate random values and keep samples where random < retention_prob
        random_vals = torch.rand(n_samples_before)
        keep_mask = random_vals < retention_probs

        # Apply mask to data
        if self.X is not None:
            self.X = self.X[keep_mask]
            self.sample_index = None
        else:
            self.sample_index = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

        self.y = self.y[keep_mask]

        n_samples_after = len(self.y)
        retention_rate = n_samples_after / n_samples_before * 100

        log.info(f"Subsampling complete: kept {n_samples_after}/{n_samples_before} samples ({retention_rate:.2f}%)")

    def __len__(self) -> int:
        """Return the number of samples in the (possibly worker-sliced) dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.ds.time) * len(self.ds.nSCHISM_hgrid_node)

    def worker_init_fn(self, worker_id: int) -> None:
        """Initialize a worker by slicing the dataset across the time dimension.

        Called after the dataset is pickled to the worker process. The dataset
        (self.ds) is already preprocessed; we just split it across workers.

        This allows lazy loading (preload_data=False) to work with multiple workers
        by having each worker load only its portion of the time axis.

        Args:
            worker_id: Worker ID assigned by DataLoader
        """
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading
            return

        self.worker_id = worker_info.id
        self.num_workers = worker_info.num_workers

        # Split the already-preprocessed dataset across workers
        total_times = len(self.ds.time)
        times_per_worker = total_times // self.num_workers
        start_time = self.worker_id * times_per_worker

        if self.worker_id == self.num_workers - 1:
            # Last worker takes remaining samples
            end_time = total_times
        else:
            end_time = start_time + times_per_worker

        # Filter dataset to this worker's samples
        self.ds = self.ds.isel(time=slice(start_time, end_time))
        self.worker_time_len = end_time - start_time

        log.info(
            f"Worker {self.worker_id}/{self.num_workers}: handling time indices [{start_time}:{end_time}] "
            f"({end_time - start_time}/{total_times})"
        )

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets, weights) as PyTorch tensors if weights are computed,
            otherwise (features, targets). In lazy mode with multiple workers, the index
            is wrapped to the worker-local time slice.
        """
        if self.preload_data:
            if self.weights is not None:
                return self.X[idx], self.y[idx], self.weights[idx]
            return self.X[idx], self.y[idx]

        # Lazy loading mode: load both features and targets from dataset
        # Indexing follows the stacked order: time is the slow axis, node is the fast axis.
        local_idx = idx

        n_nodes = self.n_nodes
        if self.worker_time_len is not None:
            local_total = self.worker_time_len * n_nodes
            local_idx = local_idx % local_total

        local_t_idx = local_idx // n_nodes
        local_x_idx = local_idx % n_nodes

        sample_data = self.ds.isel(time=local_t_idx, nSCHISM_hgrid_node=local_x_idx)

        xa = sample_data[self.features].to_dataarray()
        x = torch.from_numpy(xa.values)

        ya = sample_data[self.targets].to_dataarray()
        y = torch.from_numpy(ya.values)

        # Weights are not supported in lazy loading mode
        if self.weights is not None:
            return x, y, self.weights[idx]
        return x, y


class SequenceWWMDataset(SimpleWWMDataset):
    """Dataset that returns time sequences for LSTM/RNN models.

    Returns sequences of length `seq_len` as input features, with the next time step as target.
    Inherits data loading, scaling, and preprocessing from SimpleWWMDataset.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        seq_len: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = True,
    ) -> None:
        """Initialize the sequence dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            seq_len: Length of input sequences (number of time steps).
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            subsample_prob: Optional probability threshold for subsampling.
            preload_data: If True, preload data in memory for fast access.
        """
        self.seq_len = seq_len

        # Initialize parent class (loads and preprocesses data)
        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            subsample_prob=subsample_prob,
            preload_data=preload_data,
        )

        # Store time and node dimensions for sequence indexing
        self.n_time = len(self.ds.time)
        # Number of valid sequence start positions
        self.n_valid_times = self.n_time - self.seq_len

        if self.n_valid_times <= 0:
            raise ValueError(f"seq_len ({seq_len}) must be less than number of time steps ({self.n_time})")

        # Reshape preloaded data for efficient sequence access: (n_time, n_nodes, n_features)
        if self.preload_data and self.X is not None:
            n_features = self.X.shape[1]
            n_targets = self.y.shape[1]
            self.X = self.X.view(self.n_time, self.n_nodes, n_features)
            self.y = self.y.view(self.n_time, self.n_nodes, n_targets)
            log.info(f"Reshaped data for sequence access: X={tuple(self.X.shape)}, y={tuple(self.y.shape)}")

        log.info(
            f"SequenceWWMDataset initialized with seq_len={seq_len}, "
            f"n_valid_times={self.n_valid_times}, n_nodes={self.n_nodes}"
        )

    def __len__(self) -> int:
        """Return the number of valid sequences.

        Returns:
            Number of samples: (n_time - seq_len) * n_nodes
        """
        return self.n_valid_times * self.n_nodes

    def __getitem__(self, idx: int) -> Tuple:
        """Get a sequence sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (X_seq, y_target) where:
                - X_seq: (seq_len, n_features) tensor of input features
                - y_target: (n_targets,) tensor of target at t+seq_len
        """
        # Map flat index to (time_start, node)
        time_start = idx // self.n_nodes
        node_idx = idx % self.n_nodes

        if self.preload_data:
            # X: sequence from time_start to time_start + seq_len
            X_seq = self.X[time_start : time_start + self.seq_len, node_idx, :]

            # y: target at next time step after sequence
            y_target = self.y[time_start + self.seq_len, node_idx, :]

            return X_seq, y_target

        # Lazy loading mode
        time_indices = list(range(time_start, time_start + self.seq_len + 1))
        node = self.ds.nSCHISM_hgrid_node.values[node_idx]

        sample_data = self.ds.isel(time=time_indices).sel(nSCHISM_hgrid_node=node)

        # Features for sequence
        xa = sample_data[self.features].to_dataarray()
        X_seq = torch.from_numpy(xa.values[:, :-1].T)  # (seq_len, n_features)

        # Target at next time step
        ya = sample_data[self.targets].to_dataarray()
        y_target = torch.from_numpy(ya.values[:, -1])  # (n_targets,)

        return X_seq, y_target


class SmallGraphWWMDataset(SimpleWWMDataset, GDataset):
    """Dataset that returns graphs that represent a subregion around a center node.

    Each sample is a subgraph built via BFS from a center node on the SCHISM mesh,
    bounded by max_radius (meters), max_nodes, and max_links (max BFS hops).
    Subgraph topology is precomputed and cached; only node features change per time step.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = False,
        max_radius: float = 10000.0,
        max_nodes: int = 1000,
        max_links: int = 10,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset. Must contain graph definition.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names used for constructing the graph
                (e.g. ["SCHISM_hgrid_edge_nodes"]).
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with a column named 'nSCHISM_hgrid_node'
                containing node indices to use as center nodes. The full mesh is always
                retained for graph construction. If None, all nodes are center nodes.
            sample_weight_strategy: Strategy for computing per-sample weights.
                Options: None (no weighting), 'z_score' (weight by z-score of targets).
            sample_weight_power: Exponent to control the intensity of sample weighting (default: 1.0).
            subsample_prob: Optional probability threshold for subsampling based on z_score.
                Must be None for SmallGraphWWMDataset.
            preload_data: If True, preload X/y in memory. If False, load X/y lazily.
                Must be False for SmallGraphWWMDataset.
            max_radius: Maximum Euclidean distance (meters) from center node for
                including neighbors in the subgraph (default: 10000.).
            max_nodes: Maximum number of nodes in a subgraph (default: 1000).
            max_links: Maximum number of BFS hops from center node (default: 10).
        """
        # Pass node_idx_csv=None to parent so the full mesh is retained.
        # We handle center node selection separately below.
        SimpleWWMDataset.__init__(
            self,
            data_dir,
            data_range,
            features + graph_features,
            targets,
            scaling_config,
            None,  # keep full mesh
            sample_weight_strategy,
            sample_weight_power,
            subsample_prob,
            preload_data,
        )

        if self.subsample_prob is not None:
            raise ValueError(
                f"Subsample probability must be None for SmallGraphWWMDataset. Found {self.subsample_prob}"
            )

        self.max_radius = max_radius
        self.max_nodes = max_nodes
        self.max_links = max_links

        # Reset features (for node embeddings) vs graph_features (for mesh geometry)
        self.features = [str(f) for f in features]
        self.graph_features = [str(f) for f in graph_features]

        self.n_nodes = len(self.ds.nSCHISM_hgrid_node)

        # --- Determine center nodes from node_idx_csv ---
        if node_idx_csv is not None:
            node_idx_path = Path(node_idx_csv)
            if not node_idx_path.exists():
                raise FileNotFoundError(f"Node index CSV not found: {node_idx_path}")
            node_idx_df = pd.read_csv(node_idx_path)
            if "nSCHISM_hgrid_node" not in node_idx_df.columns:
                raise ValueError("Node index CSV must have a 'nSCHISM_hgrid_node' column")
            center_node_ids = node_idx_df["nSCHISM_hgrid_node"].to_numpy(dtype=int)
            # Map node IDs to positional indices in self.ds.nSCHISM_hgrid_node
            ds_node_ids = self.ds.nSCHISM_hgrid_node.values
            id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}
            self.center_nodes = np.array([id_to_pos[int(nid)] for nid in center_node_ids if int(nid) in id_to_pos])
            log.info(f"Selected {len(self.center_nodes)} center nodes from {node_idx_path}.")
        else:
            self.center_nodes = np.arange(self.n_nodes)
        self.n_center_nodes = len(self.center_nodes)

        # --- Load edges once (time-invariant) and build adjacency ---
        self._build_adjacency()

        # --- Extract node coordinates (not time-varying) ---
        self.node_x = self.ds["SCHISM_hgrid_node_x"].values
        self.node_y = self.ds["SCHISM_hgrid_node_y"].values

        # --- Cache subgraph topology per center node ---
        log.info(f"Precomputing subgraph topology for {self.n_center_nodes} center nodes...")
        self.graph_cache: Dict[int, Tuple[np.ndarray, torch.Tensor, int]] = {}
        for center_pos in self.center_nodes:
            self.graph_cache[int(center_pos)] = self._build_subgraph(int(center_pos))
        log.info(f"Cached {len(self.graph_cache)} subgraph topologies.")

        # Drop graph_features from dataset to free memory
        self.ds = self.ds.drop_vars(
            [v for v in self.graph_features if v in self.ds],
            errors="ignore",
        )

        # Precompute total samples as a plain int.
        # PyG's Dataset.__getattr__ intercepts attribute access, so __len__/len
        # must not traverse attribute chains like self.ds.time at call time.
        self._n_time = len(self.ds.time)
        self._total_samples = self.n_center_nodes * self._n_time

        # Initialize GDataset last, after all attributes are set,
        # because PyG's Dataset.__init__ may trigger len() or other methods.
        GDataset.__init__(self)

        log.info(
            f"SmallGraphWWMDataset initialized: {self.n_center_nodes} center nodes, "
            f"{self._n_time} time steps, {self._total_samples} total samples."
        )

    def _build_adjacency(self) -> None:
        """Load edges from timestep 0 and build an adjacency list.

        Edge node IDs are 1-indexed in the dataset; they are converted to
        0-indexed positional indices into ``self.ds.nSCHISM_hgrid_node``.
        """
        start_time = time.time()

        raw_edges = self.ds.isel(time=0)["SCHISM_hgrid_edge_nodes"].values.astype(int)

        # Map 1-indexed node IDs to 0-indexed positions
        ds_node_ids = self.ds.nSCHISM_hgrid_node.values
        id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}

        self.adjacency: Dict[int, List[int]] = defaultdict(list)
        valid_count = 0
        for a_id, b_id in raw_edges:
            a_pos = id_to_pos.get(int(a_id))
            b_pos = id_to_pos.get(int(b_id))
            if a_pos is not None and b_pos is not None:
                self.adjacency[a_pos].append(b_pos)
                self.adjacency[b_pos].append(a_pos)
                valid_count += 1

        elapsed = time.time() - start_time
        log.info(
            f"Built adjacency list in {elapsed:.1f}s: "
            f"{valid_count} valid edges, {len(self.adjacency)} nodes with neighbors."
        )

    def _build_subgraph(self, center_pos: int) -> Tuple[np.ndarray, torch.Tensor, int]:
        """Build subgraph topology around a center node using BFS.

        BFS expands along mesh edges, bounded by:
        - ``max_radius``: max Euclidean distance (meters) from center node
        - ``max_nodes``: max number of nodes in the subgraph
        - ``max_links``: max number of BFS hops from the center node

        Args:
            center_pos: 0-indexed positional index of the center node.

        Returns:
            Tuple of (subgraph_nodes, edge_index, center_local_idx) where:
                - subgraph_nodes: sorted array of 0-indexed positional node indices
                - edge_index: (2, n_edges) tensor of local edge indices
                - center_local_idx: local index of the center node in subgraph_nodes
        """
        cx, cy = self.node_x[center_pos], self.node_y[center_pos]

        visited = {center_pos}
        queue = deque([(center_pos, 0)])

        while queue and len(visited) < self.max_nodes:
            current, depth = queue.popleft()

            if depth >= self.max_links:
                continue

            for neighbor in self.adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # Check distance from center node
                dx = self.node_x[neighbor] - cx
                dy = self.node_y[neighbor] - cy
                dist = np.sqrt(dx * dx + dy * dy)
                if dist >= self.max_radius:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

                if len(visited) >= self.max_nodes:
                    break

        # Build local node mapping
        subgraph_nodes = np.sort(np.array(list(visited)))
        global_to_local = {g: l for l, g in enumerate(subgraph_nodes)}  # noqa: E741
        center_local_idx = global_to_local[center_pos]

        # Build edge_index: include all edges where both endpoints are in subgraph
        # plus self-loops on the diagonal (required by many PyG message-passing layers)
        subgraph_set = set(subgraph_nodes)
        src_list = []
        dst_list = []
        for u in subgraph_nodes:
            local_u = global_to_local[u]
            # Self-loop
            src_list.append(local_u)
            dst_list.append(local_u)
            # Neighbor edges
            for v in self.adjacency.get(u, []):
                if v in subgraph_set:
                    src_list.append(local_u)
                    dst_list.append(global_to_local[v])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        return subgraph_nodes, edge_index, center_local_idx

    def construct_graph(self, time_idx: int, center_node_pos: int) -> GData:
        """Construct a graph around the center node with features from a given time step.

        Uses precomputed subgraph topology from the cache. Only node features and
        center-node targets are loaded from the dataset.

        Args:
            time_idx: Time index in dataset (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            torch_geometric.data.Data with attributes:
                - x: (n_subgraph_nodes, n_features) node feature tensor
                - edge_index: (2, n_edges) edge index tensor
                - y: (n_targets,) target tensor for center node
                - center_idx: local index of center node in x
        """
        subgraph_nodes, edge_index, center_local_idx = self.graph_cache[center_node_pos]

        # Extract node features for subgraph at this time step
        sample = self.ds.isel(time=time_idx, nSCHISM_hgrid_node=subgraph_nodes)
        x_da = sample[self.features].to_dataarray(dim="variable")
        # x_da shape: (variable, nSCHISM_hgrid_node) → transpose to (n_nodes, n_features)
        x = torch.from_numpy(x_da.values.T.copy()).float()

        # Extract target for center node only
        center_sample = self.ds.isel(time=time_idx, nSCHISM_hgrid_node=center_node_pos)
        y_da = center_sample[self.targets].to_dataarray(dim="variable")
        y = torch.from_numpy(y_da.values.copy()).float()

        return GData(x=x, edge_index=edge_index, y=y, center_idx=center_local_idx)

    def __len__(self) -> int:
        """Return the total number of samples (center_nodes * time_steps)."""
        return self._total_samples

    def len(self) -> int:
        """Return the total number of samples (center_nodes * time_steps)."""
        return self._total_samples

    def get(self, idx: int) -> GData:
        """Get a graph sample by flat index.

        The index maps to (time_idx, center_node_pos):
            time_idx = idx // n_center_nodes
            center_pos = center_nodes[idx % n_center_nodes]

        Args:
            idx: Flat index into the dataset.

        Returns:
            GData: Graph centered on the selected node with features from the time step.
        """
        time_idx = idx // self.n_center_nodes
        center_pos = self.center_nodes[idx % self.n_center_nodes]
        return self.construct_graph(time_idx=time_idx, center_node_pos=center_pos)
