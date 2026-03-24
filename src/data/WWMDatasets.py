# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

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
    """Dataset that returns graphs that represent a subregion around a center node."""

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        graph_features: List[str],  # TODO default values
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = False,
        max_radius: float = 10000.0,
        max_nodes: int = 1000,
        max_links: int = 1000,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset. Must contain graph definition.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names to use for constructing the graph. TODO
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with a column named 'nSCHISM_hgrid_node'
                containing node indices to keep. If None, keep all nodes.
            sample_weight_strategy: Strategy for computing per-sample weights.
                Options: None (no weighting), 'z_score' (weight by z-score of targets).
            sample_weight_power: Exponent to control the intensity of sample weighting (default: 1.0).
            subsample_prob: Optional probability threshold for subsampling based on z_score.
                Must be None for SmallGraphWWMDataset.
            preload_data: If True, preload X/y in memory. If False, load X/y lazily in __getitem__.
                With multiple workers, each worker slices the time axis to avoid shared Zarr chunks.
                Must be False for SmallGraphWWMDataset.
            max_radius: Maximum radius in grid units that a node can have from the
                graph center node (default: 10000.)
            max_nodes: Maximum number of nodes in a graph (defaut: 1000).
            max_links: Maximum number of edges between the center node and a node in the graph (default: 1000)
        """
        SimpleWWMDataset.__init__(
            self,
            data_dir,
            data_range,
            features + graph_features,
            targets,
            scaling_config,
            node_idx_csv,
            sample_weight_strategy,
            sample_weight_power,
            subsample_prob,
            preload_data,
        )
        GDataset.__init__(self)

        if self.subsample_prob is not None:
            raise ValueError("Subsample probability must be None for GraphWWMDataset. Found", self.subsample_prob)

        self.max_radius = max_radius
        self.max_nodes = max_nodes
        self.max_links = max_links

        # reset features (for embeddings) and graph_features (for graph geometry)
        self.features = features
        self.graph_features = graph_features

        # graph connections
        self.ds = self.ds.assign(node_idx=self.ds[features[0]].isel(time=0))
        self.ds.node_idx.values = np.arange(1, self.ds.sizes["nSCHISM_hgrid_node"] + 1)

        self.n_nodes = len(self.ds.nSCHISM_hgrid_node)
        self.possible_graphs = self.n_nodes * len(self.ds.time)

    def construct_graph(self, time_idx=0, node_idx=0) -> GData:
        """
        Construct a graph around the center node at node_idx, with node features from time time_idx.

        The graph encompasses all nodes that are within the distance of the center node
        - max_radius defines the maximum distance (dimensionless units)
        - max_nodes defines the maximum number of nodes the graph may hold
        - max_links defines the maximum number of edges between a node and the center node

        Args:
            time_idx (int): Time index in dataset. Used to select values for node features.
            node_idx (int): Node index (nSCHISM_hgrid_node).

        Returns:
            torch_geometric.data.Data: Graph instance centered around the center node with associated
              node features.

        """

        # Select the correct center node based on node_idx
        center_node = self.ds.isel(nSCHISM_hgrid_node=node_idx).node_idx.values  # starts from 1 (like edges)
        # Select the correct embeddings based on time_idx
        embeddings = self.ds.isel(time=time_idx)

        edges = self.ds.isel(time=time_idx).SCHISM_hgrid_edge_nodes.values.astype(int)

        constructing = True

        nodes_in_graph = []
        nodes_to_check = [center_node]

        while constructing and len(nodes_to_check) > 0:
            center_node = self.ds.isel(nSCHISM_hgrid_node=nodes_to_check[0] - 1).node_idx.values

            cx = embeddings.isel(nSCHISM_hgrid_node=center_node - 1)["SCHISM_hgrid_node_x"]
            cy = embeddings.isel(nSCHISM_hgrid_node=center_node - 1)["SCHISM_hgrid_node_y"]

            # get all the node_idx that share an edge with center_node
            candidates = np.unique(edges[(edges == center_node).sum(axis=1).astype(bool), :].flatten())
            # restrict to new candidates
            candidates = np.array([c for c in candidates if c not in nodes_in_graph])

            subset = embeddings.isel(nSCHISM_hgrid_node=candidates - 1)  # starts from 0
            distances = np.sqrt(
                (subset["SCHISM_hgrid_node_x"] - cx) ** 2 + (subset["SCHISM_hgrid_node_y"] - cy) ** 2
            ).values

            for distance, candidate in zip(distances, candidates):
                # within radius
                if distance < self.max_radius and distance > 0:
                    nodes_to_check.append(candidate)

            # append current node to graph
            nodes_in_graph.append(center_node)

            if len(nodes_in_graph) >= self.max_nodes:
                constructing = False

            # we are done with this one
            nodes_to_check.pop(0)

            print(nodes_to_check)

        edge_index = torch.tensor([[3, 1, 1, 2], [2, 1, 2, 1]], dtype=torch.long)
        x = torch.tensor([[np.random.rand()], [0], [1]], dtype=torch.float)

        graph = GData(x=x, edge_index=edge_index)

        return graph

    def len(self) -> int:
        return self.possible_graphs

    def get(self, idx) -> GData:
        """
        Construct the graph at idx with corresponding embeddings. idx wraps along the nodes
         => time_idx for the feature values
         => node_idx for the center node selection

        Args:
            idx: Index in dataset.

        Returns:
            Data: Graph centered on the center node with features from given time.
        """
        time_idx = idx // self.n_nodes
        node_idx = idx % self.n_nodes

        graph = self.construct_graph(time_idx=time_idx, node_idx=node_idx)
        return graph
