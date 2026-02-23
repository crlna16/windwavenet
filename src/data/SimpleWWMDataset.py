# SPDX-FileCopyrightText: 2025-2026 Hereon
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
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power
        self.subsample_prob = subsample_prob
        self.preload_data = preload_data

        # list comprehension covers omegaconf list
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

        self.ds = self._compute_prevT(self.ds, self.features)
        self.ds = self._compute_deltaT(self.ds, self.targets)

        # keep original variables always for reconstruction
        selected_targets = self.targets + [tt.replace("delta_", "") for tt in self.targets]
        self.ds = self._select_variables(self.ds, self.features + selected_targets)

        # clear nans from computing difference with respect to time
        if np.any([tt.startswith("delta_") for tt in self.targets]) or np.any(
            [ff.startswith("prev_") for ff in self.features]
        ):
            self.ds = self.ds.isel(time=slice(1, None))

        self.n_nodes = len(self.ds.nSCHISM_hgrid_node)

        # scaling based on precomputed values
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


class SimpleWWMPredictDataset(SimpleWWMDataset):
    """Dataset to be used for predictions.

    Args:
        SimpleWWMDataset (torch.Dataset): Base dataset class handling grid-point-wise WWM data.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = True,
    ) -> None:
        """Initialize the prediction dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            sample_weight_strategy: Strategy for computing per-sample weights (inherited from parent).
            sample_weight_power: Exponent to control the intensity of sample weighting (inherited from parent).
            subsample_prob: Optional probability threshold for subsampling (inherited from parent).
            preload_data: If True, preload X/y in memory. If False, load X/y lazily in __getitem__.
        """
        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            subsample_prob=subsample_prob,
            preload_data=preload_data,
        )

    def __getitem__(self, idx: int) -> Tuple:
        """Return features, targets, and optionally weights.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets, weights) if weights are computed,
            otherwise (features, targets).
        """
        # Use parent's __getitem__ implementation
        return super().__getitem__(idx)
