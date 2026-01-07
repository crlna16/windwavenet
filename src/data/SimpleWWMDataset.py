# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import yaml
from torch.utils.data import Dataset

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class SimpleWWMDataset(Dataset):
    """Simple Wave Wind Model Dataset that loads netCDF files for a time range.

    Loads all netCDF files from a directory between start_day and end_day indices.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
    ) -> None:
        """Initialize the SimpleWWM Dataset.

        Args:
            data_dir: Directory containing netCDF data files.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [0, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            sample_weight_strategy: Strategy for computing per-sample weights.
                Options: None (no weighting), 'abs_value' (weight by absolute value of targets).
            sample_weight_power: Exponent to control the intensity of sample weighting (default: 1.0).
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.features = features
        self.targets = targets
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power

        self.files = self._collect_files(self.data_dir, self.data_range)
        self.ds = self._open_dataset(self.files)
        self.ds = self._compute_prevT(self.ds, self.features)
        self.ds = self._compute_deltaT(self.ds, self.targets)

        # keep original variables always for reconstruction
        selected_targets = self.targets + [
            tt.replace("delta_", "") for tt in self.targets
        ]
        self.ds = self._select_variables(self.ds, self.features + selected_targets)

        # clear nans from computing difference with respect to time
        self.ds = self._remove_first_timestep_if_nan(self.ds)

        self.scaling_params = (
            self._load_scaling_config(scaling_config) if scaling_config else None
        )

        if self.scaling_params:
            self.ds = self._apply_scaling(self.ds, self.features, self.scaling_params)

        # Load data in memory
        # TODO support lazy loading
        self.X = self._retrieve_data(self.ds, self.features)
        self.y = self._retrieve_data(self.ds, self.targets)

        # Compute weights
        self.compute_weights()

        log.info(f"Dataset loaded with X: {self.X.shape} / y: {self.y.shape}")

    @staticmethod
    def _collect_files(data_dir: Path, data_range: List[int]) -> List[Path]:
        """Collect netCDF files for specified day range.

        Args:
            data_dir: Directory containing netCDF data files.
            data_range: Day range as [start_day, end_day] inclusive.

        Returns:
            List of file paths for the specified day range.
        """
        all_files = sorted(data_dir.glob("out2d_*.nc"))

        if not all_files:
            raise FileNotFoundError(f"No netCDF files found in {data_dir}")

        if len(data_range) != 2:
            raise ValueError(
                f"data_range must be [start_day, end_day], got {data_range}"
            )

        start_day, end_day = data_range

        # Select files for the day range (inclusive)
        all_days = list(range(start_day, end_day + 1))
        files = []

        for d in all_days:
            files.extend(data_dir.glob(f"out2d_{d}.nc"))

        if not files:
            raise ValueError(
                f"No files found for days {start_day}-{end_day}. Total files available: {len(all_files)}"
            )

        files = sorted(files)

        log.info(
            f"Collected {len(files)} files from {data_dir} for days {start_day}-{end_day}"
        )

        return files

    @staticmethod
    def _open_dataset(files: List[Path]) -> xr.Dataset:
        """Open dataset from netCDF files without variable selection.

        Args:
            files: List of file paths to load.

        Returns:
            xarray Dataset containing all variables.
        """
        ds = xr.open_mfdataset(files).chunk("auto")
        return ds

    @staticmethod
    def _compute_deltaT(dataset: xr.Dataset, targets: List[str]) -> xr.Dataset:
        """Compute Delta(target) = target(t) - target(t-1).

        For targets starting with 'delta_', computes the difference with respect to the
        previous time step and stores that value in a new variable.

        E.g. for sigWaveHeight and time=t:
          ds["delta_sigWaveHeight"][t] = ds["sigWaveHeight"][t] - ds["sigWaveHeight"][t-1]

        Args:
            dataset: xarray Dataset containing the data.
            targets: List of target variable names.

        Returns:
            Dataset with difference wrto time per variable added.
        """
        for target in targets:
            if target.startswith("delta_"):
                original_var = target.replace("delta_", "")
                dataset[target] = dataset[original_var].diff("time")
                log.info(f"Added variable {target}")

        return dataset

    @staticmethod
    def _compute_prevT(dataset: xr.Dataset, features: List[str]) -> xr.Dataset:
        """Compute prev(feature) = feature(t-1)

        E.g. for sigWaveHeight and time=t:
          ds["prev_sigWaveHeight"][t]  = ds["sigWaveHeight"][t-1]

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
        will contain NaN values. This method checks and removes it if needed.

        Args:
            dataset: xarray Dataset containing the data.

        Returns:
            Dataset with first time step removed if it contained NaN values.
        """
        first_time = dataset.isel(time=0)
        if first_time.to_array().isnull().any():
            log.info("Removing first time step (contains NaN values)")
            return dataset.isel(time=slice(1, None))
        return dataset

    @staticmethod
    def _retrieve_data(dataset: xr.Dataset, variables: List[str]) -> np.ndarray:
        """Extract variable data from xarray Dataset.

        Args:
            dataset: xarray Dataset containing the data.
            variables: List of variable names to extract.

        Returns:
            Numpy array of shape (n_samples, n_variables).
        """
        x = []
        for var in variables:
            xx = dataset[var].values
            x.append(xx)
        return np.asarray(x).reshape(len(variables), -1).transpose()

    @staticmethod
    def _load_scaling_config(config_path: str) -> Dict[str, Dict[str, float]]:
        """Load scaling parameters from YAML file.

        Args:
            config_path: Path to YAML file containing mean and std for each ggvariable.

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
                log.warning(
                    f"No scaling parameters found for variable '{var}', leaving unscaled"
                )
        return dataset

    @staticmethod
    def _revert_scaling(
        dataset: xr.Dataset,
        variables: List[str],
        scaling_params: Dict[str, Dict[str, float]],
    ) -> xr.Dataset:
        """Revert standardization scaling of variables in the dataset.

        Args:
            dataset (xr.Dataset): xarray Dataset containing the variabels.
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
                log.info(f"Reverted scaling for varibale {var}.")
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

        log.info(
            f"Computing sample weights with strategy '{self.sample_weight_strategy}'..."
        )

        n_samples = len(self.y)
        n_targets = len(self.targets)
        weights = np.ones((n_samples, n_targets), dtype=np.float32)

        if self.sample_weight_strategy == "abs_value":
            # Weight by (1 + |target|)^power
            targets = self.y
            weights = (1.0 + np.abs(targets)) ** self.sample_weight_power

        else:
            raise ValueError(
                f"Unknown sample_weight_strategy: '{self.sample_weight_strategy}'. "
                f"Supported: None, 'abs_value'"
            )

        # Normalize weights per target so that sum = n_samples (mean = 1.0)
        for i in range(n_targets):
            weight_sum = weights[:, i].sum()
            weights[:, i] = weights[:, i] * (n_samples / weight_sum)
            log.info(
                f"Target {self.targets[i]}: normalized weights from sum={weight_sum:.2f} to {n_samples} "
                f"(mean={weights[:, i].mean():.4f}, std={weights[:, i].std():.4f}, "
                f"min={weights[:, i].min():.4f}, max={weights[:, i].max():.4f})"
            )

        self.weights = weights
        log.info(f"Sample weights computed and normalized for {n_samples} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets, weights) if weights are computed,
            otherwise (features, targets).
        """
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
        return self.X[idx], self.y[idx]


class SimpleWWMPredictDataset(SimpleWWMDataset):
    """Dataset to be used for predictions.

    Args:
        SimpleWWMDataset (torch.Dataset): Dataset class handling grid-point-wise WWM data.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
    ) -> None:
        """Initialize the SimpleWWMPredictDataset.

        Args:
            data_dir: Root directory containing netCDF data files.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [0, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            sample_weight_strategy: Strategy for computing per-sample weights (inherited from parent).
            sample_weight_power: Exponent to control the intensity of sample weighting (inherited from parent).
        """
        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
        )

    def __getitem__(self, idx: int) -> Tuple:
        """Overwritten __getitem__. Returns features, targets, and optionally weights.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets, weights) if weights are computed,
            otherwise (features, targets).
        """
        # Use parent's __getitem__ implementation
        return super().__getitem__(idx)
