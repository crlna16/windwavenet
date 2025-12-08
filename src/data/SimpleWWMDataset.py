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
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
    ) -> None:
        """Initialize the SimpleWWM Dataset.

        Args:
            data_dir: Directory containing netCDF data files to load.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.features = features
        self.targets = targets

        self.files = self._collect_files(self.data_dir)
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

        log.info(f"Dataset loaded with X: {self.X.shape} / y: {self.y.shape}")

    @staticmethod
    def _collect_files(data_dir: Path) -> List[Path]:
        """Collect all netCDF files in the specified directory.

        Args:
            data_dir: Directory containing netCDF data files.

        Returns:
            List of file paths sorted alphabetically.
        """
        files = sorted(data_dir.glob("*.nc"))

        if not files:
            raise FileNotFoundError(f"No netCDF files found in {data_dir}")

        log.info(f"Collected {len(files)} files from {data_dir}")

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
        """Apply standardization scaling to variables in xarray Dataset.

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

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets) for the given index.
        """
        return self.X[idx], self.y[idx]


class SimpleWWMPredictDataset(SimpleWWMDataset):
    """Dataset to be used for predictions.

    Args:
        SimpleWWMDataset (torch.Dataset): Dataset class handling grid-point-wise WWM data.
    """

    def __init__(
        self,
        data_dir: Path,
        start_day: int,
        end_day: int,
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
    ) -> None:
        """Initialize the SimpleWWMPredictDataset.

        Args:
            data_dir: Root directory containing netCDF data files.
            start_day: Starting day index (inclusive).
            end_day: Ending day index (exclusive).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
        """
        super().__init__(
            data_dir=data_dir,
            start_day=start_day,
            end_day=end_day,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
        )

    def __getitem__(self, idx):
        """Overwritten __getitem__. Returns previous value on top

        Args:
            idx (_type_): _description_
        """
