# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Tuple

import xarray as xr
import numpy as np
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
        start_day: int,
        end_day: int,
        features: List[str],
        targets: List[str],
    ) -> None:
        """Initialize the SimpleWWM Dataset.

        Args:
            data_dir: Root directory containing netCDF data files.
            start_day: Starting day index (inclusive).
            end_day: Ending day index (exclusive).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.start_day = start_day
        self.end_day = end_day
        self.features = features
        self.targets = targets

        # Collect all files in the specified day range
        self.files = self._collect_files(self.data_dir, self.start_day, self.end_day)
        self.ds = self._load_data(self.files)

        # Extract points
        self.X = self._retrieve_data(self.ds, self.features)
        self.y = self._retrieve_data(self.ds, self.targets)

        log.info(f"Dataset loaded with X: {self.X.shape} / y: {self.y.shape}")

    @staticmethod
    def _collect_files(data_dir: Path, start_day: int, end_day: int) -> List[Path]:
        """Collect all netCDF files in the specified day range.

        Args:
            data_dir: Root directory containing netCDF data files.
            start_day: Starting day index (inclusive).
            end_day: Ending day index (exclusive).

        Returns:
            List of file paths for the specified day range.
        """
        files = []
        for day_idx in range(start_day, end_day):
            file_pattern = f"out2d_{day_idx:2d}.nc"  # TODO maybe pass this
            matching_files = sorted(data_dir.glob(file_pattern))
            files.extend(matching_files)

        if not files:
            raise FileNotFoundError(
                f"No files found in {data_dir} for days {start_day}-{end_day}"
            )

        return files

    @staticmethod
    def _load_data(files: List[Path]) -> xr.Dataset:
        """Load data from netCDF files.

        Args:
            files: List of file paths to load.

        Returns:
            xarray Dataset containing the loaded data.
        """
        ds = xr.open_mfdataset(files).chunk("auto")
        return ds

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
