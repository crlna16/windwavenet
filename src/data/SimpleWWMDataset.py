# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Tuple

import xarray as xr
from torch.utils.data import Dataset


class SimpleWWMDataset(Dataset):
    """Simple Wave Wind Model Dataset that loads netCDF files for a time range.

    Loads all netCDF files from a directory between start_day and end_day indices.
    """

    def __init__(
        self,
        data_dir: Path,
        start_day: int,
        end_day: int,
        feature_vars: List[str],
        target_vars: List[str],
    ) -> None:
        """Initialize the SimpleWWM Dataset.

        Args:
            data_dir: Root directory containing netCDF data files.
            start_day: Starting day index (inclusive).
            end_day: Ending day index (exclusive).
            feature_vars: List of variable names to use as features.
            target_vars: List of variable names to use as targets.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.start_day = start_day
        self.end_day = end_day
        self.feature_vars = feature_vars
        self.target_vars = target_vars

        # Collect all files in the specified day range
        self.files = self._collect_files()

        # TODO: Load and process the data from files
        # Example structure for when you're ready:
        # self.data = self._load_data()

    def _collect_files(self) -> List[Path]:
        """Collect all netCDF files in the specified day range.

        Returns:
            List of file paths for the specified day range.
        """
        files = []
        for day_idx in range(self.start_day, self.end_day):
            # Adjust the file pattern to match your naming convention
            # Example: file_0001.nc, file_0002.nc, etc.
            file_pattern = f"*{day_idx:04d}.nc"  # Modify pattern as needed
            matching_files = sorted(self.data_dir.glob(file_pattern))
            files.extend(matching_files)

        if not files:
            raise FileNotFoundError(
                f"No files found in {self.data_dir} for days {self.start_day}-{self.end_day}"
            )

        return files

    def _load_data(self) -> xr.Dataset:
        """Load data from netCDF files.

        Returns:
            xarray Dataset containing the loaded data.
        """
        # TODO: Implement your loading logic here
        # Example:
        # datasets = [xr.open_dataset(f) for f in self.files]
        # combined = xr.concat(datasets, dim='time')
        # return combined
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        # TODO: Implement based on your data structure
        raise NotImplementedError("__len__ needs to be implemented")

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets) for the given index.
        """
        # TODO: Implement based on your data structure
        # Example:
        # features = self.data[self.feature_vars].isel(time=idx)
        # targets = self.data[self.target_vars].isel(time=idx)
        # return features.values, targets.values
        raise NotImplementedError("__getitem__ needs to be implemented")
