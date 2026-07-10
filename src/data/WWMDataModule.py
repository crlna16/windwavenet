# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

from torch.utils.data import Dataset

from src.data.BaseWWMDataModule import BaseWWMDataModule
from src.data.WWMDatasets import SimpleWWMDataset


class WWMDataModule(BaseWWMDataModule):
    """Lightning DataModule for point-wise Wave Wind Model data.

    Each sample is a single (time, node) pair with features and targets.
    Train, validation, and test datasets are separated by day ranges.
    """

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        sample_weight_variable: str = "sigWaveHeight",
    ) -> Dataset:
        """Create a SimpleWWMDataset instance.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.

        Returns:
            A SimpleWWMDataset instance.
        """
        return SimpleWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=self.boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )
