# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

from torch.utils.data import Dataset

from src.data.BaseWWMDataModule import BaseWWMDataModule
from src.data.WWMDatasets import SequenceWWMDataset


class SequenceWWMDataModule(BaseWWMDataModule):
    """Lightning DataModule for sequence-based Wave Wind Model data (LSTM/RNN).

    Returns time sequences of features with next-step targets.
    Train, validation, and test datasets are separated by day ranges.
    """

    def __init__(
        self,
        data_dir: str,
        train_days: List[int],
        val_days: List[int],
        test_days: List[int],
        features: List[str],
        targets: List[str],
        seq_len: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        preload_data: bool = True,
    ) -> None:
        """Initialize the Sequence WWM DataModule.

        Args:
            data_dir: Directory containing the Zarr dataset.
            train_days: Day range for training as [start_day, end_day] inclusive.
            val_days: Day range for validation as [start_day, end_day] inclusive.
            test_days: Day range for testing as [start_day, end_day] inclusive.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            seq_len: Length of input sequences (number of time steps).
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep for train/val.
                Test uses all nodes (node_idx_csv=None) for full evaluation.
            batch_size: Number of samples per batch.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            persistent_workers: If True, workers are not shut down after each epoch.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            preload_data: If True, preload data in memory for fast access.
        """
        # Store seq_len before calling super().__init__ since it uses save_hyperparameters
        self.seq_len = seq_len

        super().__init__(
            data_dir=data_dir,
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            preload_data=preload_data,
        )

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        preload_data: bool,
    ) -> Dataset:
        """Create a SequenceWWMDataset instance.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            preload_data: If True, preload data in memory.

        Returns:
            A SequenceWWMDataset instance.
        """
        return SequenceWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            seq_len=self.seq_len,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            preload_data=preload_data,
        )
