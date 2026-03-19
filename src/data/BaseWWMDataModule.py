# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class BaseWWMDataModule(L.LightningDataModule, ABC):
    """Abstract base class for WWM DataModules.

    Provides common functionality for data loading, with dataset creation
    delegated to subclasses via _create_dataset method.
    """

    def __init__(
        self,
        data_dir: str,
        train_days: List[int],
        val_days: List[int],
        test_days: List[int],
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = True,
    ) -> None:
        """Initialize the base WWM DataModule.

        Args:
            data_dir: Directory containing the Zarr dataset.
            train_days: Day range for training as [start_day, end_day] inclusive.
            val_days: Day range for validation as [start_day, end_day] inclusive.
            test_days: Day range for testing as [start_day, end_day] inclusive.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep for train/val.
                Test/predict use all nodes (node_idx_csv=None).
            batch_size: Number of samples per batch.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            persistent_workers: If True, workers are not shut down after each epoch.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            subsample_prob: Optional probability threshold for subsampling training data.
            preload_data: If True, preload data in memory for fast access.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days

        self.features = features
        self.targets = targets
        self.scaling_config = scaling_config
        self.node_idx_csv = node_idx_csv

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power
        self.subsample_prob = subsample_prob
        self.preload_data = preload_data

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None
        self.ds_pred: Optional[Dataset] = None

        if self.num_workers == 0:
            self.prefetch_factor = None
        else:
            self.prefetch_factor = 4

    @abstractmethod
    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        subsample_prob: Optional[float],
        preload_data: bool,
    ) -> Dataset:
        """Create a dataset instance. Subclasses must implement this.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            subsample_prob: Optional probability threshold for subsampling.
            preload_data: If True, preload data in memory.

        Returns:
            A Dataset instance.
        """
        pass

    def prepare_data(self) -> None:
        """Check if data directory exists. Lightning ensures this runs only on 1 GPU."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and set up datasets for each stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == "fit":
            self.ds_train = self._create_dataset(
                data_range=self.train_days,
                node_idx_csv=self.node_idx_csv,
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_power=self.sample_weight_power,
                subsample_prob=self.subsample_prob,
                preload_data=self.preload_data,
            )

            self.ds_val = self._create_dataset(
                data_range=self.val_days,
                node_idx_csv=self.node_idx_csv,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
            )

            self.ds_test = self._create_dataset(
                data_range=self.test_days,
                node_idx_csv=self.node_idx_csv,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
            )

        elif stage == "test":
            # Test on all nodes for full evaluation
            self.ds_test = self._create_dataset(
                data_range=self.test_days,
                node_idx_csv=None,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
            )

        elif stage == "predict":
            log.info("Predict dataset uses test day range with all nodes.")
            self.ds_pred = self._create_dataset(
                data_range=self.test_days,
                node_idx_csv=None,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
            )

        elif stage == "validate":
            raise NotImplementedError(f"Stage '{stage}' is not implemented.")

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""
        worker_init_fn = None
        if self.num_workers > 0 and not self.preload_data:
            if hasattr(self.ds_train, "worker_init_fn"):
                worker_init_fn = lambda worker_id: self.ds_train.worker_init_fn(worker_id)  # noqa: E731

        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.ds_val,
            batch_size=4 * self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.ds_test,
            batch_size=4 * self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create and return the predict dataloader."""
        return DataLoader(
            dataset=self.ds_pred,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
