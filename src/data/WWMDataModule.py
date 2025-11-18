# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.data.SimpleWWMDataset import SimpleWWMDataset


class WWMDataModule(L.LightningDataModule):
    """Lightning DataModule for Wave Wind Model data with time-based splits.

    Train, validation, and test datasets are separated by time periods (days).
    All data is loaded from the same root directory.
    """

    def __init__(
        self,
        data_dir: str,
        train_days: Tuple[int, int],
        val_days: Tuple[int, int],
        test_days: Tuple[int, int],
        features: List[str],
        targets: List[str],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the WWM DataModule.

        Args:
            data_dir: Root directory containing all data files.
            train_days: Tuple of (start_day, end_day) for training data.
            val_days: Tuple of (start_day, end_day) for validation data.
            test_days: Tuple of (start_day, end_day) for test data.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            batch_size: Number of samples per batch.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)

        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days

        self.features = features
        self.targets = targets

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures this runs only on 1 GPU.

        This is where you might download or preprocess data that should only
        happen once (e.g., downloading files, creating cache).
        """
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.ds_train`, `self.ds_val`, `self.ds_test`.

        This method is called by Lightning with both `trainer.fit()` and `trainer.test()`,
        so be careful not to execute things like random split twice!
        Also, it is called on every GPU in distributed training.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage == "fit":
            self.ds_train = SimpleWWMDataset(
                data_dir=self.data_dir,
                start_day=self.train_days[0],
                end_day=self.train_days[1],
                features=self.features,
                targets=self.targets,
            )

            self.ds_val = SimpleWWMDataset(
                data_dir=self.data_dir,
                start_day=self.val_days[0],
                end_day=self.val_days[1],
                features=self.features,
                targets=self.targets,
            )

            self.ds_test = SimpleWWMDataset(
                data_dir=self.data_dir,
                start_day=self.test_days[0],
                end_day=self.test_days[1],
                features=self.features,
                targets=self.targets,
            )

        elif stage == "test":
            self.ds_test = SimpleWWMDataset(
                data_dir=self.data_dir,
                start_day=self.test_days[0],
                end_day=self.test_days[1],
                features=self.features,
                targets=self.targets,
            )

        elif stage in ["validate", "predict"]:
            raise NotImplementedError(f"Stage '{stage}' is not implemented.")

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        return DataLoader(
            dataset=self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
