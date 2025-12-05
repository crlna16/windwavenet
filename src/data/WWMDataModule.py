# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import List, Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.utils.pylogger import RankedLogger
from src.data.SimpleWWMDataset import SimpleWWMDataset

log = RankedLogger(__name__, rank_zero_only=False)


class WWMDataModule(L.LightningDataModule):
    """Lightning DataModule for Wave Wind Model data with time-based splits.

    Train, validation, and test datasets are separated by time periods (days).
    All data is loaded from the same root directory.
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        test_dir: str,
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize the WWM DataModule.

        Args:
            train_dir: Directory containing training data netCDF files.
            val_dir: Directory containing validation data netCDF files.
            test_dir: Directory containing test data netCDF files.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            batch_size: Number of samples per batch.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.test_dir = Path(test_dir)

        self.features = features
        self.targets = targets
        self.scaling_config = scaling_config

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
        # Check if data directories exist
        for dir_name, dir_path in [
            ("train", self.train_dir),
            ("val", self.val_dir),
            ("test", self.test_dir),
        ]:
            if not dir_path.exists():
                raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")

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
                data_dir=self.train_dir,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
            )

            self.ds_val = SimpleWWMDataset(
                data_dir=self.val_dir,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
            )

            self.ds_test = SimpleWWMDataset(
                data_dir=self.test_dir,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
            )

        elif stage == "test":
            self.ds_test = SimpleWWMDataset(
                data_dir=self.test_dir,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
            )

        elif stage == "predict":
            log.info("Predict dataset is the same as test dataset.")
            self.ds_pred = SimpleWWMDataset(
                data_dir=self.test_dir,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
            )

        elif stage in ["validate"]:
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

    def predict_dataloader(self) -> DataLoader:
        """Create and return the predict dataloader.

        Returns:
            The predict dataloader.
        """
        return DataLoader(
            dataset=self.ds_pred,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
