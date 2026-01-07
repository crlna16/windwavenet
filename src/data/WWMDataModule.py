# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, List, Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.utils.pylogger import RankedLogger
from src.data.SimpleWWMDataset import SimpleWWMDataset

log = RankedLogger(__name__, rank_zero_only=False)


class WWMDataModule(L.LightningDataModule):
    """Lightning DataModule for Wave Wind Model data with time-based splits.

    Train, validation, and test datasets are separated by day ranges.
    All data is loaded from the same root directory.
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
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
    ) -> None:
        """Initialize the WWM DataModule.

        Args:
            data_dir: Directory containing all netCDF data files.
            train_days: Day range for training as [start_day, end_day] inclusive (e.g., [0, 10]).
            val_days: Day range for validation as [start_day, end_day] inclusive (e.g., [11, 13]).
            test_days: Day range for testing as [start_day, end_day] inclusive (e.g., [14, 15]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            batch_size: Number of samples per batch.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
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

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power

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
                data_range=self.train_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_power=self.sample_weight_power,
            )

            self.ds_val = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.val_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_power=self.sample_weight_power,
            )

            self.ds_test = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.test_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_power=self.sample_weight_power,
            )

        elif stage == "test":
            self.ds_test = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.test_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_power=self.sample_weight_power,
            )

        elif stage == "predict":
            log.info("Predict dataset is the same as test dataset.")
            self.ds_pred = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.test_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_power=self.sample_weight_power,
            )

        elif stage in ["validate"]:
            raise NotImplementedError(f"Stage '{stage}' is not implemented.")

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def compute_sample_weights(self, model: Any, device: str = "cpu") -> None:
        """Compute sample weights for all datasets using a forward pass through the model.

        This should be called after setup() and before training starts, typically in a
        Lightning callback or at the beginning of training.

        Args:
            model: The neural network model to use for forward pass.
            device: Device to run computations on (e.g., 'cpu', 'cuda').
        """
        if self.sample_weight_strategy is None:
            log.info("No sample weight strategy specified, skipping weight computation")
            return

        log.info(f"Computing sample weights for datasets on device '{device}'...")

        if self.ds_train is not None:
            log.info("Computing weights for training dataset...")
            self.ds_train.compute_weights(
                model, device=device, batch_size=self.batch_size
            )

        if self.ds_val is not None:
            log.info("Computing weights for validation dataset...")
            self.ds_val.compute_weights(
                model, device=device, batch_size=self.batch_size
            )

        if self.ds_test is not None:
            log.info("Computing weights for test dataset...")
            self.ds_test.compute_weights(
                model, device=device, batch_size=self.batch_size
            )

        log.info("Sample weight computation complete for all datasets")

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
