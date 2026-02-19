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
        persistent_workers: bool = False,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = True,
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
            persistent_workers: If True, workers are not shut down after each epoch (faster).
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            subsample_prob: Optional probability threshold for subsampling training data based on z_score.
                Only applied to training dataset. If provided, samples with higher z_scores
                (outliers) are more likely to be retained.
            preload_data: If True, preload training features in memory. Validation/test/predict
                data are always preloaded.
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
        self.persistent_workers = persistent_workers

        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power
        self.subsample_prob = subsample_prob
        self.preload_data = preload_data

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

        if self.num_workers == 0:
            self.prefetch_factor = None
        else:
            self.prefetch_factor = 4

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
                subsample_prob=self.subsample_prob,
                preload_data=self.preload_data,
            )

            self.ds_val = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.val_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
            )

            self.ds_test = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.test_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
            )

        elif stage == "test":
            self.ds_test = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.test_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
            )

        elif stage == "predict":
            log.info("Predict dataset is the same as test dataset.")
            self.ds_pred = SimpleWWMDataset(
                data_dir=self.data_dir,
                data_range=self.test_days,
                features=self.features,
                targets=self.targets,
                scaling_config=self.scaling_config,
                sample_weight_strategy=None,
                sample_weight_power=None,
                subsample_prob=None,
                preload_data=True,
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
        # Use worker_init_fn when not preloading data and using multiple workers
        worker_init_fn = None
        if self.num_workers > 0 and not self.preload_data:
            worker_init_fn = lambda worker_id: self.ds_train.worker_init_fn(worker_id)  # noqa: E731

        return DataLoader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.ds_val,
            batch_size=4 * self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,  # for limit val batches (?)
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        return DataLoader(
            dataset=self.ds_test,
            batch_size=4 * self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
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
