# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
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
    ) -> Dataset:
        """Create a dataset instance. Subclasses must implement this.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.

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
            )

            self.ds_val = self._create_dataset(
                data_range=self.val_days,
                node_idx_csv=self.node_idx_csv,
                sample_weight_strategy=None,
                sample_weight_power=None,
            )

        elif stage == "test":
            # Test on all nodes for full evaluation
            self.ds_test = self._create_dataset(
                data_range=self.test_days,
                node_idx_csv=None,
                sample_weight_strategy=None,
                sample_weight_power=None,
            )

        elif stage == "predict":
            log.info("Predict dataset uses test day range with all nodes.")
            self.ds_pred = self._create_dataset(
                data_range=self.test_days,
                node_idx_csv=None,
                sample_weight_strategy=None,
                sample_weight_power=None,
            )

        elif stage == "validate":
            raise NotImplementedError(f"Stage '{stage}' is not implemented.")

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def to_numpy(
        self, stage: str = "fit"
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        """Extract train / val / test data as numpy arrays.

        Calls :meth:`setup` if datasets have not been created yet.

        Args:
            stage: Stage to set up (``"fit"`` creates train, val, and test).

        Returns:
            ``(X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test)``
            where ``w_*`` is ``None`` when no sample weights are available.
        """
        if self.ds_train is None:
            self.setup(stage=stage)

        if self.ds_test is None:
            self.setup(stage="test")

        def _extract(ds):
            X = ds.X.cpu().numpy()
            y = ds.y.cpu().numpy()
            w = ds.weights.cpu().numpy() if ds.weights is not None else None
            return X, y, w

        X_train, y_train, w_train = _extract(self.ds_train)
        X_val, y_val, w_val = _extract(self.ds_val)
        X_test, y_test, w_test = _extract(self.ds_test)

        log.info(f"Extracted data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test

    def _make_dataloader(self, dataset, batch_size: int, shuffle: bool, worker_init_fn=None):
        """Create the appropriate DataLoader for the given dataset.

        Uses ``torch_geometric.loader.DataLoader`` for PyG datasets
        and ``torch.utils.data.DataLoader`` for standard torch datasets.
        """
        from torch_geometric.data import Dataset as GDataset

        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
        )

        if isinstance(dataset, GDataset):
            from torch_geometric.loader import DataLoader as GDataLoader

            kwargs["persistent_workers"] = self.persistent_workers if self.num_workers > 0 else False
            return GDataLoader(**kwargs)

        kwargs["persistent_workers"] = self.persistent_workers if self.num_workers > 0 else False
        if worker_init_fn is not None:
            kwargs["worker_init_fn"] = worker_init_fn
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""

        return self._make_dataloader(
            dataset=self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return self._make_dataloader(
            dataset=self.ds_val,
            batch_size=4 * self.batch_size,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return self._make_dataloader(
            dataset=self.ds_test,
            batch_size=4 * self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create and return the predict dataloader."""
        return self._make_dataloader(
            dataset=self.ds_pred,
            batch_size=self.batch_size,
            shuffle=False,
        )
