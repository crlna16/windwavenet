# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
from torch.utils.data import DataLoader, Dataset

import src.data.dataset_utils as du
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
        boundary_nodes_csv: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
    ) -> None:
        """Initialize the base WWM DataModule.

        Args:
            data_dir: Directory containing the Zarr dataset.
            train_days: Day range for training as [start_day, end_day] inclusive,
                or a list of such ranges (e.g. [[1, 180], [244, 304]]) for
                non-contiguous splits.
            val_days: Day range(s) for validation, same format as train_days.
            test_days: Day range(s) for testing, same format as train_days.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep for train/val.
                Test/predict use all nodes (node_idx_csv=None).
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
                These nodes are discarded in all stages.
            batch_size: Number of samples per batch.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            persistent_workers: If True, workers are not shut down after each epoch.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights
                (default: 'sigWaveHeight'). Any variable in the dataset, or 'delta_<var>'
                (derived on the fly if absent).
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
        self.boundary_nodes_csv = boundary_nodes_csv

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power
        self.sample_weight_variable = sample_weight_variable

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None
        self.ds_pred: Optional[Dataset] = None

        # (feature_idx, target_idx, mean, std) for every prev_<target> input.
        # Populated in setup(stage="fit"); consumed by AutoregressiveMixin
        # (rollout feedback) and by WWMEmulator's prev_* noise injection.
        self.feedback_map: List[Tuple[int, int, float, float]] = []

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
        sample_weight_variable: str,
    ) -> Dataset:
        """Create a dataset instance. Subclasses must implement this.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.

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
                sample_weight_variable=self.sample_weight_variable,
            )

            # Validation uses the same weighting strategy as training so that
            # val/loss (the checkpoint/early-stopping monitor) reflects the
            # weighted training objective. RMSE metrics remain unweighted.
            self.ds_val = self._create_dataset(
                data_range=self.val_days,
                node_idx_csv=self.node_idx_csv,
                sample_weight_strategy=self.sample_weight_strategy,
                sample_weight_power=self.sample_weight_power,
                sample_weight_variable=self.sample_weight_variable,
            )

            if not self.feedback_map:
                scaling_params = getattr(self.ds_train, "scaling_params", None)
                self.feedback_map = du.build_feedback_map(self.features, self.targets, scaling_params)

        elif stage == "test":
            # Test on all nodes for full evaluation
            self.ds_test = self._create_dataset(
                data_range=self.test_days,
                node_idx_csv=None,
                sample_weight_strategy=None,
                sample_weight_power=None,
                sample_weight_variable=self.sample_weight_variable,
            )

        elif stage == "predict":
            log.info("Predict dataset uses test day range with all nodes.")
            self.ds_pred = self._create_dataset(
                data_range=self.test_days,
                node_idx_csv=None,
                sample_weight_strategy=None,
                sample_weight_power=None,
                sample_weight_variable=self.sample_weight_variable,
            )

        elif stage == "validate":
            raise NotImplementedError(f"Stage '{stage}' is not implemented.")

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def teardown(self, stage: Optional[str] = None) -> None:
        """Free the dataset(s) created for *stage* once it has finished.

        Without this, ``train.py``'s fit-then-test flow (``test: true`` in
        train.yaml) keeps ``ds_train`` and ``ds_val`` fully resident in RAM
        while ``ds_test`` — built over the FULL mesh, unlike the node-subset
        train/val datasets — is also constructed, roughly tripling peak
        memory at the worst possible moment. Freeing here removes the
        DataModule's own reference; note that this may not immediately
        release memory if the Trainer itself still holds a reference to the
        corresponding DataLoader (Lightning version dependent). For a
        guaranteed-clean run on data that barely fits in RAM, prefer setting
        ``test: false`` in train.yaml and running ``eval.py`` afterward as a
        separate process instead of relying on this teardown alone.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        freed = False
        if stage == "fit":
            if self.ds_train is not None:
                self.ds_train = None
                freed = True
            if self.ds_val is not None:
                self.ds_val = None
                freed = True
        elif stage == "test":
            if self.ds_test is not None:
                self.ds_test = None
                freed = True
        elif stage == "predict":
            if self.ds_pred is not None:
                self.ds_pred = None
                freed = True

        if freed:
            gc.collect()
            log.info(f"Freed dataset(s) for stage '{stage}'.")

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
            shuffle=False,
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
