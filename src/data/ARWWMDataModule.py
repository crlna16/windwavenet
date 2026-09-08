# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""DataModule for autoregressive (rollout) training of pointwise WWM models."""

from typing import List, Optional, Tuple

from torch.utils.data import Dataset

from src.data import dataset_utils as du
from src.data.AutoregressiveWWMDatasets import SimpleARWWMDataset
from src.data.WWMDataModule import WWMDataModule
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class ARWWMDataModule(WWMDataModule):
    """Lightning DataModule for autoregressive point-wise WWM training.

    Train/validation samples are rollout windows of ``rollout_steps``
    consecutive time steps (``SimpleARWWMDataset``); test/predict keep the
    single-step ``SimpleWWMDataset`` inherited from ``WWMDataModule``, since
    autoregressive rollout *evaluation* is handled separately by
    ``src/rollout.py``.
    """

    def __init__(self, *args, rollout_steps: int, **kwargs) -> None:
        """Initialize the AR WWM DataModule.

        Args:
            *args: Positional args forwarded to ``WWMDataModule``.
            rollout_steps: Number of consecutive time steps (T) per training
                rollout window.
            **kwargs: Keyword args forwarded to ``WWMDataModule``.
        """
        self.rollout_steps = rollout_steps
        super().__init__(*args, **kwargs)
        # Populated in setup(stage="fit"); consumed by AutoregressiveMixin.
        self.feedback_map: List[Tuple[int, int, float, float]] = []
        self._ar_stage: Optional[str] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Build datasets for *stage*, plus the feedback map after the fit datasets exist.

        Args:
            stage: Either 'fit', 'test', or 'predict'.
        """
        self._ar_stage = stage
        super().setup(stage)
        if stage == "fit" and not self.feedback_map:
            scaling_params = getattr(self.ds_train, "scaling_params", None)
            self.feedback_map = du.build_feedback_map(self.features, self.targets, scaling_params)

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        sample_weight_variable: str = "sigWaveHeight",
    ) -> Dataset:
        """Create the AR rollout dataset for fit, the plain dataset otherwise.

        Args:
            data_range: Day range as [start_day, end_day] inclusive.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.

        Returns:
            A SimpleARWWMDataset (fit stage, i.e. train/val) or SimpleWWMDataset
            (test/predict stage).
        """
        if self._ar_stage != "fit":
            return super()._create_dataset(
                data_range, node_idx_csv, sample_weight_strategy, sample_weight_power, sample_weight_variable
            )

        return SimpleARWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            rollout_steps=self.rollout_steps,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=self.boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )
