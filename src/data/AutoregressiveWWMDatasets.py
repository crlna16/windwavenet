# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Autoregressive (rollout) training datasets for the WWM emulator.

These datasets return short windows of consecutive time steps instead of a
single step, so that a LightningModule using ``AutoregressiveMixin``
(``src/model/autoregressive.py``) can roll out ``rollout_steps`` predictions
and feed each one back into the ``prev_<target>`` input slot of the next
step.

Kept separate from ``WWMDatasets.py`` to avoid cluttering the core
(single-step) dataset implementations; classes here subclass the base
datasets and reuse their data loading/scaling/preprocessing pipeline.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch

from src.data import dataset_utils as du
from src.data.WWMDatasets import SimpleWWMDataset
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)

DayRange = List[int]
DataRange = Union[DayRange, List[DayRange]]


class SimpleARWWMDataset(SimpleWWMDataset):
    """Point-wise dataset returning ``rollout_steps`` consecutive time steps.

    Each sample is a window of ``rollout_steps`` consecutive (time, node)
    pairs at a fixed node, used to train pointwise (MLP) models
    autoregressively: the model predicts step 0, its prediction is fed back
    into the ``prev_<target>`` feature for step 1, and so on.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
    ) -> None:
        """Initialize the autoregressive point-wise dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits (e.g., [[1, 10], [20, 30]]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            rollout_steps: Number of consecutive time steps per window (T).
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights
                (default: 'sigWaveHeight'). Any variable in the dataset, or 'delta_<var>'
                (derived on the fly if absent).
        """
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
        # Feeding back a delta prediction would require reconstructing the absolute
        # state at every step, coupling reconstruction error into the rollout loss.
        if any(str(t).startswith("delta_") for t in targets):
            raise ValueError(f"Autoregressive training does not support delta targets: {targets}")

        self.rollout_steps = rollout_steps

        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )

        self.n_time = len(self.ds.time)

        # Valid window start positions per contiguous day range, so that no
        # window straddles the gap between ranges. Raises if a range is
        # shorter than rollout_steps.
        self._valid_starts = du.compute_valid_sequence_starts(self.range_lengths, self.rollout_steps)
        self.n_valid_times = len(self._valid_starts)

        # Reshape preloaded data for efficient windowed access: (n_time, n_nodes, n_features)
        n_features = self.X.shape[1]
        n_targets = self.y.shape[1]
        self.X = self.X.view(self.n_time, self.n_nodes, n_features)
        self.y = self.y.view(self.n_time, self.n_nodes, n_targets)

        log.info(
            f"SimpleARWWMDataset initialized with rollout_steps={rollout_steps}, "
            f"n_valid_times={self.n_valid_times}, n_nodes={self.n_nodes}"
        )

    def __len__(self) -> int:
        """Return the number of valid rollout windows.

        Returns:
            Number of samples: n_valid_times * n_nodes
        """
        return self.n_valid_times * self.n_nodes

    def __getitem__(self, idx: int) -> Tuple:
        """Get a rollout window sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (X_win, y_win[, weight_win]) as PyTorch tensors, where:
                - X_win: (rollout_steps, n_features) input features, one row
                  per time step in the window.
                - y_win: (rollout_steps, n_targets) targets at the same steps.
                - weight_win: (rollout_steps, 1) per-step sample weight, if
                  sample weighting is enabled.
        """
        # Map flat index to (window start, node); window starts skip range gaps
        time_start = int(self._valid_starts[idx // self.n_nodes])
        node_idx = idx % self.n_nodes

        X_win = self.X[time_start : time_start + self.rollout_steps, node_idx, :]
        y_win = self.y[time_start : time_start + self.rollout_steps, node_idx, :]

        if self.weights is not None:
            # Weights are stored flat over (time, node) with time as the slow axis.
            flat_idx = (time_start + torch.arange(self.rollout_steps)) * self.n_nodes + node_idx
            weight_win = self.weights[flat_idx]
            return X_win, y_win, weight_win

        return X_win, y_win
