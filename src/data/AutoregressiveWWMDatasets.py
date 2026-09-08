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
from src.data.WWMDatasets import SequenceGraphWWMDataset, SequenceWWMDataset, SimpleWWMDataset, WWMGraphData
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)

DayRange = List[int]
DataRange = Union[DayRange, List[DayRange]]


def _reject_rollout_steps_below_one(rollout_steps: int) -> None:
    """Raise if ``rollout_steps`` is not a valid rollout horizon."""
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")


def _reject_delta_targets(targets: List[str]) -> None:
    """Raise if any target is a ``delta_*`` variable.

    Feeding back a delta prediction would require reconstructing the absolute
    state at every step, coupling reconstruction error into the rollout loss.
    """
    if any(str(t).startswith("delta_") for t in targets):
        raise ValueError(f"Autoregressive training does not support delta targets: {targets}")


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
        _reject_delta_targets(targets)

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


class SequenceARWWMDataset(SequenceWWMDataset):
    """Sequence dataset returning ``rollout_steps`` consecutive LSTM windows.

    Each sample is a buffer of ``seq_len + rollout_steps - 1`` consecutive
    time steps at a fixed node. Rollout step ``s`` consumes the sub-window
    ``[s, s + seq_len)`` (an LSTM input of length ``seq_len``) and predicts
    the target at ``s + seq_len - 1``; its prediction is fed back into the
    ``prev_<target>`` feature at buffer position ``s + seq_len`` before the
    next step, exactly like ``SimpleARWWMDataset`` but with a sliding
    ``seq_len``-step window instead of a single time step per rollout
    iteration.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        seq_len: int,
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
    ) -> None:
        """Initialize the autoregressive sequence dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits (e.g., [[1, 10], [20, 30]]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            seq_len: Length of the LSTM input window consumed per rollout iteration.
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights
                (default: 'sigWaveHeight'). Any variable in the dataset, or 'delta_<var>'
                (derived on the fly if absent).
        """
        _reject_rollout_steps_below_one(rollout_steps)
        _reject_delta_targets(targets)

        self.history_len = seq_len
        self.rollout_steps = rollout_steps

        # Combined buffer length: `seq_len` history steps for the first
        # rollout iteration, plus one extra step per subsequent iteration.
        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            seq_len=seq_len + rollout_steps - 1,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
        )

        log.info(
            f"SequenceARWWMDataset initialized with seq_len={seq_len}, rollout_steps={rollout_steps}, "
            f"n_valid_times={self.n_valid_times}, n_nodes={self.n_nodes}"
        )

    def __getitem__(self, idx: int) -> Tuple:
        """Get a rollout window sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (X_buf, y_win[, weight_win]) as PyTorch tensors, where:
                - X_buf: (seq_len + rollout_steps - 1, n_features) rolling
                  input buffer; rollout step ``s`` reads ``X_buf[s : s + seq_len]``.
                - y_win: (rollout_steps, n_targets) targets, one per rollout step.
                - weight_win: (rollout_steps, 1) per-step sample weight, if
                  sample weighting is enabled.
        """
        # Map flat index to (window start, node); window starts skip range gaps
        time_start = int(self._valid_starts[idx // self.n_nodes])
        node_idx = idx % self.n_nodes

        X_buf = self.X[time_start : time_start + self.seq_len, node_idx, :]

        present_times = time_start + self.history_len - 1 + torch.arange(self.rollout_steps)
        y_win = self.y[present_times, node_idx, :]

        if self.weights is not None:
            # Weights are stored flat over (time, node) with time as the slow axis.
            flat_idx = present_times * self.n_nodes + node_idx
            weight_win = self.weights[flat_idx]
            return X_buf, y_win, weight_win

        return X_buf, y_win


class GraphARWWMDataset(SequenceGraphWWMDataset):
    """Windowed graph dataset for AR rollout training of pointwise graph models.

    For ``MultiHeadGAT`` / ``MultiHeadGATVN`` (no per-node temporal history):
    every subgraph node's features and targets are windowed over
    ``rollout_steps`` consecutive time steps, so rollout step ``s`` reads
    ``x[:, s, :]`` (all nodes' current features) and predicts ``y[:, s, :]``
    (all nodes' targets); its prediction is fed back into the ``prev_<target>``
    feature at position ``s + 1`` before the next step.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
        max_nodes: int = 1000,
        edge_weight_scale: float = 1000.0,
        cache_mode: bool = True,
    ) -> None:
        """Initialize the autoregressive pointwise graph dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names used for constructing the graph.
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with center node indices.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.
            max_nodes: Maximum number of nodes in a subgraph.
            edge_weight_scale: Length scale (meters) for inverse-distance edge weights.
            cache_mode: If True, preload target and feature tensors into shared RAM.
        """
        _reject_rollout_steps_below_one(rollout_steps)
        _reject_delta_targets(targets)

        self.rollout_steps = rollout_steps

        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            graph_features=graph_features,
            seq_len=rollout_steps,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
            max_nodes=max_nodes,
            edge_weight_scale=edge_weight_scale,
            cache_mode=cache_mode,
        )

        log.info(
            f"GraphARWWMDataset initialized: rollout_steps={rollout_steps}, "
            f"{self.n_center_nodes} center nodes, {self._n_valid_times} valid times."
        )

    def _extract_target_window(self, time_start: int, subgraph_nodes) -> torch.Tensor:
        """Extract targets for a subgraph over ``rollout_steps`` consecutive time steps.

        Args:
            time_start: First time index of the window (0-indexed).
            subgraph_nodes: Array of 0-indexed positional node indices.

        Returns:
            Tensor of shape (n_subgraph_nodes, rollout_steps, n_targets).
        """
        time_indices = list(range(time_start, time_start + self.rollout_steps))
        if hasattr(self, "_targets_ram"):
            compact = self._global_to_compact[subgraph_nodes]
            return self._targets_ram[time_indices][:, compact, :].permute(1, 0, 2).contiguous()

        sample = self.ds.isel(time=time_indices, nSCHISM_hgrid_node=subgraph_nodes)
        y_da = sample[self.targets].to_dataarray(dim="variable")
        values = y_da.values.transpose(2, 1, 0).copy()  # (node, time, variable)
        return torch.from_numpy(values).float()

    def _extract_weight_window(self, subgraph_nodes, time_indices: List[int]) -> Optional[torch.Tensor]:
        """Extract per-node, per-step sample weights, or None if unavailable.

        Args:
            subgraph_nodes: Array of 0-indexed positional node indices.
            time_indices: Time indices of the window (0-indexed).

        Returns:
            Tensor of shape (n_subgraph_nodes, len(time_indices), 1), or None.
        """
        if self.weights is None:
            return None
        compact = self._global_to_compact[subgraph_nodes]  # (n_nodes,)
        times = torch.as_tensor(time_indices, dtype=torch.long)
        flat_idx = times.unsqueeze(0) * self.n_used_nodes + torch.as_tensor(compact, dtype=torch.long).unsqueeze(1)
        return self.weights[flat_idx]  # (n_nodes, len(time_indices), 1)

    def construct_graph(self, time_idx: int, center_node_pos: int) -> "WWMGraphData":
        """Construct a windowed graph sample for rollout training.

        Args:
            time_idx: First time index of the window (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            WWMGraphData with attributes:
                - x: (n_subgraph_nodes, rollout_steps, n_features) node features
                - edge_index: (2, n_edges) edge index tensor
                - edge_weight: (n_edges,) inverse-distance edge weights
                - y: (n_subgraph_nodes, rollout_steps, n_targets) targets
                - center_idx: local index of center node in x
                - sample_weight: (n_subgraph_nodes, rollout_steps, 1), or None
        """
        subgraph_nodes, edge_index, center_local_idx, edge_weight = self.graph_cache[center_node_pos]
        x = self._extract_sequence_node_features(time_idx, subgraph_nodes)
        y = self._extract_target_window(time_idx, subgraph_nodes)
        time_indices = list(range(time_idx, time_idx + self.rollout_steps))
        sample_weight = self._extract_weight_window(subgraph_nodes, time_indices)

        return WWMGraphData(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            center_idx=center_local_idx,
            sample_weight=sample_weight,
        )


class CenterHistoryARWWMDataset(SequenceGraphWWMDataset):
    """Windowed graph dataset for AR rollout training of center-history graph models.

    For ``MultiHeadGATLSTMVN``: every subgraph node's features are windowed
    over ``history_len + rollout_steps - 1`` consecutive time steps, so any
    node's ``prev_<target>`` feature can be fed back during rollout. Rollout
    step ``s`` derives:
      - ``x`` (current-time features for ALL nodes) from buffer position
        ``s + history_len - 1``;
      - ``x_center_seq`` (temporal history for the CENTER node only) from the
        buffer sub-window ``[s, s + history_len)``.
    Its prediction (for all subgraph nodes) is fed back into the
    ``prev_<target>`` feature at buffer position ``s + history_len`` before
    the next step. Targets are windowed at the same "present" times as ``x``.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: DataRange,
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        history_len: int,
        rollout_steps: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        sample_weight_variable: str = "sigWaveHeight",
        max_nodes: int = 1000,
        edge_weight_scale: float = 1000.0,
        cache_mode: bool = True,
    ) -> None:
        """Initialize the autoregressive center-history graph dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive, or a list of
                such ranges for non-contiguous splits.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names used for constructing the graph.
            history_len: Length of the center node's LSTM history window.
            rollout_steps: Number of future steps (T) to roll out and average the loss over.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with center node indices.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            sample_weight_variable: Variable whose z-score drives sample weights.
            max_nodes: Maximum number of nodes in a subgraph.
            edge_weight_scale: Length scale (meters) for inverse-distance edge weights.
            cache_mode: If True, preload target and feature tensors into shared RAM.
        """
        _reject_rollout_steps_below_one(rollout_steps)
        _reject_delta_targets(targets)

        self.history_len = history_len
        self.rollout_steps = rollout_steps

        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            graph_features=graph_features,
            seq_len=history_len + rollout_steps - 1,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            sample_weight_variable=sample_weight_variable,
            max_nodes=max_nodes,
            edge_weight_scale=edge_weight_scale,
            cache_mode=cache_mode,
        )

        log.info(
            f"CenterHistoryARWWMDataset initialized: history_len={history_len}, "
            f"rollout_steps={rollout_steps}, {self.n_center_nodes} center nodes, "
            f"{self._n_valid_times} valid times."
        )

    def _extract_target_window(self, time_start: int, subgraph_nodes) -> torch.Tensor:
        """Extract targets at the ``rollout_steps`` "present" times (aligned with ``x``).

        Args:
            time_start: First time index of the buffer window (0-indexed).
            subgraph_nodes: Array of 0-indexed positional node indices.

        Returns:
            Tensor of shape (n_subgraph_nodes, rollout_steps, n_targets).
        """
        present_start = time_start + self.history_len - 1
        time_indices = list(range(present_start, present_start + self.rollout_steps))
        if hasattr(self, "_targets_ram"):
            compact = self._global_to_compact[subgraph_nodes]
            return self._targets_ram[time_indices][:, compact, :].permute(1, 0, 2).contiguous()

        sample = self.ds.isel(time=time_indices, nSCHISM_hgrid_node=subgraph_nodes)
        y_da = sample[self.targets].to_dataarray(dim="variable")
        values = y_da.values.transpose(2, 1, 0).copy()  # (node, time, variable)
        return torch.from_numpy(values).float()

    def _extract_weight_window(self, subgraph_nodes, time_indices: List[int]) -> Optional[torch.Tensor]:
        """Extract per-node, per-step sample weights, or None if unavailable.

        Args:
            subgraph_nodes: Array of 0-indexed positional node indices.
            time_indices: Time indices of the window (0-indexed).

        Returns:
            Tensor of shape (n_subgraph_nodes, len(time_indices), 1), or None.
        """
        if self.weights is None:
            return None
        compact = self._global_to_compact[subgraph_nodes]  # (n_nodes,)
        times = torch.as_tensor(time_indices, dtype=torch.long)
        flat_idx = times.unsqueeze(0) * self.n_used_nodes + torch.as_tensor(compact, dtype=torch.long).unsqueeze(1)
        return self.weights[flat_idx]  # (n_nodes, len(time_indices), 1)

    def construct_graph(self, time_idx: int, center_node_pos: int) -> "WWMGraphData":
        """Construct a windowed center-history graph sample for rollout training.

        Args:
            time_idx: First time index of the buffer window (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            WWMGraphData with attributes:
                - x: (n_subgraph_nodes, history_len + rollout_steps - 1,
                  n_features) ALL-node feature buffer. ``ARCenterHistoryGraphWWMEmulator``
                  derives the current-time slice and the center node's
                  ``x_center_seq`` history from this per rollout step.
                - edge_index: (2, n_edges) edge index tensor
                - edge_weight: (n_edges,) inverse-distance edge weights
                - y: (n_subgraph_nodes, rollout_steps, n_targets) targets at
                  the "present" time of each rollout step
                - center_idx: local index of center node in x
                - sample_weight: (n_subgraph_nodes, rollout_steps, 1), or None
        """
        subgraph_nodes, edge_index, center_local_idx, edge_weight = self.graph_cache[center_node_pos]
        x = self._extract_sequence_node_features(time_idx, subgraph_nodes)
        y = self._extract_target_window(time_idx, subgraph_nodes)
        present_start = time_idx + self.history_len - 1
        time_indices = list(range(present_start, present_start + self.rollout_steps))
        sample_weight = self._extract_weight_window(subgraph_nodes, time_indices)

        return WWMGraphData(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            center_idx=center_local_idx,
            sample_weight=sample_weight,
        )
