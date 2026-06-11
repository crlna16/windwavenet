# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Dataset as GDataset
from torch_geometric.data import Data as GData

from src.data import dataset_utils as du
from src.data import graph_utils as gu
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class SimpleWWMDataset(Dataset):
    """Dataset that loads a Zarr dataset and returns features and targets per grid point and time.

    Note that preprocessing the zarr files is handled externally (CreateZarr.ipynb).
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with a column named 'nSCHISM_hgrid_node'
                containing node indices to keep. If None, keep all nodes.
            sample_weight_strategy: Strategy for computing per-sample weights.
                Options: None (no weighting), 'z_score' (weight by z-score of targets).
            sample_weight_power: Exponent to control the intensity of sample weighting (default: 1.0).
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.features = [str(var) for var in list(features)]
        self.targets = [str(var) for var in list(targets)]
        self.node_idx_csv = node_idx_csv
        self.scaling_config = scaling_config
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power

        # Store parameters for worker initialization
        self.worker_id = None
        self.num_workers = None
        self.worker_time_len = None

        # --- Pipeline: open, filter, preprocess, scale, load ---
        self.ds, self.samples_per_day = du.open_and_slice_dataset(self.data_dir, self.data_range)
        self.ds = du.filter_nodes(self.ds, self.node_idx_csv)
        self.ds, self.n_nodes = du.preprocess_variables(self.ds, self.features, self.targets)
        self.ds, self.scaling_params = du.scale_variables(self.ds, self.features, self.scaling_config)

        self.X = du.retrieve_data(self.ds, self.features)
        self.y = du.retrieve_data(self.ds, self.targets)

        # Compute weights (only works with preloaded data)
        if self.sample_weight_strategy is not None:
            self.weights = du.compute_weights(
                self.y, self.targets, self.sample_weight_strategy, self.sample_weight_power
            )
        else:
            self.weights = None

        num_samples = len(self.ds.time) * len(self.ds.nSCHISM_hgrid_node)
        log.info(f"Dataset initialized with {num_samples} samples.")

    def __len__(self) -> int:
        """Return the number of samples in the (possibly worker-sliced) dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.ds.time) * len(self.ds.nSCHISM_hgrid_node)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets, weights) as PyTorch tensors if weights are computed,
            otherwise (features, targets).
        """
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
        return self.X[idx], self.y[idx]


class SequenceWWMDataset(SimpleWWMDataset):
    """Dataset that returns time sequences for LSTM/RNN models.

    Returns sequences of length `seq_len` as input features, with the last time step in the
    sequence as target. Inherits data loading, scaling, and preprocessing from SimpleWWMDataset.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        seq_len: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
    ) -> None:
        """Initialize the sequence dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            seq_len: Length of input sequences (number of time steps).
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with node indices to keep.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent to control the intensity of sample weighting.
        """
        self.seq_len = seq_len

        # Initialize parent class (loads and preprocesses data)
        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
        )

        # Store time and node dimensions for sequence indexing
        self.n_time = len(self.ds.time)

        # Validate seq_len
        if self.n_time < self.seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be less than or equal to number of time steps ({self.n_time})"
            )

        # Number of valid sequence start positions
        self.n_valid_times = self.n_time - self.seq_len + 1

        # Reshape preloaded data for efficient sequence access: (n_time, n_nodes, n_features)
        n_features = self.X.shape[1]
        n_targets = self.y.shape[1]
        self.X = self.X.view(self.n_time, self.n_nodes, n_features)
        self.y = self.y.view(self.n_time, self.n_nodes, n_targets)
        log.info(f"Reshaped data for sequence access: X={tuple(self.X.shape)}, y={tuple(self.y.shape)}")

        log.info(
            f"SequenceWWMDataset initialized with seq_len={seq_len}, "
            f"n_valid_times={self.n_valid_times}, n_nodes={self.n_nodes}"
        )

    def __len__(self) -> int:
        """Return the number of valid sequences.

        Returns:
            Number of samples: (n_time - seq_len + 1) * n_nodes
        """
        return self.n_valid_times * self.n_nodes

    def __getitem__(self, idx: int) -> Tuple:
        """Get a sequence sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (X_seq, y_target) where:
                - X_seq: (seq_len, n_features) tensor of input features
                - y_target: (n_targets,) tensor of target at last step in sequence
        """
        # Map flat index to (time_start, node)
        time_start = idx // self.n_nodes
        node_idx = idx % self.n_nodes

        # X: sequence from time_start to time_start + seq_len
        X_seq = self.X[time_start : time_start + self.seq_len, node_idx, :]

        # y: target at last step in the sequence
        y_target = self.y[time_start + self.seq_len - 1, node_idx, :]

        return X_seq, y_target


class GraphWWMDataset(GDataset):
    """Graph dataset built from the SCHISM mesh.

    Handles graph construction (adjacency, BFS subgraphs, caching) and provides
    sample access.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        max_radius: float = 10000.0,
        max_nodes: int = 1000,
        max_links: int = 10,
        edge_weight_scale: float = 1000.0,
    ) -> None:
        """Initialize the graph dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset. Must contain graph definition.
            data_range: Day range as [start_day, end_day] inclusive (e.g., [1, 10]).
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names used for constructing the graph
                (e.g. ["SCHISM_hgrid_edge_nodes"]).
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with a column named 'nSCHISM_hgrid_node'
                containing node indices to use as center nodes. The full mesh is always
                retained for graph construction. If None, all nodes are center nodes.
            boundary_nodes_csv: Optional CSV file with a column named 'nSCHISM_hgrid_node'
                listing open-sea boundary nodes to exclude from the graph. All edges
                connecting to these nodes are removed.
            sample_weight_strategy: Strategy for computing per-sample weights.
                Options: None (no weighting), 'z_score' (weight by z-score of targets).
            sample_weight_power: Exponent to control the intensity of sample weighting (default: 1.0).
            max_radius: Maximum Euclidean distance (meters) from center node for
                including neighbors in the subgraph (default: 10000.).
            max_nodes: Maximum number of nodes in a subgraph (default: 1000).
            max_links: Maximum number of BFS hops from center node (default: 10).
            edge_weight_scale: Length scale (meters) for inverse-distance edge
                weights.  An edge whose Euclidean length equals this value
                receives weight 0.5 (default: 1000.0).
        """
        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.features = [str(var) for var in list(features)]
        self.targets = [str(var) for var in list(targets)]
        self.scaling_config = scaling_config
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power
        self.graph_features = [str(f) for f in graph_features]
        self.max_radius = max_radius
        self.max_nodes = max_nodes
        self.max_links = max_links
        self.edge_weight_scale = edge_weight_scale

        # --- Pipeline: open, preprocess (keep graph vars), scale ---
        self.ds, self.samples_per_day = du.open_and_slice_dataset(self.data_dir, self.data_range)
        # No node filtering — full mesh retained for graph construction
        self.ds, self.n_nodes = du.preprocess_variables(
            self.ds, self.features, self.targets, extra_keep_vars=self.graph_features
        )
        self.ds, self.scaling_params = du.scale_variables(self.ds, self.features, self.scaling_config)

        # --- Determine center nodes from node_idx_csv ---
        if node_idx_csv is not None:
            center_node_ids = du.load_node_indices(node_idx_csv)
            ds_node_ids = self.ds.nSCHISM_hgrid_node.values
            id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}
            self.center_nodes = np.array([id_to_pos[int(nid)] for nid in center_node_ids if int(nid) in id_to_pos])
            log.info(f"Selected {len(self.center_nodes)} center nodes from {node_idx_csv}.")
        else:
            self.center_nodes = np.arange(self.n_nodes)
        self.n_center_nodes = len(self.center_nodes)

        # --- Load edges once (time-invariant) and build adjacency ---
        self.adjacency = gu.build_adjacency(self.ds)

        # --- Remove open-sea boundary nodes if CSV is provided ---
        if boundary_nodes_csv is not None:
            boundary_node_ids = du.load_node_indices(boundary_nodes_csv)
            ds_node_ids = self.ds.nSCHISM_hgrid_node.values
            id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}
            boundary_positions = np.array([id_to_pos[int(nid)] for nid in boundary_node_ids if int(nid) in id_to_pos])
            self.adjacency = gu.remove_boundary_nodes(self.adjacency, boundary_positions)
            # Also remove boundary nodes from center nodes if present
            boundary_set = set(int(p) for p in boundary_positions)
            self.center_nodes = np.array([c for c in self.center_nodes if int(c) not in boundary_set])
            self.n_center_nodes = len(self.center_nodes)
            log.info(f"After boundary removal: {self.n_center_nodes} center nodes remain.")

        # --- Extract node coordinates (not time-varying) ---
        self.node_x = self.ds["SCHISM_hgrid_node_x"].values
        self.node_y = self.ds["SCHISM_hgrid_node_y"].values

        # --- Cache subgraph topology + edge weights per center node ---
        log.info(f"Precomputing subgraph topology for {self.n_center_nodes} center nodes...")
        self.graph_cache: Dict[int, Tuple[np.ndarray, torch.Tensor, int, torch.Tensor]] = {}
        for center_pos in self.center_nodes:
            subgraph_nodes, edge_index, center_local_idx = gu.build_subgraph(
                center_pos=int(center_pos),
                adjacency=self.adjacency,
                node_x=self.node_x,
                node_y=self.node_y,
                max_radius=self.max_radius,
                max_nodes=self.max_nodes,
                max_links=self.max_links,
            )
            edge_weight = gu.compute_edge_weights(
                edge_index,
                subgraph_nodes,
                self.node_x,
                self.node_y,
                scale=self.edge_weight_scale,
            )
            self.graph_cache[int(center_pos)] = (subgraph_nodes, edge_index, center_local_idx, edge_weight)
        log.info(
            f"Cached {len(self.graph_cache)} subgraph topologies (with edge weights, scale={self.edge_weight_scale})."
        )

        # Drop graph_features from dataset to free memory
        self.ds = self.ds.drop_vars(
            [v for v in self.graph_features if v in self.ds],
            errors="ignore",
        )

        # Now that graph_features are gone, preload node features if requested
        self.X = du.retrieve_data(self.ds, self.features)
        self.y = du.retrieve_data(self.ds, self.targets)

        # Compute weights (only works with preloaded data)
        if self.sample_weight_strategy is not None:
            self.weights = du.compute_weights(
                self.y, self.targets, self.sample_weight_strategy, self.sample_weight_power
            )
        else:
            self.weights = None

        # Precompute number of time steps as a plain int.
        # PyG's Dataset.__getattr__ intercepts attribute access, so __len__/len
        # must not traverse attribute chains like self.ds.time at call time.
        self._n_time = len(self.ds.time)

        self._total_samples = self.n_center_nodes * self._n_time

        # Initialize GDataset last, after all attributes are set,
        # because PyG's Dataset.__init__ may trigger len() or other methods.
        GDataset.__init__(self)

        log.info(
            f"GraphWWMDataset initialized: {self.n_center_nodes} center nodes, "
            f"{self._n_time} time steps, {self._total_samples} total samples. "
        )

    def _extract_node_features(self, time_idx: int, subgraph_nodes: np.ndarray) -> torch.Tensor:
        """Extract node features for a subgraph at a single time step.

        Args:
            time_idx: Time index in dataset (0-indexed).
            subgraph_nodes: Array of 0-indexed positional node indices.

        Returns:
            Tensor of shape (n_subgraph_nodes, n_features).
        """
        sample = self.ds.isel(time=time_idx, nSCHISM_hgrid_node=subgraph_nodes)
        x_da = sample[self.features].to_dataarray(dim="variable")
        return torch.from_numpy(x_da.values.T.copy()).float()

    def _extract_targets(self, time_idx: int, node_indices: Union[int, np.ndarray]) -> torch.Tensor:
        """Extract targets for given nodes at a single time step.

        Args:
            time_idx: Time index in dataset (0-indexed).
            node_indices: 0-indexed positional node index (int) or array of indices.

        Returns:
            Tensor of targets. Shape is (n_targets,) for a single node
            or (n_nodes, n_targets) for multiple nodes.
        """
        sample = self.ds.isel(time=time_idx, nSCHISM_hgrid_node=node_indices)
        y_da = sample[self.targets].to_dataarray(dim="variable")
        values = y_da.values.copy()
        # Single node: values shape is (n_targets,); multiple: (n_targets, n_nodes)
        if values.ndim == 1:
            return torch.from_numpy(values).float()
        return torch.from_numpy(values.T).float()

    def construct_graph(self, time_idx: int, center_node_pos: int) -> "WWMGraphData":
        """Construct a graph sample.

        Args:
            time_idx: Time index in dataset (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            WWMGraphData with attributes:
                - x: (n_subgraph_nodes, n_features) node feature tensor
                - edge_index: (2, n_edges) edge index tensor
                - edge_weight: (n_edges,) inverse-distance edge weights
                - y: target tensor (n_subgraph_nodes, n_targets)
                - center_idx: local index of center node in x
                - sample_weight: (n_targets,) center-node z-score weight, or None
        """
        subgraph_nodes, edge_index, center_local_idx, edge_weight = self.graph_cache[center_node_pos]
        x = self._extract_node_features(time_idx, subgraph_nodes)
        target_nodes = subgraph_nodes
        y = self._extract_targets(time_idx, target_nodes)

        sample_weight = None
        if self.weights is not None:
            center_global_idx = subgraph_nodes[center_local_idx]
            flat_idx = time_idx * self.n_nodes + int(center_global_idx)
            sample_weight = self.weights[flat_idx]  # (n_targets,)

        return WWMGraphData(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            center_idx=center_local_idx,
            sample_weight=sample_weight,
        )

    def __len__(self) -> int:
        """Return the total number of samples (center_nodes * time_steps)."""
        return self._total_samples

    def len(self) -> int:
        """Return the total number of samples (center_nodes * time_steps)."""
        return self._total_samples

    def get(self, idx: int) -> GData:
        """Get a graph sample by flat index.

        The index maps to (time_idx, center_node_pos):
            time_idx = idx // n_center_nodes
            center_pos = center_nodes[idx % n_center_nodes]

        Args:
            idx: Flat index into the dataset.

        Returns:
            GData: Graph centered on the selected node with features from the time step.
        """
        time_idx = idx // self.n_center_nodes
        center_pos = self.center_nodes[idx % self.n_center_nodes]
        return self.construct_graph(time_idx=time_idx, center_node_pos=center_pos)


class SequenceGraphWWMDataset(GraphWWMDataset):
    """Graph dataset with temporal sequences as node features.

    Each node's features span ``seq_len`` consecutive time steps instead of a
    single time step, giving the model temporal context on every graph node.
    """

    def __init__(
        self,
        data_dir: Path,
        data_range: List[int],
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        seq_len: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        boundary_nodes_csv: Optional[str] = None,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: float = 1.0,
        max_radius: float = 10000.0,
        max_nodes: int = 1000,
        max_links: int = 10,
        edge_weight_scale: float = 1000.0,
    ) -> None:
        """Initialize the sequence graph dataset.

        Args:
            data_dir: Directory of the preprocessed Zarr dataset.
            data_range: Day range as [start_day, end_day] inclusive.
            features: List of variable names to use as features.
            targets: List of variable names to use as targets.
            graph_features: List of variable names used for constructing the graph.
            seq_len: Number of consecutive time steps for node feature sequences.
            scaling_config: Path to YAML file containing mean and std for each variable.
            node_idx_csv: Optional CSV file with center node indices.
            boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            max_radius: Maximum Euclidean distance (meters) from center node.
            max_nodes: Maximum number of nodes in a subgraph.
            max_links: Maximum number of BFS hops from center node.
            edge_weight_scale: Length scale (meters) for inverse-distance edge
                weights (default: 1000.0).
        """
        self.seq_len = seq_len

        super().__init__(
            data_dir=data_dir,
            data_range=data_range,
            features=features,
            targets=targets,
            graph_features=graph_features,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            boundary_nodes_csv=boundary_nodes_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            max_radius=max_radius,
            max_nodes=max_nodes,
            max_links=max_links,
            edge_weight_scale=edge_weight_scale,
        )

        # Validate seq_len
        if self._n_time < self.seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be less than or equal to number of time steps ({self._n_time})"
            )

        self._n_valid_times = self._n_time - self.seq_len + 1

        self._total_samples = self.n_center_nodes * self._n_valid_times

        log.info(
            f"SequenceGraphWWMDataset initialized: {self.n_center_nodes} center nodes, "
            f"seq_len={self.seq_len}, {self._n_valid_times} valid times, "
            f"{self._total_samples} total samples."
        )

    def _extract_sequence_node_features(self, time_start: int, subgraph_nodes: np.ndarray) -> torch.Tensor:
        """Extract node features for a subgraph over a sequence of time steps.

        Args:
            time_start: First time index of the sequence (0-indexed).
            subgraph_nodes: Array of 0-indexed positional node indices.

        Returns:
            Tensor of shape (n_subgraph_nodes, seq_len, n_features).
        """
        time_indices = list(range(time_start, time_start + self.seq_len))
        sample = self.ds.isel(time=time_indices, nSCHISM_hgrid_node=subgraph_nodes)
        x_da = sample[self.features].to_dataarray(dim="variable")
        # x_da shape: (variable, time, nSCHISM_hgrid_node)
        # Transpose to (nSCHISM_hgrid_node, time, variable) = (n_nodes, seq_len, n_features)
        values = x_da.values.transpose(2, 1, 0).copy()
        return torch.from_numpy(values).float()

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self._total_samples

    def len(self) -> int:
        """Return the total number of samples."""
        return self._total_samples

    def get(self, idx: int) -> GData:
        """Get a sequence-graph sample by flat index.

        The index maps to (time_start, center_node_pos):
            time_start = idx // n_center_nodes
            center_pos = center_nodes[idx % n_center_nodes]

        Args:
            idx: Flat index into the dataset.

        Returns:
            GData: Graph with sequential node features.
        """
        time_idx = idx // self.n_center_nodes
        center_pos = self.center_nodes[idx % self.n_center_nodes]
        return self.construct_graph(time_idx=time_idx, center_node_pos=center_pos)

    def construct_graph(self, time_idx: int, center_node_pos: int) -> "WWMGraphData":
        """Construct a sequence graph sample.

        Args:
            time_idx: First time index of the feature sequence (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            WWMGraphData with attributes:
                - x: (n_subgraph_nodes, seq_len, n_features) node feature tensor
                - edge_index: (2, n_edges) edge index tensor
                - edge_weight: (n_edges,) inverse-distance edge weights
                - y: target tensor (n_subgraph_nodes, n_targets)
                - center_idx: local index of center node in x
        """
        subgraph_nodes, edge_index, center_local_idx, edge_weight = self.graph_cache[center_node_pos]
        x = self._extract_sequence_node_features(time_idx, subgraph_nodes)
        target_time = time_idx + self.seq_len - 1
        target_nodes = subgraph_nodes
        y = self._extract_targets(target_time, target_nodes)

        sample_weight = None
        if self.weights is not None:
            center_global_idx = subgraph_nodes[center_local_idx]
            flat_idx = target_time * self.n_nodes + int(center_global_idx)
            sample_weight = self.weights[flat_idx]  # (n_targets,)

        return WWMGraphData(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            center_idx=center_local_idx,
            sample_weight=sample_weight,
        )


class WWMGraphData(GData):
    """PyG Data subclass for WWM graph samples.

    Handles batching of graph-level attributes by stacking them along a new
    batch dimension instead of concatenating along the node dimension.  Any
    attribute listed in ``_GRAPH_LEVEL_KEYS`` gets this treatment.
    """

    _GRAPH_LEVEL_KEYS = {"x_center_seq", "sample_weight"}

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key in self._GRAPH_LEVEL_KEYS:
            return None  # stack → (batch_size, ...)
        return super().__cat_dim__(key, value, *args, **kwargs)


class CenterHistoryGraphWWMDataset(SequenceGraphWWMDataset):
    """Graph dataset where only the center node carries temporal history.

    Non-center nodes receive current-time features (2-D), while the center
    node's history is provided as a separate ``x_center_seq`` attribute.
    This is designed for the ``VirutalNodeCenterLSTMGCNNet`` backbone which processes
    center history with an LSTM and current features with a GCN.
    """

    def construct_graph(self, time_idx: int, center_node_pos: int) -> "WWMGraphData":
        """Construct a center-history graph sample.

        Args:
            time_idx: First time index of the feature sequence (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            WWMGraphData with attributes:
                - x: (n_subgraph_nodes, n_features) current-time node features
                - x_center_seq: (seq_len, n_features) center node temporal history
                - edge_index: (2, n_edges) edge index tensor
                - edge_weight: (n_edges,) inverse-distance edge weights
                - y: (n_subgraph_nodes, n_targets) targets at last time step
                - center_idx: local index of center node in x
                - sample_weight: (n_targets,) center-node z-score weight, or None
        """
        subgraph_nodes, edge_index, center_local_idx, edge_weight = self.graph_cache[center_node_pos]

        # Current-time features for ALL nodes (at the last time step of the window)
        present_time = time_idx + self.seq_len - 1
        x = self._extract_node_features(present_time, subgraph_nodes)

        # Temporal sequence for the CENTER node only
        center_global_idx = subgraph_nodes[center_local_idx]
        time_indices = list(range(time_idx, time_idx + self.seq_len))
        sample = self.ds.isel(time=time_indices, nSCHISM_hgrid_node=center_global_idx)
        x_da = sample[self.features].to_dataarray(dim="variable")
        # x_da shape: (variable, time) — single node
        x_center_seq = torch.from_numpy(x_da.values.T.copy()).float()  # (seq_len, n_features)

        # Targets at the last time step for all subgraph nodes
        y = self._extract_targets(present_time, subgraph_nodes)

        # Graph-level sample weight from center node's precomputed z-score weight
        # self.weights shape: (n_times * n_nodes, n_targets), stacked time-major
        sample_weight = None
        if self.weights is not None:
            flat_idx = present_time * self.n_nodes + int(center_global_idx)
            sample_weight = self.weights[flat_idx]  # (n_targets,)

        return WWMGraphData(
            x=x,
            x_center_seq=x_center_seq,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            center_idx=center_local_idx,
            sample_weight=sample_weight,
        )
