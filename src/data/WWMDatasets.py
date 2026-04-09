# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import time
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Dataset as GDataset
from torch_geometric.data import Data as GData

from src.data import dataset_utils as du
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
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = True,
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
            subsample_prob: Optional probability threshold for subsampling based on z_score.
                If provided, samples are retained with probability derived from their z_score.
                Higher z_score (outliers) have higher retention probability.
            preload_data: If True, preload X/y in memory. If False, load X/y lazily in __getitem__.
                With multiple workers, each worker slices the time axis to avoid shared Zarr chunks.
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
        self.subsample_prob = subsample_prob
        self.preload_data = preload_data

        # Store parameters for worker initialization
        self.worker_id = None
        self.num_workers = None
        self.worker_time_len = None

        # --- Pipeline: open, filter, preprocess, scale, load ---
        self.ds, self.samples_per_day = du.open_and_slice_dataset(self.data_dir, self.data_range)
        self.ds = du.filter_nodes(self.ds, self.node_idx_csv)
        self.ds, self.n_nodes = du.preprocess_variables(self.ds, self.features, self.targets)
        self.ds, self.scaling_params = du.scale_variables(self.ds, self.features, self.scaling_config)
        self.X, self.y = du.load_and_preload_data(self.ds, self.features, self.targets, self.preload_data)

        # Apply subsampling if requested (only works with preloaded data)
        if self.subsample_prob is not None:
            if self.preload_data:
                self.X, self.y = du.subsample_by_zscore(self.X, self.y, self.subsample_prob)
            else:
                log.warning("Subsampling (subsample_prob) requires preload_data=True. Skipping subsampling.")

        # Compute weights (only works with preloaded data)
        if self.sample_weight_strategy is not None and self.preload_data:
            self.weights = du.compute_weights(
                self.y, self.targets, self.sample_weight_strategy, self.sample_weight_power
            )
        else:
            self.weights = None

        num_samples = len(self.ds.time) * len(self.ds.nSCHISM_hgrid_node)
        log.info(f"Dataset initialized with {num_samples} samples. Preload data: {self.preload_data}.")

    def __len__(self) -> int:
        """Return the number of samples in the (possibly worker-sliced) dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.ds.time) * len(self.ds.nSCHISM_hgrid_node)

    def worker_init_fn(self, worker_id: int) -> None:
        """Initialize a worker by slicing the dataset across the time dimension.

        Called after the dataset is pickled to the worker process. The dataset
        (self.ds) is already preprocessed; we just split it across workers.

        This allows lazy loading (preload_data=False) to work with multiple workers
        by having each worker load only its portion of the time axis.

        Args:
            worker_id: Worker ID assigned by DataLoader
        """
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading
            return

        self.worker_id = worker_info.id
        self.num_workers = worker_info.num_workers

        # Split the already-preprocessed dataset across workers
        total_times = len(self.ds.time)
        times_per_worker = total_times // self.num_workers
        start_time = self.worker_id * times_per_worker

        if self.worker_id == self.num_workers - 1:
            # Last worker takes remaining samples
            end_time = total_times
        else:
            end_time = start_time + times_per_worker

        # Filter dataset to this worker's samples
        self.ds = self.ds.isel(time=slice(start_time, end_time))
        self.worker_time_len = end_time - start_time

        log.info(
            f"Worker {self.worker_id}/{self.num_workers}: handling time indices [{start_time}:{end_time}] "
            f"({end_time - start_time}/{total_times})"
        )

    def __getitem__(self, idx: int) -> Tuple:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, targets, weights) as PyTorch tensors if weights are computed,
            otherwise (features, targets). In lazy mode with multiple workers, the index
            is wrapped to the worker-local time slice.
        """
        if self.preload_data:
            if self.weights is not None:
                return self.X[idx], self.y[idx], self.weights[idx]
            return self.X[idx], self.y[idx]

        # Lazy loading mode: load both features and targets from dataset
        # Indexing follows the stacked order: time is the slow axis, node is the fast axis.
        local_idx = idx

        n_nodes = self.n_nodes
        if self.worker_time_len is not None:
            local_total = self.worker_time_len * n_nodes
            local_idx = local_idx % local_total

        local_t_idx = local_idx // n_nodes
        local_x_idx = local_idx % n_nodes

        sample_data = self.ds.isel(time=local_t_idx, nSCHISM_hgrid_node=local_x_idx)

        xa = sample_data[self.features].to_dataarray()
        x = torch.from_numpy(xa.values)

        ya = sample_data[self.targets].to_dataarray()
        y = torch.from_numpy(ya.values)

        # Weights are not supported in lazy loading mode
        if self.weights is not None:
            return x, y, self.weights[idx]
        return x, y


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
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = True,
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
            subsample_prob: Optional probability threshold for subsampling.
            preload_data: If True, preload data in memory for fast access.
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
            subsample_prob=subsample_prob,
            preload_data=preload_data,
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
        if self.preload_data and self.X is not None:
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

        if self.preload_data:
            # X: sequence from time_start to time_start + seq_len
            X_seq = self.X[time_start : time_start + self.seq_len, node_idx, :]

            # y: target at last step in the sequence
            y_target = self.y[time_start + self.seq_len - 1, node_idx, :]

            return X_seq, y_target

        # Lazy loading mode
        time_indices = list(range(time_start, time_start + self.seq_len))
        node = self.ds.nSCHISM_hgrid_node.values[node_idx]

        sample_data = self.ds.isel(time=time_indices).sel(nSCHISM_hgrid_node=node)

        # Features for sequence
        xa = sample_data[self.features].to_dataarray()
        X_seq = torch.from_numpy(xa.values.T)  # (seq_len, n_features)

        # Target at last step in sequence
        ya = sample_data[self.targets].to_dataarray()
        y_target = torch.from_numpy(ya.values[:, -1])  # (n_targets,)

        return X_seq, y_target


class GraphWWMDataset(GDataset):
    """Graph dataset built from the SCHISM mesh.

    Handles graph construction (adjacency, BFS subgraphs, caching) and provides
    sample access. Set ``full_graph_target=True`` for targets at all subgraph
    nodes; ``full_graph_target=False`` returns only the center node as the target.
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
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = False,
        max_radius: float = 10000.0,
        max_nodes: int = 1000,
        max_links: int = 10,
        full_graph_target: bool = True,
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
            sample_weight_strategy: Strategy for computing per-sample weights.
                Options: None (no weighting), 'z_score' (weight by z-score of targets).
            sample_weight_power: Exponent to control the intensity of sample weighting (default: 1.0).
            subsample_prob: Optional probability threshold for subsampling based on z_score.
                Must be None for graph datasets.
            preload_data: If True, preload X/y in memory. If False, load X/y lazily.
            max_radius: Maximum Euclidean distance (meters) from center node for
                including neighbors in the subgraph (default: 10000.).
            max_nodes: Maximum number of nodes in a subgraph (default: 1000).
            max_links: Maximum number of BFS hops from center node (default: 10).
            full_graph_target: If True (default), target covers all subgraph nodes.
                If False, target is center node only.
        """
        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.features = [str(var) for var in list(features)]
        self.targets = [str(var) for var in list(targets)]
        self.scaling_config = scaling_config
        self.sample_weight_strategy = sample_weight_strategy
        self.sample_weight_power = sample_weight_power
        self.preload_data = preload_data
        self.full_graph_target = full_graph_target
        self.graph_features = [str(f) for f in graph_features]
        self.max_radius = max_radius
        self.max_nodes = max_nodes
        self.max_links = max_links

        if subsample_prob is not None:
            raise ValueError(f"Subsample probability must be None for graph datasets. Found {subsample_prob}")

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
        self._build_adjacency()

        # --- Extract node coordinates (not time-varying) ---
        self.node_x = self.ds["SCHISM_hgrid_node_x"].values
        self.node_y = self.ds["SCHISM_hgrid_node_y"].values

        # --- Cache subgraph topology per center node ---
        log.info(f"Precomputing subgraph topology for {self.n_center_nodes} center nodes...")
        self.graph_cache: Dict[int, Tuple[np.ndarray, torch.Tensor, int]] = {}
        for center_pos in self.center_nodes:
            self.graph_cache[int(center_pos)] = self._build_subgraph(int(center_pos))
        log.info(f"Cached {len(self.graph_cache)} subgraph topologies.")

        # Drop graph_features from dataset to free memory
        self.ds = self.ds.drop_vars(
            [v for v in self.graph_features if v in self.ds],
            errors="ignore",
        )

        # Now that graph_features are gone, preload node features if requested
        if self.preload_data:
            self.X = du.retrieve_data(self.ds, self.features)
            self.y = du.retrieve_data(self.ds, self.targets)
        else:
            self.X = None
            self.y = None

        # Compute weights (only works with preloaded data)
        if self.sample_weight_strategy is not None and self.preload_data:
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

        target_mode = "all subgraph nodes" if self.full_graph_target else "center node only"
        log.info(
            f"GraphWWMDataset initialized: {self.n_center_nodes} center nodes, "
            f"{self._n_time} time steps, {self._total_samples} total samples, "
            f"target: {target_mode}."
        )

    def _build_adjacency(self) -> None:
        """Load edges from timestep 0 and build an adjacency list.

        Edge node IDs are 1-indexed in the dataset; they are converted to
        0-indexed positional indices into ``self.ds.nSCHISM_hgrid_node``.
        """
        start_time = time.time()

        raw_edges = self.ds.isel(time=0)["SCHISM_hgrid_edge_nodes"].values.astype(int)

        # Map 1-indexed node IDs to 0-indexed positions
        ds_node_ids = self.ds.nSCHISM_hgrid_node.values
        id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}

        self.adjacency: Dict[int, List[int]] = defaultdict(list)
        valid_count = 0
        for a_id, b_id in raw_edges:
            # Edge node IDs are 1-indexed; subtract 1 to get 0-indexed node IDs
            a_pos = id_to_pos.get(int(a_id) - 1)
            b_pos = id_to_pos.get(int(b_id) - 1)
            if a_pos is not None and b_pos is not None:
                self.adjacency[a_pos].append(b_pos)
                self.adjacency[b_pos].append(a_pos)
                valid_count += 1

        elapsed = time.time() - start_time
        log.info(
            f"Built adjacency list in {elapsed:.1f}s: "
            f"{valid_count} valid edges, {len(self.adjacency)} nodes with neighbors."
        )

    def _build_subgraph(self, center_pos: int) -> Tuple[np.ndarray, torch.Tensor, int]:
        """Build subgraph topology around a center node using BFS.

        BFS expands along mesh edges, bounded by:
        - ``max_radius``: max Euclidean distance (meters) from center node
        - ``max_nodes``: max number of nodes in the subgraph
        - ``max_links``: max number of BFS hops from the center node

        Args:
            center_pos: 0-indexed positional index of the center node.

        Returns:
            Tuple of (subgraph_nodes, edge_index, center_local_idx) where:
                - subgraph_nodes: sorted array of 0-indexed positional node indices
                - edge_index: (2, n_edges) tensor of local edge indices
                - center_local_idx: local index of the center node in subgraph_nodes
        """
        cx, cy = self.node_x[center_pos], self.node_y[center_pos]

        visited = {center_pos}
        queue = deque([(center_pos, 0)])

        while queue and len(visited) < self.max_nodes:
            current, depth = queue.popleft()

            if depth >= self.max_links:
                continue

            for neighbor in self.adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # Check distance from center node
                dx = self.node_x[neighbor] - cx
                dy = self.node_y[neighbor] - cy
                dist = np.sqrt(dx * dx + dy * dy)
                if dist >= self.max_radius:
                    continue

                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

                if len(visited) >= self.max_nodes:
                    break

        # Build local node mapping
        subgraph_nodes = np.sort(np.array(list(visited)))
        global_to_local = {g: l for l, g in enumerate(subgraph_nodes)}  # noqa: E741
        center_local_idx = global_to_local[center_pos]

        # Build edge_index: include all edges where both endpoints are in subgraph
        # plus self-loops on the diagonal (required by many PyG message-passing layers)
        subgraph_set = set(subgraph_nodes)
        src_list = []
        dst_list = []
        for u in subgraph_nodes:
            local_u = global_to_local[u]
            # Self-loop
            src_list.append(local_u)
            dst_list.append(local_u)
            # Neighbor edges
            for v in self.adjacency.get(u, []):
                if v in subgraph_set:
                    src_list.append(local_u)
                    dst_list.append(global_to_local[v])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        return subgraph_nodes, edge_index, center_local_idx

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

    def construct_graph(self, time_idx: int, center_node_pos: int) -> GData:
        """Construct a graph sample.

        Args:
            time_idx: Time index in dataset (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            torch_geometric.data.Data with attributes:
                - x: (n_subgraph_nodes, n_features) node feature tensor
                - edge_index: (2, n_edges) edge index tensor
                - y: target tensor (shape depends on full_graph_target)
                - center_idx: local index of center node in x
        """
        subgraph_nodes, edge_index, center_local_idx = self.graph_cache[center_node_pos]
        x = self._extract_node_features(time_idx, subgraph_nodes)
        target_nodes = subgraph_nodes if self.full_graph_target else center_node_pos
        y = self._extract_targets(time_idx, target_nodes)
        return GData(x=x, edge_index=edge_index, y=y, center_idx=center_local_idx)

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
    Set ``full_graph_target=True`` for targets at all subgraph nodes;
    default targets the center node only.
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
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = False,
        max_radius: float = 10000.0,
        max_nodes: int = 1000,
        max_links: int = 10,
        full_graph_target: bool = False,
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
            sample_weight_strategy: Strategy for computing per-sample weights.
            sample_weight_power: Exponent for sample weighting.
            subsample_prob: Must be None for graph datasets.
            preload_data: If True, preload X/y in memory.
            max_radius: Maximum Euclidean distance (meters) from center node.
            max_nodes: Maximum number of nodes in a subgraph.
            max_links: Maximum number of BFS hops from center node.
            full_graph_target: If True, target covers all subgraph nodes.
                If False (default), target is center node only.
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
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            subsample_prob=subsample_prob,
            preload_data=preload_data,
            max_radius=max_radius,
            max_nodes=max_nodes,
            max_links=max_links,
            full_graph_target=full_graph_target,
        )

        # Validate seq_len
        if self._n_time < self.seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be less than or equal to number of time steps ({self._n_time})"
            )

        self._n_valid_times = self._n_time - self.seq_len + 1

        self._total_samples = self.n_center_nodes * self._n_valid_times

        target_mode = "all subgraph nodes" if self.full_graph_target else "center node only"
        log.info(
            f"SequenceGraphWWMDataset initialized: {self.n_center_nodes} center nodes, "
            f"seq_len={self.seq_len}, {self._n_valid_times} valid times, "
            f"{self._total_samples} total samples, target: {target_mode}."
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

    def construct_graph(self, time_idx: int, center_node_pos: int) -> GData:
        """Construct a sequence graph sample.

        Args:
            time_idx: First time index of the feature sequence (0-indexed).
            center_node_pos: 0-indexed positional index of the center node.

        Returns:
            torch_geometric.data.Data with attributes:
                - x: (n_subgraph_nodes, seq_len, n_features) node feature tensor
                - edge_index: (2, n_edges) edge index tensor
                - y: target tensor (shape depends on full_graph_target)
                - center_idx: local index of center node in x
        """
        subgraph_nodes, edge_index, center_local_idx = self.graph_cache[center_node_pos]
        x = self._extract_sequence_node_features(time_idx, subgraph_nodes)
        target_time = time_idx + self.seq_len - 1
        target_nodes = subgraph_nodes if self.full_graph_target else center_node_pos
        y = self._extract_targets(target_time, target_nodes)
        return GData(x=x, edge_index=edge_index, y=y, center_idx=center_local_idx)
