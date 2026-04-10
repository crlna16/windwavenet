# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Graph construction utilities for the SCHISM mesh.

Centralises adjacency building, BFS sub-graph expansion, edge-length
computation, and domain partitioning so that training datasets and
prediction scripts share the same logic.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import xarray as xr

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------


def build_adjacency(ds: xr.Dataset) -> Dict[int, List[int]]:
    """Build a 0-indexed node adjacency list from the dataset's edge array.

    Edge node IDs in the dataset are 1-indexed; they are converted to
    0-indexed positional indices into ``ds.nSCHISM_hgrid_node``.

    Args:
        ds: xarray Dataset containing ``SCHISM_hgrid_edge_nodes`` at time 0.

    Returns:
        Dict mapping node position -> list of neighbour positions (0-indexed).
    """
    raw_edges = ds.isel(time=0)["SCHISM_hgrid_edge_nodes"].values.astype(int)

    ds_node_ids = ds.nSCHISM_hgrid_node.values
    id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}

    adjacency: Dict[int, List[int]] = defaultdict(list)
    valid_count = 0
    for a_id, b_id in raw_edges:
        a_pos = id_to_pos.get(int(a_id) - 1)
        b_pos = id_to_pos.get(int(b_id) - 1)
        if a_pos is not None and b_pos is not None:
            adjacency[a_pos].append(b_pos)
            adjacency[b_pos].append(a_pos)
            valid_count += 1

    log.info(f"Built adjacency list: {valid_count} valid edges, {len(adjacency)} nodes with neighbours.")
    return adjacency


# ---------------------------------------------------------------------------
# BFS sub-graph
# ---------------------------------------------------------------------------


def build_subgraph(
    center_pos: int,
    adjacency: Dict[int, List[int]],
    node_x: np.ndarray,
    node_y: np.ndarray,
    max_radius: float,
    max_nodes: int,
    max_links: int,
) -> Tuple[np.ndarray, torch.Tensor, int]:
    """BFS sub-graph expansion around a center node.

    Args:
        center_pos: 0-indexed position of the center node.
        adjacency: Dict of node -> neighbours.
        node_x: Node x-coordinates array.
        node_y: Node y-coordinates array.
        max_radius: Maximum Euclidean distance from center (meters).
        max_nodes: Maximum number of nodes in the sub-graph.
        max_links: Maximum BFS hop depth.

    Returns:
        Tuple ``(subgraph_nodes, edge_index, center_local_idx)`` where
        *subgraph_nodes* is a sorted array of global 0-indexed node positions,
        *edge_index* is a ``(2, n_edges)`` tensor of local indices (includes
        self-loops), and *center_local_idx* is the position of the center
        within *subgraph_nodes*.
    """
    cx, cy = node_x[center_pos], node_y[center_pos]

    visited = {center_pos}
    queue: deque = deque([(center_pos, 0)])

    while queue and len(visited) < max_nodes:
        current, depth = queue.popleft()
        if depth >= max_links:
            continue
        for neighbour in adjacency.get(current, []):
            if neighbour in visited:
                continue
            dx = node_x[neighbour] - cx
            dy = node_y[neighbour] - cy
            if np.sqrt(dx * dx + dy * dy) >= max_radius:
                continue
            visited.add(neighbour)
            queue.append((neighbour, depth + 1))
            if len(visited) >= max_nodes:
                break

    subgraph_nodes = np.sort(np.array(list(visited)))
    global_to_local = {g: l for l, g in enumerate(subgraph_nodes)}  # noqa: E741
    center_local_idx = global_to_local[center_pos]

    subgraph_set = set(subgraph_nodes.tolist())
    src_list: List[int] = []
    dst_list: List[int] = []
    for u in subgraph_nodes:
        local_u = global_to_local[u]
        # Self-loop
        src_list.append(local_u)
        dst_list.append(local_u)
        # Neighbour edges
        for v in adjacency.get(u, []):
            if v in subgraph_set:
                src_list.append(local_u)
                dst_list.append(global_to_local[v])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return subgraph_nodes, edge_index, center_local_idx


# ---------------------------------------------------------------------------
# Edge weights
# ---------------------------------------------------------------------------


def compute_edge_weights(
    edge_index: torch.Tensor,
    subgraph_nodes: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    scale: float,
) -> torch.Tensor:
    """Compute inverse-distance edge weights for a sub-graph.

    Weight formula: ``w = 1 / (1 + length / scale)``.  Self-loops (length 0)
    naturally receive weight 1.0.  Shorter edges get higher weights, encoding
    the idea that closer nodes influence each other more strongly.

    Args:
        edge_index: ``(2, n_edges)`` tensor of *local* node indices.
        subgraph_nodes: Sorted array of *global* 0-indexed node positions
            that maps local index -> global position.
        node_x: Full-domain node x-coordinates.
        node_y: Full-domain node y-coordinates.
        scale: Length scale (meters) controlling the decay rate.  An edge
            whose length equals *scale* receives weight 0.5.

    Returns:
        ``(n_edges,)`` 1-D float tensor of edge weights (compatible with
        ``GCNConv.forward(..., edge_weight=...)``).
    """
    src_local = edge_index[0].numpy()
    dst_local = edge_index[1].numpy()

    src_global = subgraph_nodes[src_local]
    dst_global = subgraph_nodes[dst_local]

    dx = node_x[src_global] - node_x[dst_global]
    dy = node_y[src_global] - node_y[dst_global]
    lengths = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    weights = 1.0 / (1.0 + lengths / scale)

    return torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# Domain partitioning (for full-domain prediction)
# ---------------------------------------------------------------------------


def partition_domain(
    n_nodes: int,
    adjacency: Dict[int, List[int]],
    node_x: np.ndarray,
    node_y: np.ndarray,
    max_radius: float,
    max_nodes: int,
    max_links: int,
) -> List[Tuple[int, np.ndarray, torch.Tensor, int]]:
    """Partition all mesh nodes into a covering set of BFS sub-graphs.

    Greedy set-cover: iterate nodes in order 0..n_nodes-1, pick the first
    uncovered node as center, BFS-expand its sub-graph, mark all nodes in the
    sub-graph as covered (including any overlap at the boundaries), repeat
    until every node is covered.

    Args:
        n_nodes: Total number of mesh nodes.
        adjacency: 0-indexed adjacency list.
        node_x: Node x-coordinates.
        node_y: Node y-coordinates.
        max_radius: BFS radius constraint.
        max_nodes: BFS node count cap.
        max_links: BFS hop depth cap.

    Returns:
        List of ``(center_pos, subgraph_nodes, edge_index, center_local_idx)``
        tuples, one per partition.
    """
    covered = np.zeros(n_nodes, dtype=bool)
    partitions = []

    for node in range(n_nodes):
        if covered[node]:
            continue

        subgraph_nodes, edge_index, center_local_idx = build_subgraph(
            center_pos=node,
            adjacency=adjacency,
            node_x=node_x,
            node_y=node_y,
            max_radius=max_radius,
            max_nodes=max_nodes,
            max_links=max_links,
        )

        covered[subgraph_nodes] = True
        partitions.append((node, subgraph_nodes, edge_index, center_local_idx))

    log.info(f"Domain partitioned into {len(partitions)} sub-graphs covering {covered.sum()}/{n_nodes} nodes.")
    return partitions
