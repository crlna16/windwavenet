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
# Boundary node removal
# ---------------------------------------------------------------------------


def remove_boundary_nodes(
    adjacency: Dict[int, List[int]],
    boundary_positions: np.ndarray,
) -> Dict[int, List[int]]:
    """Remove boundary nodes from the adjacency list and prune dangling edges.

    All edges that connect to a boundary node are removed.  The adjacency
    is modified in-place and also returned.

    Args:
        adjacency: Dict mapping node position -> list of neighbour positions.
        boundary_positions: Array of 0-indexed node positions to remove.

    Returns:
        The pruned adjacency dict (same object, modified in-place).
    """
    boundary_set = set(int(p) for p in boundary_positions)

    n_nodes_removed = 0
    n_edges_removed = 0

    for node_pos in boundary_set:
        if node_pos in adjacency:
            n_edges_removed += len(adjacency[node_pos])
            del adjacency[node_pos]
            n_nodes_removed += 1

    # Remove references to boundary nodes from remaining neighbor lists
    for node_pos in list(adjacency.keys()):
        original_len = len(adjacency[node_pos])
        adjacency[node_pos] = [nb for nb in adjacency[node_pos] if nb not in boundary_set]
        n_edges_removed += original_len - len(adjacency[node_pos])

    log.info(
        f"Removed {n_nodes_removed} boundary nodes and {n_edges_removed} edges "
        f"from adjacency ({len(boundary_set)} boundary positions requested)."
    )
    return adjacency


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
# Edge index
# ---------------------------------------------------------------------------


def build_edge_index(
    subgraph_nodes: np.ndarray,
    adjacency: Dict[int, List[int]],
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Build a local ``edge_index`` (with self-loops) for a fixed node set.

    Args:
        subgraph_nodes: Sorted array of global 0-indexed node positions.
        adjacency: Dict of node -> neighbours (0-indexed).

    Returns:
        Tuple ``(edge_index, global_to_local)`` where *edge_index* is a
        ``(2, n_edges)`` tensor of local indices including self-loops, and
        *global_to_local* maps global position -> local index.
    """
    global_to_local = {int(g): l for l, g in enumerate(subgraph_nodes)}  # noqa: E741
    subgraph_set = set(int(n) for n in subgraph_nodes)

    src_list: List[int] = []
    dst_list: List[int] = []
    for u in subgraph_nodes:
        u = int(u)
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
    return edge_index, global_to_local


# ---------------------------------------------------------------------------
# BFS sub-graph
# ---------------------------------------------------------------------------


def build_subgraph(
    center_pos: int,
    adjacency: Dict[int, List[int]],
    max_nodes: int,
) -> Tuple[np.ndarray, torch.Tensor, int]:
    """BFS sub-graph expansion around a center node.

    Nodes are collected in breadth-first order from the center until
    ``max_nodes`` is reached, so the sub-graph is the center node's
    ``max_nodes``-nearest mesh neighbourhood (in hop distance).

    Args:
        center_pos: 0-indexed position of the center node.
        adjacency: Dict of node -> neighbours.
        max_nodes: Maximum number of nodes in the sub-graph.

    Returns:
        Tuple ``(subgraph_nodes, edge_index, center_local_idx)`` where
        *subgraph_nodes* is a sorted array of global 0-indexed node positions,
        *edge_index* is a ``(2, n_edges)`` tensor of local indices (includes
        self-loops), and *center_local_idx* is the position of the center
        within *subgraph_nodes*.
    """
    visited = {center_pos}
    queue: deque = deque([center_pos])

    while queue and len(visited) < max_nodes:
        current = queue.popleft()
        for neighbour in adjacency.get(current, []):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            queue.append(neighbour)
            if len(visited) >= max_nodes:
                break

    subgraph_nodes = np.sort(np.array(list(visited)))
    edge_index, global_to_local = build_edge_index(subgraph_nodes, adjacency)
    center_local_idx = int(global_to_local[center_pos])

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

    The weights feed the scalar edge encoding of the GAT backbones, so
    *scale* should be comparable to the typical (e.g. median) mesh edge
    length: that spreads the weights over (0, 1) so edge-length differences
    remain resolvable.  If *scale* is much larger than the mesh edges, all
    weights cluster near 1.0 and the encoding degenerates to a near-constant
    signal.

    Args:
        edge_index: ``(2, n_edges)`` tensor of *local* node indices.
        subgraph_nodes: Sorted array of *global* 0-indexed node positions
            that maps local index -> global position.
        node_x: Full-domain node x-coordinates.
        node_y: Full-domain node y-coordinates.
        scale: Length scale (meters) controlling the decay rate.  An edge
            whose length equals *scale* receives weight 0.5.  Choose a value
            comparable to the median mesh edge length (see above).

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


def _subgraph_center(
    subgraph_nodes: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> int:
    """Return the patch node closest to the midpoint of its x/y bounding box.

    Picks the node nearest to the centre of the axis-aligned bounding box that
    encloses the patch, i.e. a geometrically central node rather than the
    region-growing seed (which sits on the patch edge).  Suitable because the
    center node is connected to every other node in the sub-graph.

    Args:
        subgraph_nodes: Array of global 0-indexed node positions in the patch.
        node_x: Full-domain node x-coordinates.
        node_y: Full-domain node y-coordinates.

    Returns:
        Global 0-indexed position of the chosen center node.
    """
    xs = node_x[subgraph_nodes]
    ys = node_y[subgraph_nodes]
    mid_x = 0.5 * (xs.min() + xs.max())
    mid_y = 0.5 * (ys.min() + ys.max())
    dist_sq = (xs - mid_x) ** 2 + (ys - mid_y) ** 2
    return int(subgraph_nodes[int(np.argmin(dist_sq))])


def partition_domain_disjoint(
    n_nodes: int,
    adjacency: Dict[int, List[int]],
    node_x: np.ndarray,
    node_y: np.ndarray,
    target_nodes: int,
    tolerance: float = 0.1,
) -> List[Tuple[int, np.ndarray, torch.Tensor, int]]:
    """Partition all mesh nodes into *disjoint* BFS sub-graphs.

    Every node belongs to exactly one sub-graph (no overlap), so full-domain
    metrics are not biased by nodes being overwritten with predictions from a
    far-away center.  This comes at the cost of a fuzzy sub-graph size: each
    patch targets ``target_nodes`` nodes but is allowed to vary within
    ``+/- tolerance`` because the mesh cannot be tiled exactly into equal
    fixed-count patches.

    Greedy region growing: iterate nodes 0..n_nodes-1, pick the first
    uncovered node as a seed, BFS-expand *only into still-uncovered nodes*
    until the upper size bound is reached or the local uncovered region is
    exhausted, then mark the patch covered.  Undersized leftover patches
    (below the lower bound) are merged into an adjacent patch; a merge may
    push that patch above the upper bound, which is accepted to preserve
    exact, single coverage.  The reported center of each patch is *not* the
    seed (which lies on the patch edge) but the node closest to the midpoint of
    the patch's x/y bounding box, so it is geometrically central.

    Args:
        n_nodes: Total number of mesh nodes.
        adjacency: 0-indexed adjacency list.
        node_x: Full-domain node x-coordinates (for geometric center selection).
        node_y: Full-domain node y-coordinates (for geometric center selection).
        target_nodes: Desired node count per sub-graph.
        tolerance: Fractional size slack; bounds are
            ``[target_nodes * (1 - tolerance), target_nodes * (1 + tolerance)]``.

    Returns:
        List of ``(center_pos, subgraph_nodes, edge_index, center_local_idx)``
        tuples, one per partition.  The partitions' ``subgraph_nodes`` are
        mutually disjoint and their union is all ``n_nodes`` nodes.
    """
    min_nodes = max(1, int(round(target_nodes * (1.0 - tolerance))))
    max_nodes = int(round(target_nodes * (1.0 + tolerance)))

    covered = np.zeros(n_nodes, dtype=bool)
    # Each entry: {"nodes": set[int] | None}; None marks a merged-away patch.
    # The center node is chosen later (post-merge) as the graph-central node.
    patches: List[Dict] = []

    for seed in range(n_nodes):
        if covered[seed]:
            continue

        visited = {seed}
        covered[seed] = True
        queue: deque = deque([seed])
        while queue and len(visited) < max_nodes:
            current = queue.popleft()
            for neighbour in adjacency.get(current, []):
                if covered[neighbour] or neighbour in visited:
                    continue
                visited.add(neighbour)
                covered[neighbour] = True
                queue.append(neighbour)
                if len(visited) >= max_nodes:
                    break

        patches.append({"nodes": visited})

    # Map each node to its patch so undersized patches can find a neighbour.
    node_to_patch: Dict[int, int] = {}
    for pi, patch in enumerate(patches):
        for nd in patch["nodes"]:
            node_to_patch[nd] = pi

    # Merge undersized patches into an adjacent surviving patch.
    n_merged = 0
    for pi, patch in enumerate(patches):
        if patch["nodes"] is None or len(patch["nodes"]) >= min_nodes:
            continue

        neighbour_pi = None
        for nd in patch["nodes"]:
            for nb in adjacency.get(nd, []):
                other = node_to_patch.get(nb)
                if other is not None and other != pi and patches[other]["nodes"] is not None:
                    neighbour_pi = other
                    break
            if neighbour_pi is not None:
                break

        if neighbour_pi is None:
            continue  # isolated component: keep it as its own (small) patch

        patches[neighbour_pi]["nodes"] |= patch["nodes"]
        for nd in patch["nodes"]:
            node_to_patch[nd] = neighbour_pi
        patch["nodes"] = None
        n_merged += 1

    partitions: List[Tuple[int, np.ndarray, torch.Tensor, int]] = []
    for patch in patches:
        if patch["nodes"] is None:
            continue
        subgraph_nodes = np.sort(np.fromiter(patch["nodes"], dtype=int))
        # Relabel the center to the geometrically central node (closest to the
        # patch's x/y bounding-box midpoint), matching real BFS-ball inference.
        center = _subgraph_center(subgraph_nodes, node_x, node_y)
        edge_index, global_to_local = build_edge_index(subgraph_nodes, adjacency)
        center_local_idx = int(global_to_local[center])
        partitions.append((center, subgraph_nodes, edge_index, center_local_idx))

    total = sum(len(sn) for _, sn, _, _ in partitions)
    log.info(
        f"Disjoint partitioning: {len(partitions)} sub-graphs, {n_merged} undersized patches merged, "
        f"covering {total}/{n_nodes} nodes (target {target_nodes} +/- {tolerance:.0%})."
    )
    return partitions
