# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Full-domain prediction with a GCNFullNet model.

The mesh is partitioned into non-overlapping (or minimally overlapping) sub-graphs
via greedy set cover.  For each time step, each sub-graph is processed
independently by the trained GCNFullNet and the per-node predictions are
assembled into a full-domain array.  The output format matches predict.py.

Usage example::

    python src/predict_gcn.py \
        ckpt_path=<path/to/checkpoint.ckpt> \
        data.test_days=[335,365]
"""

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from torch_geometric.data import Batch, Data as GData

from src.data import dataset_utils as du
from src.utils.pylogger import RankedLogger
from src.utils.utils import print_config_tree, unscale_target_predictions

# Initialize the distributed rank so RankedLogger works without a Lightning trainer.
if getattr(rank_zero_only, "rank", None) is None:
    rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Graph utilities (extracted from GraphWWMDataset for standalone use)
# ---------------------------------------------------------------------------


def _build_adjacency(ds: xr.Dataset) -> Dict[int, List[int]]:
    """Build a 0-indexed node adjacency list from the dataset's edge array.

    Edge node IDs are 1-indexed in the dataset; they are converted to
    0-indexed positional indices into ``ds.nSCHISM_hgrid_node``.

    Args:
        ds: xarray Dataset containing ``SCHISM_hgrid_edge_nodes`` at time 0.

    Returns:
        Dict mapping node position → list of neighbour positions (0-indexed).
    """
    raw_edges = ds.isel(time=0)["SCHISM_hgrid_edge_nodes"].values.astype(int)

    ds_node_ids = ds.nSCHISM_hgrid_node.values
    id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}

    adjacency: Dict[int, List[int]] = defaultdict(list)
    for a_id, b_id in raw_edges:
        # Edge node IDs are 1-indexed; subtract 1 to get 0-indexed node IDs
        a_pos = id_to_pos.get(int(a_id) - 1)
        b_pos = id_to_pos.get(int(b_id) - 1)
        if a_pos is not None and b_pos is not None:
            adjacency[a_pos].append(b_pos)
            adjacency[b_pos].append(a_pos)

    log.info(f"Built adjacency list: {len(adjacency)} nodes with neighbours.")
    return adjacency


def _build_subgraph(
    center_pos: int,
    adjacency: Dict[int, List[int]],
    node_x: np.ndarray,
    node_y: np.ndarray,
    max_radius: float,
    max_nodes: int,
    max_links: int,
) -> Tuple[np.ndarray, torch.Tensor, int]:
    """BFS sub-graph expansion around a center node.

    Mirrors ``GraphWWMDataset._build_subgraph`` exactly.

    Args:
        center_pos: 0-indexed position of the center node.
        adjacency: Dict of node → neighbours.
        node_x: Node x-coordinates array.
        node_y: Node y-coordinates array.
        max_radius: Maximum Euclidean distance from center (meters).
        max_nodes: Maximum number of nodes in the sub-graph.
        max_links: Maximum BFS hop depth.

    Returns:
        Tuple ``(subgraph_nodes, edge_index, center_local_idx)`` where
        *subgraph_nodes* is a sorted array of global 0-indexed node positions,
        *edge_index* is a ``(2, n_edges)`` tensor of local indices, and
        *center_local_idx* is the position of the center within *subgraph_nodes*.
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
        src_list.append(local_u)
        dst_list.append(local_u)
        for v in adjacency.get(u, []):
            if v in subgraph_set:
                src_list.append(local_u)
                dst_list.append(global_to_local[v])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return subgraph_nodes, edge_index, center_local_idx


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

    A node may appear in more than one sub-graph's topology (so message-passing
    has full context at boundaries), but for prediction assembly only the first
    sub-graph that covers a node is authoritative ("first seen wins").

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

        subgraph_nodes, edge_index, center_local_idx = _build_subgraph(
            center_pos=node,
            adjacency=adjacency,
            node_x=node_x,
            node_y=node_y,
            max_radius=max_radius,
            max_nodes=max_nodes,
            max_links=max_links,
        )

        # Mark all nodes in this sub-graph as covered (first-seen wins)
        covered[subgraph_nodes] = True
        partitions.append((node, subgraph_nodes, edge_index, center_local_idx))

    log.info(f"Domain partitioned into {len(partitions)} sub-graphs covering {covered.sum()}/{n_nodes} nodes.")
    return partitions


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_gcn(cfg: DictConfig) -> xr.Dataset:
    """Predict over the full mesh domain using a trained GCNFullNet model.

    Args:
        cfg: Hydra configuration.

    Returns:
        xr.Dataset with predictions for all nodes and time steps.
    """
    assert cfg.ckpt_path, "ckpt_path must be provided"

    log.info("Printing config tree")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    # ------------------------------------------------------------------
    # 1.  Open and preprocess dataset
    # ------------------------------------------------------------------
    data_dir = Path(cfg.data.data_dir)
    features: List[str] = list(cfg.data.features)
    targets: List[str] = list(cfg.data.targets)
    graph_features: List[str] = list(cfg.data.graph_features)
    test_days: List[int] = list(cfg.data.test_days)

    log.info(f"Opening dataset from {data_dir}")
    ds, _ = du.open_and_slice_dataset(data_dir, test_days)

    ds, n_nodes = du.preprocess_variables(ds, features, targets, extra_keep_vars=graph_features)
    log.info(f"Dataset: {n_nodes} nodes, {ds.sizes['time']} time steps")

    ds, scaling_params = du.scale_variables(ds, features, cfg.data.get("scaling_config", None))

    # ------------------------------------------------------------------
    # 2.  Build mesh topology
    # ------------------------------------------------------------------
    node_x: np.ndarray = ds["SCHISM_hgrid_node_x"].values
    node_y: np.ndarray = ds["SCHISM_hgrid_node_y"].values

    log.info("Building adjacency list ...")
    adjacency = _build_adjacency(ds)

    # Drop graph topology variables after adjacency is built
    ds = ds.drop_vars([v for v in graph_features if v in ds], errors="ignore")

    # ------------------------------------------------------------------
    # 3.  Partition domain into covering sub-graphs
    # ------------------------------------------------------------------
    max_radius: float = float(cfg.data.max_radius)
    max_nodes: int = int(cfg.data.max_nodes)
    max_links: int = int(cfg.data.max_links)

    log.info(f"Partitioning domain with max_radius={max_radius}, max_nodes={max_nodes}, max_links={max_links} ...")
    partitions = partition_domain(
        n_nodes=n_nodes,
        adjacency=adjacency,
        node_x=node_x,
        node_y=node_y,
        max_radius=max_radius,
        max_nodes=max_nodes,
        max_links=max_links,
    )

    # ------------------------------------------------------------------
    # 4.  Load model from checkpoint
    # ------------------------------------------------------------------
    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model_cls_path: str = cfg.model._target_
    module_path, class_name = model_cls_path.rsplit(".", 1)

    import importlib

    model_module = importlib.import_module(module_path)
    model_cls = getattr(model_module, class_name)

    backbone = hydra.utils.instantiate(cfg.model.backbone)
    model = model_cls.load_from_checkpoint(
        cfg.ckpt_path,
        backbone=backbone,
        strict=False,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(f"Model loaded on {device}")

    # Verify target names match checkpoint
    config_targets = list(cfg.data.targets)
    model_targets = model.target_names
    if config_targets != model_targets:
        raise ValueError(
            f"Target mismatch: config specifies {config_targets} but model was trained on {model_targets}."
        )
    n_targets = len(targets)

    # ------------------------------------------------------------------
    # 5.  Prepare for fast inference
    # ------------------------------------------------------------------

    # 5a. Precompute which global nodes each partition "owns" (first-seen wins).
    #     Done once here; avoids rebuilding a `filled` mask on every time step.
    log.info("Precomputing partition ownership ...")
    assigned = np.zeros(n_nodes, dtype=bool)
    owned_map: List[Tuple[np.ndarray, np.ndarray]] = []  # (owned_local, owned_global)
    for _, subgraph_nodes, _, _ in partitions:
        mask = ~assigned[subgraph_nodes]
        owned_local = np.where(mask)[0]
        owned_global = subgraph_nodes[owned_local]
        assigned[owned_global] = True
        owned_map.append((owned_local, owned_global))
    log.info(f"Ownership assigned: {assigned.sum()}/{n_nodes} nodes covered.")

    # 5b. Preload all node features into a single float32 array (n_time, n_nodes, n_features).
    #     Avoids tens-of-thousands of expensive xarray isel() calls inside the loop.
    log.info("Preloading feature array into memory ...")
    X_all = np.stack([ds[f].values for f in features], axis=-1).astype(np.float32)
    log.info(f"Feature array shape: {X_all.shape}")

    # 5c. Number of sub-graphs packed into a single PyG Batch per forward pass.
    #     Larger values give better GPU utilisation at the cost of more host memory.
    batch_size_graphs: int = int(cfg.get("batch_size_graphs", 32))

    # ------------------------------------------------------------------
    # 6.  Run inference: iterate time steps, process partitions in batches
    # ------------------------------------------------------------------
    n_time = ds.sizes["time"]
    all_predictions = np.empty((n_time, n_nodes, n_targets), dtype=np.float32)

    log.info(
        f"Running inference over {n_time} time steps, "
        f"{len(partitions)} partitions, batch_size_graphs={batch_size_graphs} ..."
    )

    with torch.no_grad():
        for time_idx in range(n_time):
            X_t = X_all[time_idx]  # (n_nodes, n_features) — view, no copy

            for batch_start in range(0, len(partitions), batch_size_graphs):
                batch_end = min(batch_start + batch_size_graphs, len(partitions))

                # Build a list of GData objects (on CPU) and collate into a single Batch
                data_list = [
                    GData(
                        x=torch.from_numpy(X_t[subgraph_nodes].copy()),
                        edge_index=edge_index,
                    )
                    for _, subgraph_nodes, edge_index, _ in partitions[batch_start:batch_end]
                ]
                batch = Batch.from_data_list(data_list).to(device)

                preds = model.forward(batch)  # (total_nodes_in_batch, n_targets)
                preds_np = preds.cpu().numpy()

                # Scatter predictions into the output buffer via precomputed ownership
                ptr = batch.ptr.cpu().numpy()
                for j in range(batch_end - batch_start):
                    part_idx = batch_start + j
                    owned_local, owned_global = owned_map[part_idx]
                    seg = preds_np[ptr[j] : ptr[j + 1]]  # (n_sg_nodes, n_targets)
                    all_predictions[time_idx, owned_global, :] = seg[owned_local]

            if (time_idx + 1) % max(1, n_time // 10) == 0:
                log.info(f"  Processed {time_idx + 1}/{n_time} time steps")

    log.info("Inference complete.")

    # ------------------------------------------------------------------
    # 6.  Post-process: unscale / delta-reconstruct
    # ------------------------------------------------------------------
    data_vars: Dict[str, Any] = {}

    for target_idx, target_name in enumerate(targets):
        target_preds = torch.from_numpy(all_predictions[:, :, target_idx])  # (n_time, n_nodes)

        if target_name.startswith("delta_"):
            original_var = target_name.replace("delta_", "")
            log.info(f"Reconstructing {original_var} from delta predictions")
            delta_np = target_preds.numpy()
            reconstructed, ds = du.reconstruct_from_delta(
                delta_predictions=delta_np,
                dataset=ds,
                target_name=target_name,
                features=features,
                scaling_params=scaling_params,
            )
            data_vars[original_var] = (["time", "nSCHISM_hgrid_node"], reconstructed)
            data_vars[target_name] = (["time", "nSCHISM_hgrid_node"], delta_np)
        else:
            target_preds = unscale_target_predictions(
                target_preds,
                target_name,
                features,
                scaling_params,
            )
            data_vars[target_name] = (
                ["time", "nSCHISM_hgrid_node"],
                target_preds.numpy(),
            )

    # ------------------------------------------------------------------
    # 7.  Build output xarray Dataset and save to NetCDF
    # ------------------------------------------------------------------
    output_target_names = [tn.replace("delta_", "") for tn in targets]

    ds_pred = xr.Dataset(
        data_vars,
        coords={
            "time": ds.time,
            "nSCHISM_hgrid_node": ds.nSCHISM_hgrid_node,
            "nSCHISM_hgrid_node_x": ds.SCHISM_hgrid_node_x,
            "nSCHISM_hgrid_node_y": ds.SCHISM_hgrid_node_y,
        },
        attrs={
            "description": f"GCN full-domain predictions for {', '.join(output_target_names)}",
            "checkpoint": cfg.ckpt_path,
            "model": cfg.model._target_,
            "n_partitions": len(partitions),
            "max_radius": max_radius,
            "max_nodes": max_nodes,
            "max_links": max_links,
        },
    )

    output_path = f"{cfg.paths.output_dir}/predictions.nc"
    log.info(f"Saving predictions to {output_path}")
    ds_pred.to_netcdf(output_path)
    log.info(f"Predictions saved successfully to {output_path}")

    return ds_pred


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict_gcn.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for GCN full-domain prediction."""
    predict_gcn(cfg)


if __name__ == "__main__":
    main()
