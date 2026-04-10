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
from src.data import graph_utils as gu
from src.utils.pylogger import RankedLogger
from src.utils.utils import print_config_tree, unscale_target_predictions

# Initialize the distributed rank so RankedLogger works without a Lightning trainer.
if getattr(rank_zero_only, "rank", None) is None:
    rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)


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
    adjacency = gu.build_adjacency(ds)

    # Drop graph topology variables after adjacency is built
    ds = ds.drop_vars([v for v in graph_features if v in ds], errors="ignore")

    # ------------------------------------------------------------------
    # 3.  Partition domain into covering sub-graphs
    # ------------------------------------------------------------------
    max_radius: float = float(cfg.data.max_radius)
    max_nodes: int = int(cfg.data.max_nodes)
    max_links: int = int(cfg.data.max_links)

    log.info(f"Partitioning domain with max_radius={max_radius}, max_nodes={max_nodes}, max_links={max_links} ...")
    partitions = gu.partition_domain(
        n_nodes=n_nodes,
        adjacency=adjacency,
        node_x=node_x,
        node_y=node_y,
        max_radius=max_radius,
        max_nodes=max_nodes,
        max_links=max_links,
    )

    # Precompute edge lengths for every partition
    edge_attrs = [
        gu.compute_edge_lengths(edge_index, subgraph_nodes, node_x, node_y)
        for _, subgraph_nodes, edge_index, _ in partitions
    ]

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
                        edge_attr=edge_attrs[batch_start + i],
                    )
                    for i, (_, subgraph_nodes, edge_index, _) in enumerate(partitions[batch_start:batch_end])
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
