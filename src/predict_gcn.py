# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Full-domain prediction with graph-based models (GCN, GCN-EPD, GCN-LSTM).

The mesh is partitioned into non-overlapping (or minimally overlapping) sub-graphs
via greedy set cover.  For each time step, each sub-graph is processed
independently by the trained model and the per-node predictions are
assembled into a full-domain array.  The output format matches predict.py.

Supports both single-step backbones and sequence backbones that require ``seq_len`` consecutive
time steps as input.

Usage example::

    python src/predict_gcn.py --config-name=predict_gcn \
        ckpt_path=<path/to/checkpoint.ckpt> \
        data.test_days=[335,365]
"""

from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from torch_geometric.data import Batch, Data as GData

from src.data import dataset_utils as du
from src.data import graph_utils as gu
from src.data.WWMDatasets import CenterHistoryData
from src.utils.pylogger import RankedLogger
from src.utils.instantiators import instantiate_loggers
from src.utils.utils import get_bin_mask, print_config_tree

# Initialize the distributed rank so RankedLogger works without a Lightning trainer.
if getattr(rank_zero_only, "rank", None) is None:
    rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------


def predict_gcn(cfg: DictConfig) -> xr.Dataset:
    """Predict over the full mesh domain using a trained graph model.

    Supports graph model backbones with and without history on the center node.

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

    # Remove open-sea boundary nodes if configured
    boundary_nodes_csv = cfg.data.get("boundary_nodes_csv", None)
    boundary_positions: np.ndarray | None = None
    if boundary_nodes_csv is not None:
        boundary_node_ids = du.load_node_indices(boundary_nodes_csv)
        ds_node_ids = ds.nSCHISM_hgrid_node.values
        id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}
        boundary_positions = np.array([id_to_pos[int(nid)] for nid in boundary_node_ids if int(nid) in id_to_pos])
        adjacency = gu.remove_boundary_nodes(adjacency, boundary_positions)

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

    # Precompute edge weights for every partition
    edge_weight_scale: float = float(cfg.data.edge_weight_scale)
    log.info(f"Computing edge weights with scale={edge_weight_scale} ...")
    edge_weights = [
        gu.compute_edge_weights(edge_index, subgraph_nodes, node_x, node_y, scale=edge_weight_scale)
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

    # 5c. Sequence length for temporal models
    seq_len: int = int(cfg.data.get("seq_len", 1))
    log.info(f"Sequence length: {seq_len}")

    # 5d. Number of sub-graphs packed into a single PyG Batch per forward pass.
    #     Larger values give better GPU utilisation at the cost of more host memory.
    batch_size_graphs: int = int(cfg.get("batch_size_graphs", 32))

    # 5e. Center-history mode: only the center node of each partition receives
    #     temporal history; all other nodes get current-time features only.
    center_history_only: bool = bool(cfg.data.get("center_history_only", False))
    if center_history_only:
        log.info("Center-history mode enabled: only center nodes receive temporal sequences.")

    # ------------------------------------------------------------------
    # 6.  Run inference: iterate time steps, process partitions in batches
    # ------------------------------------------------------------------
    n_time = ds.sizes["time"]
    all_predictions = np.full((n_time, n_nodes, n_targets), np.nan, dtype=np.float32)

    log.info(
        f"Running inference over {n_time} time steps, "
        f"{len(partitions)} partitions, batch_size_graphs={batch_size_graphs} ..."
    )

    with torch.no_grad():
        for time_idx in range(n_time):
            # For sequence models, gather seq_len consecutive frames;
            # for single-step models this is just X_all[time_idx].
            if seq_len > 1:
                t_start = max(0, time_idx - seq_len + 1)
                X_window = X_all[t_start : time_idx + 1]  # (<=seq_len, n_nodes, n_feat)
                # Pad with the earliest available frame if near the start
                if X_window.shape[0] < seq_len:
                    pad = np.repeat(X_window[:1], seq_len - X_window.shape[0], axis=0)
                    X_window = np.concatenate([pad, X_window], axis=0)
                # X_window: (seq_len, n_nodes, n_features)
            else:
                X_window = None

            X_t = X_all[time_idx]  # (n_nodes, n_features)

            for batch_start in range(0, len(partitions), batch_size_graphs):
                batch_end = min(batch_start + batch_size_graphs, len(partitions))

                # Build a list of GData objects (on CPU) and collate into a single Batch
                data_list = []
                for i, (_, subgraph_nodes, edge_index, center_local_idx) in enumerate(
                    partitions[batch_start:batch_end]
                ):
                    if center_history_only and seq_len > 1:
                        # Current-time features for all nodes (2-D)
                        x = torch.from_numpy(X_t[subgraph_nodes].copy())
                        # Temporal sequence for center node only
                        center_global = subgraph_nodes[center_local_idx]
                        x_center_seq = torch.from_numpy(X_window[:, center_global, :].copy())  # (seq_len, n_features)
                        data_list.append(
                            CenterHistoryData(
                                x=x,
                                x_center_seq=x_center_seq,
                                edge_index=edge_index,
                                edge_weight=edge_weights[batch_start + i],
                                center_idx=center_local_idx,
                            )
                        )
                    elif seq_len > 1:
                        # (seq_len, n_sg_nodes, n_feat) -> (n_sg_nodes, seq_len, n_feat)
                        x = torch.from_numpy(X_window[:, subgraph_nodes, :].transpose(1, 0, 2).copy())
                        data_list.append(
                            GData(
                                x=x,
                                edge_index=edge_index,
                                edge_weight=edge_weights[batch_start + i],
                            )
                        )
                    else:
                        x = torch.from_numpy(X_t[subgraph_nodes].copy())
                        data_list.append(
                            GData(
                                x=x,
                                edge_index=edge_index,
                                edge_weight=edge_weights[batch_start + i],
                            )
                        )
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
    # 6.  Compute test metrics (optional, numpy-only — no extra GPU pass)
    # ------------------------------------------------------------------
    if cfg.get("test_metrics", False):
        log.info("Computing test metrics from prediction buffer ...")
        Y_all = np.stack([ds[t].values for t in targets], axis=-1).astype(np.float32)
        # Y_all shape: (n_time, n_nodes, n_targets)  — same space as all_predictions

        # Exclude boundary nodes (NaN positions) from metrics
        valid_mask = ~np.isnan(all_predictions[:, :, 0])
        n_excluded = n_nodes - valid_mask[0].sum() if boundary_positions is not None else 0
        log.info(f"Metrics exclude {n_excluded} boundary nodes ({valid_mask.sum()} valid samples).")

        metrics: Dict[str, float] = {}
        for target_idx, target_name in enumerate(targets):
            pred = all_predictions[:, :, target_idx][valid_mask]
            truth = Y_all[:, :, target_idx][valid_mask]
            rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
            mae = float(np.mean(np.abs(pred - truth)))
            bias = float(np.mean(pred - truth))
            metrics[f"test/rmse_{target_name}"] = rmse
            metrics[f"test/mae_{target_name}"] = mae
            metrics[f"test/bias_{target_name}"] = bias
            log.info(f"  {target_name}:  RMSE={rmse:.6f}  MAE={mae:.6f}  bias={bias:.6f}")

        # Per-bin RMSE (if target_bins are configured)
        target_bins = cfg.model.get("target_bins") or {}
        for target_idx, target_name in enumerate(targets):
            if target_name not in target_bins:
                continue
            bins = list(target_bins[target_name])
            pred = all_predictions[:, :, target_idx][valid_mask]
            truth = Y_all[:, :, target_idx][valid_mask]
            for bin_idx in range(len(bins) + 1):
                mask = get_bin_mask(truth, bins, bin_idx)
                if mask.sum() == 0:
                    continue
                if bin_idx == 0:
                    bin_label = f"<{bins[0]}"
                elif bin_idx == len(bins):
                    bin_label = f">={bins[-1]}"
                else:
                    bin_label = f"[{bins[bin_idx - 1]},{bins[bin_idx]})"
                rmse_bin = float(np.sqrt(np.mean((pred[mask] - truth[mask]) ** 2)))
                metrics[f"test/rmse_{target_name}_{bin_label}"] = rmse_bin
                log.info(f"  {target_name} {bin_label}:  RMSE={rmse_bin:.6f}  (n={int(mask.sum())})")

        # Log to wandb if a logger is configured
        loggers = instantiate_loggers(cfg.get("logger"))
        for lgr in loggers:
            if hasattr(lgr, "log_metrics"):
                lgr.log_metrics(metrics, step=0)
            if hasattr(lgr, "finalize"):
                lgr.finalize("success")
        log.info("Test metrics logged.")

    # ------------------------------------------------------------------
    # 7.  Post-process and build output xarray Dataset
    # ------------------------------------------------------------------
    ds_pred = du.build_prediction_dataset(
        predictions=all_predictions,
        target_names=targets,
        ds=ds,
        features=features,
        scaling_params=scaling_params,
        time_coords=ds.time,
        extra_attrs={
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
