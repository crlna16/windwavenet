# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Pseudo-coupled autoregressive rollout evaluation.

Evaluates a trained emulator in the configuration it will face when coupled
back to SCHISM: at every time step the forcing features (wind, elevation,
currents, ...) come from ground truth — mimicking the external update by the
host model — while every ``prev_<target>`` feature is fed back from the
emulator's OWN prediction at the previous step instead of from ground truth.

The rollout starts from a ground-truth initial state and runs over the test
day range.  Errors are reported per lead time (hours since the last observed
wave state), next to two baselines:

- persistence: holds the last observed state constant for all lead times.
- teacher-forced: same model, but ``prev_<target>`` features always come
  from ground truth instead of the model's own rolled-forward prediction —
  isolates the one-step model error from the compounding autoregressive error.

Supported model families (auto-detected from ``data._target_``):

- ``WWMDataModule``                   → pointwise MLP
- ``SequenceWWMDataModule``           → LSTM (per-node sequence-to-one)
- ``GraphWWMDataModule``              → MultiHeadGAT / MultiHeadGATVN
- ``SequenceGraphWWMDataModule``      → graph models with sequences at all nodes
- ``CenterHistoryGraphWWMDataModule`` → MultiHeadGATLSTMVN

Graph models are evaluated over the full mesh with the same disjoint domain
partitioning as ``predict_gcn.py``: every node belongs to exactly one
sub-graph, sized within ``+/- rollout.disjoint_partition_tolerance`` of
``data.max_nodes``.

Usage::

    python src/rollout.py experiment=swh_1h_mlp ckpt_path=<ckpt>
    python src/rollout.py experiment=swh_1h_lstm ckpt_path=<ckpt>
    python src/rollout.py experiment=swh_1h_multi-head-gat-vn ckpt_path=<ckpt>
    python src/rollout.py experiment=swh_1h_multi-head-gat-lstm-vn ckpt_path=<ckpt>

    # Limit the horizon to 48 rollout steps:
    python src/rollout.py experiment=swh_1h_lstm ckpt_path=<ckpt> rollout.steps=48

Notes:

- Delta targets (``delta_*``) are not supported: feeding back a delta
  prediction requires reconstructing the absolute state, which couples the
  reconstruction error into the rollout in a way that needs its own analysis.
- If ``data.boundary_nodes_csv`` is set, the boundary nodes' ``prev_*``
  features are held at ground truth (mimicking prescribed boundary
  conditions in the coupled setup), their predictions are NaN'd in the
  output, and they are excluded from metrics.

Outputs (in the Hydra run directory):

- ``rollout_predictions_from_<forecast_start>_<offset_hours>.nc``   — full-domain predictions per rollout step
- ``rollout_metrics_lead_from_<forecast_start>_<offset_hours>.csv`` — per-lead-time RMSE / MAE / bias per target,
  persistence-baseline RMSE, and per-bin RMSE for configured target_bins
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data as GData

from src.data import dataset_utils as du
from src.data import graph_utils as gu
from src.data.WWMDatasets import WWMGraphData
from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.utils import get_bin_mask, print_config_tree
from src.utils.predict_utils import load_model_from_checkpoint

# Initialize the distributed rank so RankedLogger works without a Lightning trainer.
if getattr(rank_zero_only, "rank", None) is None:
    rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Re-exported for backward compatibility (lives in dataset_utils since it is
# also used by src/model/noise.py for prev_* noise injection).
build_feedback_map = du.build_feedback_map

# ---------------------------------------------------------------------------
# Per-family single-step predictors
#
# Each factory returns ``step(t) -> (n_nodes, n_targets)`` predictions in
# physical units for time index ``t``, reading inputs from the shared rolling
# feature buffer ``X_all`` (which the rollout loop mutates between steps).
# ---------------------------------------------------------------------------


def make_pointwise_predictor(
    model, X_all: np.ndarray, device: torch.device, node_chunk_size: int
) -> Callable[[int], np.ndarray]:
    """Pointwise MLP: independent forward per node at the current time step."""
    n_nodes = X_all.shape[1]

    def step(t: int) -> np.ndarray:
        out = []
        for s in range(0, n_nodes, node_chunk_size):
            x = torch.from_numpy(X_all[t, s : s + node_chunk_size].copy()).to(device)
            out.append(model.forward(x).cpu().numpy())
        return np.concatenate(out, axis=0)

    return step


def make_sequence_predictor(
    model, X_all: np.ndarray, seq_len: int, device: torch.device, node_chunk_size: int
) -> Callable[[int], np.ndarray]:
    """LSTM: per-node window of the last ``seq_len`` steps ending at ``t``."""
    n_nodes = X_all.shape[1]

    def step(t: int) -> np.ndarray:
        window = X_all[t - seq_len + 1 : t + 1]  # (seq_len, n_nodes, n_features)
        out = []
        for s in range(0, n_nodes, node_chunk_size):
            # (seq_len, chunk, F) -> (chunk, seq_len, F)
            x = torch.from_numpy(window[:, s : s + node_chunk_size].transpose(1, 0, 2).copy()).to(device)
            out.append(model.forward(x).cpu().numpy())
        return np.concatenate(out, axis=0)

    return step


def make_graph_predictor(
    model,
    X_all: np.ndarray,
    partitions: List[Tuple[int, np.ndarray, torch.Tensor, int]],
    edge_weights: List[torch.Tensor],
    owned_map: List[Tuple[np.ndarray, np.ndarray]],
    n_targets: int,
    device: torch.device,
    batch_size_graphs: int,
    seq_len: int,
    center_history: bool,
) -> Callable[[int], np.ndarray]:
    """Graph models: batched inference over the greedy domain partitions.

    Mirrors the inference loop of ``predict_gcn.py`` — see there for details
    on partition ownership.
    """
    n_nodes = X_all.shape[1]

    def step(t: int) -> np.ndarray:
        preds_full = np.full((n_nodes, n_targets), np.nan, dtype=np.float32)
        X_t = X_all[t]
        window = X_all[t - seq_len + 1 : t + 1] if seq_len > 1 else None

        for batch_start in range(0, len(partitions), batch_size_graphs):
            batch_end = min(batch_start + batch_size_graphs, len(partitions))
            data_list = []
            for i, (_, subgraph_nodes, edge_index, center_local_idx) in enumerate(partitions[batch_start:batch_end]):
                if center_history and seq_len > 1:
                    x = torch.from_numpy(X_t[subgraph_nodes].copy())
                    center_global = subgraph_nodes[center_local_idx]
                    x_center_seq = torch.from_numpy(window[:, center_global, :].copy())
                    data_list.append(
                        WWMGraphData(
                            x=x,
                            x_center_seq=x_center_seq,
                            edge_index=edge_index,
                            edge_weight=edge_weights[batch_start + i],
                            center_idx=center_local_idx,
                        )
                    )
                elif seq_len > 1:
                    x = torch.from_numpy(window[:, subgraph_nodes, :].transpose(1, 0, 2).copy())
                    data_list.append(GData(x=x, edge_index=edge_index, edge_weight=edge_weights[batch_start + i]))
                else:
                    x = torch.from_numpy(X_t[subgraph_nodes].copy())
                    data_list.append(GData(x=x, edge_index=edge_index, edge_weight=edge_weights[batch_start + i]))

            batch = Batch.from_data_list(data_list).to(device)
            preds = model.forward(batch).cpu().numpy()
            ptr = batch.ptr.cpu().numpy()
            for j in range(batch_end - batch_start):
                owned_local, owned_global = owned_map[batch_start + j]
                seg = preds[ptr[j] : ptr[j + 1]]
                preds_full[owned_global] = seg[owned_local]
        return preds_full

    return step


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------


def rollout(cfg: DictConfig) -> xr.Dataset:
    """Run the pseudo-coupled rollout and write predictions + lead-time metrics.

    Args:
        cfg: Hydra configuration (see ``configs/rollout.yaml``).

    Returns:
        xr.Dataset with rollout predictions.
    """
    assert cfg.ckpt_path, "ckpt_path must be provided"

    log.info("Printing config tree")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    features: List[str] = list(cfg.data.features)
    targets: List[str] = list(cfg.data.targets)
    test_ranges = du.normalize_day_ranges(list(cfg.data.test_days))
    if len(test_ranges) > 1:
        raise NotImplementedError(
            "The rollout is autoregressive and requires a contiguous test range; "
            "run once per range instead, e.g. data.test_days=[212,242]."
        )
    test_days: List[int] = test_ranges[0]

    for t in targets:
        if t.startswith("delta_"):
            raise NotImplementedError(
                f"Rollout does not support delta targets (got '{t}'). Use an absolute-target experiment."
            )

    # ------------------------------------------------------------------
    # 1.  Detect model family from the data module class
    # ------------------------------------------------------------------
    data_target: str = cfg.data._target_
    is_graph = "Graph" in data_target
    center_history = "CenterHistory" in data_target
    is_sequence = ("Sequence" in data_target) and not is_graph
    seq_len = int(cfg.data.get("seq_len", 1)) if (is_sequence or is_graph) else 1
    mode = "graph" if is_graph else ("sequence" if is_sequence else "pointwise")
    log.info(f"Model family: {mode}" + (f" (center_history, seq_len={seq_len})" if center_history else ""))

    # ------------------------------------------------------------------
    # 2.  Open and preprocess dataset (test day range)
    # ------------------------------------------------------------------
    data_dir = Path(cfg.data.data_dir)
    graph_features: List[str] = list(cfg.data.get("graph_features", [])) if is_graph else []

    log.info(f"Opening dataset from {data_dir}")
    ds, samples_per_day = du.open_and_slice_dataset(data_dir, test_days)
    ds, n_nodes = du.preprocess_variables(ds, features, targets, extra_keep_vars=graph_features)
    ds, scaling_params = du.scale_variables(ds, features, cfg.data.get("scaling_config", None))
    n_time = ds.sizes["time"]
    n_targets = len(targets)
    log.info(f"Dataset: {n_nodes} nodes, {n_time} time steps, {n_targets} targets")

    # ------------------------------------------------------------------
    # 3.  Graph topology (graph mode only)
    # ------------------------------------------------------------------
    partitions = edge_weights = owned_map = None
    boundary_positions: Optional[np.ndarray] = None
    boundary_nodes_csv = cfg.data.get("boundary_nodes_csv", None)
    if boundary_nodes_csv is not None:
        boundary_node_ids = du.load_node_indices(boundary_nodes_csv)
        ds_node_ids = ds.nSCHISM_hgrid_node.values
        id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}
        boundary_positions = np.array([id_to_pos[int(nid)] for nid in boundary_node_ids if int(nid) in id_to_pos])
        log.info(f"Loaded {len(boundary_positions)} boundary node positions.")

    if is_graph:
        node_x = ds["SCHISM_hgrid_node_x"].values
        node_y = ds["SCHISM_hgrid_node_y"].values
        log.info("Building adjacency list ...")
        adjacency = gu.build_adjacency(ds)
        if boundary_positions is not None:
            adjacency = gu.remove_boundary_nodes(adjacency, boundary_positions)
        ds = ds.drop_vars([v for v in graph_features if v in ds], errors="ignore")

        max_nodes_per_partition = int(cfg.data.max_nodes)
        tolerance = float(cfg.rollout.get("disjoint_partition_tolerance", 0.1))
        log.info(
            f"Partitioning domain (disjoint) with target_nodes={max_nodes_per_partition}, tolerance={tolerance} ..."
        )
        partitions = gu.partition_domain_disjoint(
            n_nodes=n_nodes,
            adjacency=adjacency,
            node_x=node_x,
            node_y=node_y,
            target_nodes=max_nodes_per_partition,
            tolerance=tolerance,
            excluded_nodes=boundary_positions,
        )
        edge_weight_scale = float(cfg.data.edge_weight_scale)
        edge_weights = [
            gu.compute_edge_weights(edge_index, subgraph_nodes, node_x, node_y, scale=edge_weight_scale)
            for _, subgraph_nodes, edge_index, _ in partitions
        ]

        # Partition ownership: first-seen wins (same as predict_gcn.py)
        assigned = np.zeros(n_nodes, dtype=bool)
        owned_map = []
        for _, subgraph_nodes, _, _ in partitions:
            mask = ~assigned[subgraph_nodes]
            owned_local = np.where(mask)[0]
            owned_global = subgraph_nodes[owned_local]
            assigned[owned_global] = True
            owned_map.append((owned_local, owned_global))
        log.info(f"Ownership assigned: {assigned.sum()}/{n_nodes} nodes covered.")

    # ------------------------------------------------------------------
    # 4.  Preload feature buffer and ground truth
    # ------------------------------------------------------------------
    log.info("Preloading feature buffer and ground-truth targets ...")
    X_all = np.stack([ds[f].values for f in features], axis=-1).astype(np.float32)
    Y_all = np.stack([ds[t].values for t in targets], axis=-1).astype(np.float32)
    log.info(f"Feature buffer: {X_all.shape}, truth: {Y_all.shape}")

    feedback = build_feedback_map(features, targets, scaling_params)
    if not feedback:
        log.warning("No prev_<target> feedback channels found — the rollout is identical to teacher-forced prediction!")

    # Boundary nodes keep ground-truth prev_* features (prescribed boundary conditions).
    boundary_from_truth = bool(cfg.rollout.get("boundary_from_truth", True))
    boundary_truth: Dict[int, np.ndarray] = {}
    if boundary_positions is not None and boundary_from_truth:
        for f_idx, _, _, _ in feedback:
            boundary_truth[f_idx] = X_all[:, boundary_positions, f_idx].copy()
        log.info(f"Boundary nodes keep ground-truth values in {len(boundary_truth)} feedback channel(s).")

    # ------------------------------------------------------------------
    # 5.  Load model and build the single-step predictor
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(cfg, device)

    node_chunk_size = int(cfg.rollout.get("node_chunk_size", 65536))
    batch_size_graphs = int(cfg.rollout.get("batch_size_graphs", 32))

    if mode == "pointwise":
        step = make_pointwise_predictor(model, X_all, device, node_chunk_size)
    elif mode == "sequence":
        step = make_sequence_predictor(model, X_all, seq_len, device, node_chunk_size)
    else:
        step = make_graph_predictor(
            model,
            X_all,
            partitions,
            edge_weights,
            owned_map,
            n_targets,
            device,
            batch_size_graphs,
            seq_len,
            center_history,
        )

    # Teacher-forced predictor: identical model, but bound to a frozen copy of
    # the ground-truth buffer so prev_* feedback is never overwritten with
    # the model's own predictions.
    teacher_forced = bool(cfg.rollout.get("teacher_forced", True))
    step_tf = None
    if teacher_forced:
        X_all_truth = X_all.copy()
        if mode == "pointwise":
            step_tf = make_pointwise_predictor(model, X_all_truth, device, node_chunk_size)
        elif mode == "sequence":
            step_tf = make_sequence_predictor(model, X_all_truth, seq_len, device, node_chunk_size)
        else:
            step_tf = make_graph_predictor(
                model,
                X_all_truth,
                partitions,
                edge_weights,
                owned_map,
                n_targets,
                device,
                batch_size_graphs,
                seq_len,
                center_history,
            )

    # ------------------------------------------------------------------
    # 6.  Rollout loop
    # ------------------------------------------------------------------
    # First predicted step: needs a full ground-truth history window and at
    # least one anchor step before it for the persistence baseline. An
    # optional offset_hours shifts this later within the loaded test range,
    # e.g. test_days=[5, 10] with offset_hours=5 starts at step 5 of day 5.
    offset_hours = int(cfg.rollout.get("offset_hours", 0))
    steps_per_hour = samples_per_day / 24
    offset_steps = int(round(offset_hours * steps_per_hour))
    if abs(offset_steps - offset_hours * steps_per_hour) > 1e-6:
        raise ValueError(
            f"offset_hours={offset_hours} does not align with the dataset's {samples_per_day} samples/day."
        )
    t_first = max(seq_len - 1, 1) + offset_steps
    steps_cap = cfg.rollout.get("steps", None)
    n_time_eff = n_time if steps_cap is None else min(n_time, t_first + int(steps_cap))
    n_leads = n_time_eff - t_first
    if n_leads <= 0:
        raise ValueError(f"No rollout steps to run: t_first={t_first}, n_time={n_time_eff}.")
    log.info(
        f"Rollout: {n_leads} steps starting at time index {t_first} (anchor at {t_first - 1}), "
        f"offset_hours={offset_hours}."
    )

    preds_out = np.full((n_time_eff, n_nodes, n_targets), np.nan, dtype=np.float32)
    preds_tf_out = np.full((n_time_eff, n_nodes, n_targets), np.nan, dtype=np.float32) if teacher_forced else None

    with torch.no_grad():
        for t in range(t_first, n_time_eff):
            y_hat = step(t)  # (n_nodes, n_targets), physical units
            preds_out[t] = y_hat
            if teacher_forced:
                preds_tf_out[t] = step_tf(t)

            if t + 1 < n_time_eff:
                for f_idx, t_idx, mean, std in feedback:
                    fb = (y_hat[:, t_idx] - mean) / std
                    bad = ~np.isfinite(fb)
                    if bad.any():
                        log.warning(
                            f"t={t}: {int(bad.sum())} non-finite feedback values in feature "
                            f"{features[f_idx]} — keeping ground truth there."
                        )
                        fb[bad] = X_all[t + 1, bad, f_idx]
                    X_all[t + 1, :, f_idx] = fb
                    if f_idx in boundary_truth:
                        X_all[t + 1, boundary_positions, f_idx] = boundary_truth[f_idx][t + 1]

            if (t - t_first + 1) % max(1, n_leads // 10) == 0:
                log.info(f"  Rollout step {t - t_first + 1}/{n_leads}")

    log.info("Rollout complete.")

    # NaN-out boundary node predictions for fair comparison and metrics.
    if boundary_positions is not None:
        preds_out[:, boundary_positions, :] = np.nan
        if teacher_forced:
            preds_tf_out[:, boundary_positions, :] = np.nan

    # ------------------------------------------------------------------
    # 7.  Lead-time metrics (model vs persistence-of-initial-state)
    # ------------------------------------------------------------------
    valid_nodes = np.ones(n_nodes, dtype=bool)
    if boundary_positions is not None:
        valid_nodes[boundary_positions] = False

    anchor = Y_all[t_first - 1]  # (n_nodes, n_targets): last observed state
    target_bins = dict(cfg.model.get("target_bins") or {})

    rows = []
    for t in range(t_first, n_time_eff):
        lead = t - t_first + 1
        row: Dict[str, object] = {"lead": lead, "time": str(ds.time.values[t])}
        for t_idx, name in enumerate(targets):
            pred = preds_out[t, valid_nodes, t_idx]
            truth = Y_all[t, valid_nodes, t_idx]
            persist = anchor[valid_nodes, t_idx]
            err = pred - truth
            row[f"rmse_{name}"] = float(np.sqrt(np.nanmean(err**2)))
            row[f"mae_{name}"] = float(np.nanmean(np.abs(err)))
            row[f"bias_{name}"] = float(np.nanmean(err))
            row[f"rmse_persistence_{name}"] = float(np.sqrt(np.nanmean((persist - truth) ** 2)))
            if teacher_forced:
                pred_tf = preds_tf_out[t, valid_nodes, t_idx]
                row[f"rmse_teacher_forced_{name}"] = float(np.sqrt(np.nanmean((pred_tf - truth) ** 2)))

            if name in target_bins:
                bins = list(target_bins[name])
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
                    row[f"rmse_{name}_{bin_label}"] = float(np.sqrt(np.nanmean(err[mask] ** 2)))
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    forecast_start = test_days[0]
    metrics_path = f"{cfg.paths.output_dir}/rollout_metrics_lead_from_{forecast_start}_{offset_hours}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"Lead-time metrics saved to {metrics_path}")

    # Summary: overall and final-lead RMSE per target, logged and sent to wandb.
    summary: Dict[str, float] = {}
    for name in targets:
        summary[f"rollout/rmse_{name}"] = float(
            np.sqrt(
                np.nanmean(
                    (
                        preds_out[t_first:, valid_nodes, targets.index(name)]
                        - Y_all[t_first:n_time_eff, valid_nodes, targets.index(name)]
                    )
                    ** 2
                )
            )
        )
        summary[f"rollout/rmse_{name}_final_lead"] = float(metrics_df[f"rmse_{name}"].iloc[-1])
        summary[f"rollout/rmse_persistence_{name}"] = float(
            np.sqrt(
                np.nanmean(
                    (
                        anchor[valid_nodes, targets.index(name)][None, :]
                        - Y_all[t_first:n_time_eff, valid_nodes, targets.index(name)]
                    )
                    ** 2
                )
            )
        )
        if teacher_forced:
            summary[f"rollout/rmse_teacher_forced_{name}"] = float(
                np.sqrt(
                    np.nanmean(
                        (
                            preds_tf_out[t_first:, valid_nodes, targets.index(name)]
                            - Y_all[t_first:n_time_eff, valid_nodes, targets.index(name)]
                        )
                        ** 2
                    )
                )
            )
    for key, val in summary.items():
        log.info(f"  {key}: {val:.6f}")

    loggers = instantiate_loggers(cfg.get("logger"))
    for lgr in loggers:
        if hasattr(lgr, "experiment"):
            run = lgr.experiment
            run.summary.update(summary)
            if hasattr(run, "finish"):
                run.finish()

    # ------------------------------------------------------------------
    # 8.  Save predictions as netCDF
    # ------------------------------------------------------------------
    ds_pred = None
    if bool(cfg.rollout.get("save_predictions", True)):
        ds_pred = du.build_prediction_dataset(
            predictions=preds_out[t_first:],
            target_names=targets,
            ds=ds,
            features=features,
            scaling_params=scaling_params,
            time_coords=ds.time[t_first:n_time_eff],
            extra_attrs={
                "checkpoint": cfg.ckpt_path,
                "model": cfg.model._target_,
                "rollout": "autoregressive prev_* feedback, ground-truth forcings",
                "rollout_t_first": t_first,
                "seq_len": seq_len,
            },
        )
        output_path = f"{cfg.paths.output_dir}/rollout_predictions_from_{forecast_start}_{offset_hours}.nc"
        ds_pred.to_netcdf(output_path)
        log.info(f"Rollout predictions saved to {output_path}")

    return ds_pred


@hydra.main(version_base="1.3", config_path="../configs", config_name="rollout.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for rollout evaluation."""
    rollout(cfg)


if __name__ == "__main__":
    main()
