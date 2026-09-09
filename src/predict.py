# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Full-domain prediction entrypoint for pointwise/sequence WWM emulator models (MLP/LSTM).

Invoked via Hydra using ``configs/predict.yaml`` as the base config. Loads a trained
checkpoint and runs vectorized single-step or windowed sequence prediction directly on
preloaded feature arrays (no Lightning ``Trainer``/``DataModule``/``DataLoader``), mirroring
the fast, values-based approach used by ``src/rollout.py`` and ``src/predict_gcn.py``. For
graph-based backbones (GAT/GAT-VN/GAT-LSTM-VN), use ``src/predict_gcn.py`` instead.

If ``data.test_days`` is a list of ``[start, end]`` ranges (e.g. ``[[183, 213], [336, 366]]``),
prediction runs once per range and each result is saved to its own
``predictions_<start>-<end>.nc`` file.
"""

import importlib
import warnings
from pathlib import Path
from typing import Dict, List, Union

import hydra
import numpy as np
import torch
import xarray as xr
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig

from src.data import dataset_utils as du
from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.utils import circular_mae, circular_rmse, get_bin_mask, print_config_tree

# Initialize the distributed rank so RankedLogger works without a Lightning trainer.
if getattr(rank_zero_only, "rank", None) is None:
    rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)

warnings.filterwarnings("ignore")


def predict(cfg: DictConfig) -> Union[xr.Dataset, List[xr.Dataset]]:
    """Predicts given checkpoint directly on preloaded feature arrays (no Trainer).

    If ``cfg.data.test_days`` specifies multiple day ranges, the prediction
    pipeline runs once per range and each result is saved to its own
    ``predictions_<start>-<end>.nc`` file.

    Args:
      cfg (DictConfig): Configuration composed by hydra.

    Returns:
      xr.Dataset or List[xr.Dataset]: Dataset(s) with predictions, one per
        test day range.
    """
    assert cfg.ckpt_path

    log.info("Printing config tree with Rich <cfg.extras.print_config=True>")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    test_ranges = du.normalize_day_ranges(list(cfg.data.test_days))

    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    module_path, class_name = cfg.model._target_.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module_path), class_name)

    backbone = hydra.utils.instantiate(cfg.model.backbone)
    model = model_cls.load_from_checkpoint(cfg.ckpt_path, backbone=backbone, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(f"Model loaded on {device}")

    config_targets = list(cfg.data.targets)
    model_targets = model.target_names
    if config_targets != model_targets:
        raise ValueError(
            f"Target mismatch! Config specifies {config_targets} but model was trained on {model_targets}. "
            f"Please ensure the config targets match the model checkpoint."
        )

    is_sequence_dataset = "Sequence" in cfg.data._target_

    is_multi_range = len(test_ranges) > 1
    results = [
        _predict_single_range(cfg, model, device, is_sequence_dataset, day_range, is_multi_range)
        for day_range in test_ranges
    ]

    return results[0] if len(results) == 1 else results


def _predict_single_range(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
    is_sequence_dataset: bool,
    day_range: List[int],
    is_multi_range: bool,
) -> xr.Dataset:
    """Runs vectorized prediction for a single contiguous day range and saves the result.

    Args:
        cfg: Hydra configuration.
        model: Loaded and device-placed model, shared across ranges.
        device: Device the model was placed on.
        is_sequence_dataset: Whether ``cfg.data`` describes a sequence (LSTM) dataset.
        day_range: Single ``[start_day, end_day]`` range to predict on.
        is_multi_range: Whether multiple ranges are being processed, which
            determines the output filename.

    Returns:
        xr.Dataset: Dataset with predictions for this range.
    """
    data_dir = Path(cfg.data.data_dir)
    features: List[str] = list(cfg.data.features)
    target_names: List[str] = list(cfg.data.targets)

    log.info(f"Opening dataset from {data_dir} for day range {day_range}")
    ds, _ = du.open_and_slice_dataset(data_dir, day_range)
    ds, n_nodes = du.preprocess_variables(ds, features, target_names)
    ds, scaling_params = du.scale_variables(ds, features, cfg.data.get("scaling_config", None))

    n_time = ds.sizes["time"]
    log.info(f"Dataset dimensions: time={n_time}, nSCHISM_hgrid_node={n_nodes}")

    seq_len = int(cfg.data.get("seq_len", 1)) if is_sequence_dataset else 1
    time_start = seq_len - 1 if seq_len > 1 else 0
    pred_n_time = n_time - time_start
    log.info(
        f"{'Sequence' if seq_len > 1 else 'Simple'} dataset detected: seq_len={seq_len}, pred_n_time={pred_n_time}"
    )

    log.info("Preloading feature array into memory ...")
    X_all = np.stack([ds[f].values for f in features], axis=-1).astype(np.float32)

    node_chunk_size = int(cfg.get("node_chunk_size", 65536))
    all_predictions = np.empty((pred_n_time, n_nodes, len(target_names)), dtype=np.float32)

    log.info(f"Running inference over {pred_n_time} time steps, {n_nodes} nodes, node_chunk_size={node_chunk_size} ...")
    with torch.no_grad():
        for time_idx in range(time_start, n_time):
            window = X_all[time_idx - seq_len + 1 : time_idx + 1] if seq_len > 1 else None
            out_chunks = []
            for s in range(0, n_nodes, node_chunk_size):
                if seq_len > 1:
                    # (seq_len, chunk, F) -> (chunk, seq_len, F)
                    x = torch.from_numpy(window[:, s : s + node_chunk_size].transpose(1, 0, 2).copy()).to(device)
                else:
                    x = torch.from_numpy(X_all[time_idx, s : s + node_chunk_size].copy()).to(device)
                out_chunks.append(model.forward(x).cpu().numpy())
            all_predictions[time_idx - time_start] = np.concatenate(out_chunks, axis=0)

            if (time_idx - time_start + 1) % max(1, pred_n_time // 10) == 0:
                log.info(f"  Processed {time_idx - time_start + 1}/{pred_n_time} time steps")

    log.info("Inference complete.")

    # NaN-out boundary nodes if configured (fair comparison with graph models)
    boundary_nodes_csv = cfg.data.get("boundary_nodes_csv", None)
    if boundary_nodes_csv is not None:
        boundary_node_ids = du.load_node_indices(boundary_nodes_csv)
        ds_node_ids = ds.nSCHISM_hgrid_node.values
        id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}
        boundary_positions = np.array([id_to_pos[int(nid)] for nid in boundary_node_ids if int(nid) in id_to_pos])
        all_predictions[:, boundary_positions, :] = np.nan
        log.info(f"Set {len(boundary_positions)} boundary node predictions to NaN.")

    # For sequence datasets, slice the source dataset to match the trimmed prediction
    # time range (needed for correct prev_* alignment during delta-target reconstruction).
    ds_out = ds.isel(time=slice(time_start, None)) if seq_len > 1 else ds

    # ------------------------------------------------------------------
    # Compute test metrics (optional, numpy-only — no extra forward pass)
    # ------------------------------------------------------------------
    if cfg.get("test_metrics", False):
        log.info("Computing test metrics from prediction buffer ...")
        Y_all = np.stack([ds_out[t].values for t in target_names], axis=-1).astype(np.float32)

        valid_mask = ~np.isnan(all_predictions[:, :, 0])
        n_excluded = int(n_nodes - valid_mask[0].sum())
        log.info(f"Metrics exclude {n_excluded} boundary nodes ({valid_mask.sum()} valid samples).")

        metrics: Dict[str, float] = {}
        for target_idx, target_name in enumerate(target_names):
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
        for target_idx, target_name in enumerate(target_names):
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

        # Circular metrics for sin/cos target pairs (e.g. sin_meanWaveDirection + cos_meanWaveDirection).
        # Reconstructs the angle via atan2 and computes circular MAE/RMSE against the original
        # angle variable kept in ds by preprocess_variables.
        target_name_to_idx = {n: i for i, n in enumerate(target_names)}
        sin_targets = {n[len("sin_") :]: n for n in target_names if n.startswith("sin_")}
        cos_targets = {n[len("cos_") :]: n for n in target_names if n.startswith("cos_")}
        for base_var in sin_targets:
            if base_var not in cos_targets:
                continue
            sin_name = sin_targets[base_var]
            cos_name = cos_targets[base_var]
            sin_pred = all_predictions[:, :, target_name_to_idx[sin_name]][valid_mask]
            cos_pred = all_predictions[:, :, target_name_to_idx[cos_name]][valid_mask]
            angle_pred = np.degrees(np.arctan2(sin_pred, cos_pred)) % 360.0
            if base_var in ds_out:
                angle_truth = ds_out[base_var].values.astype(np.float32)[valid_mask]
                c_mae = circular_mae(angle_pred, angle_truth)
                c_rmse = circular_rmse(angle_pred, angle_truth)
                metrics[f"test/circular_mae_{base_var}"] = c_mae
                metrics[f"test/circular_rmse_{base_var}"] = c_rmse
                log.info(f"  {base_var} (reconstructed):  circular RMSE={c_rmse:.4f}°  circular MAE={c_mae:.4f}°")
            else:
                log.warning(f"  Cannot compute circular metrics for {base_var}: not found in dataset.")

        # Log to wandb if a logger is configured
        loggers = instantiate_loggers(cfg.get("logger"))
        for lgr in loggers:
            if hasattr(lgr, "experiment"):
                run = lgr.experiment
                # Use summary for test metrics — not tied to training step axis
                run.summary.update(metrics)
                if hasattr(run, "finish"):
                    run.finish()
        log.info("Test metrics logged.")

    # ------------------------------------------------------------------
    # Build output xarray Dataset
    # ------------------------------------------------------------------
    ds_pred = du.build_prediction_dataset(
        predictions=all_predictions,
        target_names=target_names,
        ds=ds_out,
        features=features,
        scaling_params=scaling_params,
        time_coords=ds_out.time,
        extra_attrs={
            "checkpoint": cfg.ckpt_path,
            "model": cfg.model._target_,
            "seq_len": seq_len,
        },
    )

    # Save predictions to netCDF
    if is_multi_range:
        output_name = f"predictions_{day_range[0]}-{day_range[1]}.nc"
    else:
        output_name = "predictions.nc"
    output_path = f"{cfg.paths.output_dir}/{output_name}"
    log.info(f"Saving predictions to {output_path}")
    ds_pred.to_netcdf(output_path)
    log.info(f"Predictions saved successfully to {output_path}")

    return ds_pred


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg [DictConfig]: DictConfig configuration composed by Hydra.
    """
    predict(cfg)


if __name__ == "__main__":
    main()
