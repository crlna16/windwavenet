# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Shared helpers for full-domain prediction entrypoints (``src/predict.py`` and ``src/predict_gcn.py``).

Factors out the logic that is identical between the pointwise/sequence (MLP/LSTM) and
graph (GAT/GAT-VN/GAT-LSTM-VN) prediction pipelines: checkpoint loading with target-name
validation, test-metric computation (RMSE/MAE/bias, per-bin RMSE, circular metrics, wandb
logging), and saving the output dataset to netCDF.
"""

import importlib
from typing import Dict, List, Union

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig

from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.utils import circular_mae, circular_rmse, get_bin_mask

log = RankedLogger(__name__, rank_zero_only=True)


def load_model_from_checkpoint(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Instantiates the backbone, loads the checkpoint, and validates target names.

    Args:
        cfg: Hydra configuration with ``cfg.model._target_``, ``cfg.model.backbone``,
            ``cfg.ckpt_path``, and ``cfg.data.targets``.
        device: Device to move the loaded model to.

    Returns:
        The loaded model in eval mode, placed on ``device``.

    Raises:
        ValueError: If ``cfg.data.targets`` does not match the checkpoint's target names.
    """
    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    module_path, class_name = cfg.model._target_.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module_path), class_name)

    backbone = hydra.utils.instantiate(cfg.model.backbone)
    model = model_cls.load_from_checkpoint(cfg.ckpt_path, backbone=backbone, strict=False)
    model.eval()
    model = model.to(device)
    log.info(f"Model loaded on {device}")

    config_targets = list(cfg.data.targets)
    model_targets = model.target_names
    if config_targets != model_targets:
        raise ValueError(
            f"Target mismatch! Config specifies {config_targets} but model was trained on {model_targets}. "
            f"Please ensure the config targets match the model checkpoint."
        )

    return model


def compute_test_metrics(
    cfg: DictConfig,
    all_predictions: np.ndarray,
    ds_out: xr.Dataset,
    target_names: List[str],
    n_nodes: int,
    n_boundary: Union[int, None] = None,
) -> Dict[str, float]:
    """Computes and logs test metrics from a prediction buffer (numpy-only, no extra forward pass).

    Computes overall RMSE/MAE/bias per target, per-bin RMSE (if ``cfg.model.target_bins`` is
    configured), and circular MAE/RMSE for sin/cos target pairs. Logs the result to wandb if a
    logger is configured. No-op unless ``cfg.test_metrics`` is truthy.

    Args:
        cfg: Hydra configuration.
        all_predictions: Prediction buffer of shape ``(n_time, n_nodes, n_targets)``, already
            trimmed to match ``ds_out``'s time dimension. Excluded/unpredicted nodes must be NaN.
        ds_out: Source dataset, trimmed to the same time range as ``all_predictions``, supplying
            ground truth for ``target_names`` and any circular base variables.
        target_names: Ordered list of target variable names matching the last axis of
            ``all_predictions``.
        n_nodes: Total number of nodes in the domain (used only for the excluded-node log message).
        n_boundary: If given, the expected number of excluded (e.g. boundary) nodes; a warning is
            logged if the actually-excluded count differs (used by the graph pipeline to detect
            nodes outside every partition).

    Returns:
        Dict[str, float]: Computed metrics, keyed like ``"test/rmse_<target>"``. Empty if
            ``cfg.test_metrics`` is falsy.
    """
    if not cfg.get("test_metrics", False):
        return {}

    log.info("Computing test metrics from prediction buffer ...")
    Y_all = np.stack([ds_out[t].values for t in target_names], axis=-1).astype(np.float32)

    valid_mask = ~np.isnan(all_predictions[:, :, 0])
    n_excluded = int(n_nodes - valid_mask[0].sum())
    log.info(f"Metrics exclude {n_excluded} boundary nodes ({valid_mask.sum()} valid samples).")
    if n_boundary is not None and n_excluded != n_boundary:
        log.warning(
            f"{n_excluded} nodes are unpredicted but {n_boundary} boundary nodes were configured — "
            "the mesh has nodes outside every partition."
        )

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
    # angle variable kept in ds_out by preprocess_variables.
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

    return metrics


def save_predictions(ds_pred: xr.Dataset, cfg: DictConfig, day_range: List[int], is_multi_range: bool) -> str:
    """Saves a prediction dataset to netCDF under ``cfg.paths.output_dir``.

    Args:
        ds_pred: Prediction dataset to save.
        cfg: Hydra configuration supplying ``cfg.paths.output_dir``.
        day_range: The ``[start_day, end_day]`` range this dataset covers, used in the filename
            when ``is_multi_range`` is True.
        is_multi_range: Whether multiple day ranges are being processed, which determines whether
            the range is embedded in the output filename.

    Returns:
        str: The path the dataset was saved to.
    """
    if is_multi_range:
        output_name = f"predictions_{day_range[0]}-{day_range[1]}.nc"
    else:
        output_name = "predictions.nc"
    output_path = f"{cfg.paths.output_dir}/{output_name}"
    log.info(f"Saving predictions to {output_path}")
    ds_pred.to_netcdf(output_path)
    log.info(f"Predictions saved successfully to {output_path}")
    return output_path
