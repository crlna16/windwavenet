# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Shared utility functions for WWM dataset classes.

These pure functions handle Zarr I/O, variable preprocessing, scaling,
data retrieval, weighting, and subsampling. They are used by both
``SimpleWWMDataset`` (torch Dataset) and ``GraphWWMDataset``
(torch_geometric Dataset) to avoid code duplication without shared
inheritance.
"""

import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


# ---------------------------------------------------------------------------
# Zarr I/O helpers
# ---------------------------------------------------------------------------


def open_and_slice_dataset(data_dir: Path, data_range: List[int]) -> Tuple[xr.Dataset, int]:
    """Open a Zarr dataset and slice to the requested day range.

    Args:
        data_dir: Path to the Zarr store.
        data_range: Day range as ``[start_day, end_day]`` (1-indexed, inclusive).

    Returns:
        Tuple of ``(dataset, samples_per_day)``.
    """
    ds = xr.open_dataset(data_dir, engine="zarr")
    time_delta = ds.time.values[1] - ds.time.values[0]
    samples_per_day = int(pd.Timedelta(days=1) / time_delta)
    log.info(f"Dataset has {samples_per_day} samples per day.")
    start_idx = (data_range[0] - 1) * samples_per_day
    end_idx = (data_range[1] - 1) * samples_per_day
    ds = ds.isel(time=slice(start_idx, end_idx))
    log.info(f"Loaded dataset with times ranging from {ds.time.values[0]} to {ds.time.values[-1]}.")
    return ds, samples_per_day


def filter_nodes(ds: xr.Dataset, node_idx_csv: Optional[str]) -> xr.Dataset:
    """Filter nodes based on a CSV file, if provided.

    Args:
        ds: xarray Dataset to filter.
        node_idx_csv: Path to CSV with a ``nSCHISM_hgrid_node`` column, or ``None``.

    Returns:
        Filtered dataset (or the original if *node_idx_csv* is ``None``).
    """
    if node_idx_csv:
        node_idx = load_node_indices(node_idx_csv)
        ds = ds.sel(nSCHISM_hgrid_node=node_idx)
        log.info(f"Selected {len(node_idx)} nodes from {node_idx_csv}.")
    return ds


# ---------------------------------------------------------------------------
# Variable preprocessing
# ---------------------------------------------------------------------------


def compute_prevT(dataset: xr.Dataset, features: List[str]) -> xr.Dataset:
    """Compute ``prev_feature = feature(t-1)`` for features prefixed with ``prev_``.

    Example for ``sigWaveHeight`` at time *t*::

        ds["prev_sigWaveHeight"][t] = ds["sigWaveHeight"][t-1]

    Args:
        dataset: xarray Dataset containing the data.
        features: List of feature variable names.

    Returns:
        Dataset with shifted variables added.
    """
    for feature in features:
        if feature.startswith("prev_"):
            original_var = feature.replace("prev_", "")
            dataset[feature] = dataset[original_var].shift(time=1)
            log.info(f"Added variable {feature}")
    return dataset


def compute_deltaT(dataset: xr.Dataset, targets: List[str]) -> xr.Dataset:
    """Compute ``delta_target = target(t) - target(t-1)`` for targets prefixed with ``delta_``.

    Example for ``sigWaveHeight`` at time *t*::

        ds["delta_sigWaveHeight"][t] = ds["sigWaveHeight"][t] - ds["sigWaveHeight"][t-1]

    Args:
        dataset: xarray Dataset containing the data.
        targets: List of target variable names.

    Returns:
        Dataset with per-target time differences added.
    """
    for target in targets:
        if target.startswith("delta_"):
            original_var = target.replace("delta_", "")
            dataset[target] = dataset[original_var].diff("time")
            log.info(f"Added variable {target}")
    return dataset


def select_variables(dataset: xr.Dataset, variables: List[str]) -> xr.Dataset:
    """Select only the specified variables from a dataset.

    Args:
        dataset: xarray Dataset.
        variables: Variable names to keep.

    Returns:
        Dataset containing only the listed variables.
    """
    return dataset[[v for v in variables]]


def remove_first_timestep_if_nan(dataset: xr.Dataset) -> xr.Dataset:
    """Remove the first time step (always NaN after shift/diff operations).

    Args:
        dataset: xarray Dataset.

    Returns:
        Dataset with the first time step removed.
    """
    log.info("Removing first time step (NaN from shift/diff operations)")
    return dataset.isel(time=slice(1, None))


def preprocess_variables(
    ds: xr.Dataset,
    features: List[str],
    targets: List[str],
    extra_keep_vars: Optional[List[str]] = None,
) -> Tuple[xr.Dataset, int]:
    """Compute derived variables, select relevant ones, and drop NaN timestep.

    Args:
        ds: xarray Dataset.
        features: Feature variable names.
        targets: Target variable names.
        extra_keep_vars: Additional variable names to keep in the dataset
            (e.g. graph topology variables). Defaults to ``None``.

    Returns:
        Tuple of ``(dataset, n_nodes)``.
    """
    ds = compute_prevT(ds, features)
    ds = compute_deltaT(ds, targets)
    selected_targets = targets + [tt.replace("delta_", "") for tt in targets]
    keep = features + selected_targets
    if extra_keep_vars:
        keep = keep + list(extra_keep_vars)
    ds = select_variables(ds, keep)
    ds = remove_first_timestep_if_nan(ds)
    n_nodes = len(ds.nSCHISM_hgrid_node)
    return ds, n_nodes


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def load_scaling_config(config_path: str) -> Dict[str, Dict[str, float]]:
    """Load scaling parameters from a YAML file.

    Args:
        config_path: Path to YAML with ``{variable: {mean: …, std: …}}``.

    Returns:
        Dictionary mapping variable names to their mean and std.
    """
    with open(config_path, "r") as f:
        scaling_params = yaml.safe_load(f)
    return scaling_params


def apply_scaling(
    dataset: xr.Dataset,
    variables: List[str],
    scaling_params: Dict[str, Dict[str, float]],
) -> xr.Dataset:
    """Apply standardisation ``(x - mean) / std`` to the given variables.

    Args:
        dataset: xarray Dataset.
        variables: Variable names to scale.
        scaling_params: ``{var: {mean: …, std: …}}``.

    Returns:
        Dataset with scaled variables (modified in-place).
    """
    for var in variables:
        if var in scaling_params:
            mean = scaling_params[var]["mean"]
            std = scaling_params[var]["std"]
            dataset[var] = (dataset[var] - mean) / std
            log.info(f"Scaled variable '{var}' with mean={mean:.4f}, std={std:.4f}")
        else:
            log.warning(f"No scaling parameters found for variable '{var}', leaving unscaled")
    return dataset


def revert_scaling(
    dataset: xr.Dataset,
    variables: List[str],
    scaling_params: Dict[str, Dict[str, float]],
) -> xr.Dataset:
    """Revert standardisation: ``x * std + mean``.

    Args:
        dataset: xarray Dataset.
        variables: Variable names to un-scale.
        scaling_params: ``{var: {mean: …, std: …}}``.

    Returns:
        Dataset with un-scaled variables (modified in-place).

    Raises:
        ValueError: If scaling parameters are missing for a variable.
    """
    for var in variables:
        if var in scaling_params:
            mean = scaling_params[var]["mean"]
            std = scaling_params[var]["std"]
            dataset[var] = dataset[var] * std + mean
            log.info(f"Reverted scaling for variable {var}.")
        else:
            raise ValueError(f"No parameters found to revert scaling for {var}.")
    return dataset


def scale_variables(
    ds: xr.Dataset,
    features: List[str],
    scaling_config: Optional[str],
) -> Tuple[xr.Dataset, Optional[Dict[str, Dict[str, float]]]]:
    """Load scaling config (if any) and apply feature scaling.

    Args:
        ds: xarray Dataset.
        features: Feature variable names.
        scaling_config: Path to YAML, or ``None``.

    Returns:
        Tuple of ``(dataset, scaling_params)``.
    """
    scaling_params = load_scaling_config(scaling_config) if scaling_config else None
    if scaling_params:
        ds = apply_scaling(ds, features, scaling_params)
    return ds, scaling_params


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------


def retrieve_data(dataset: xr.Dataset, variables: List[str]) -> torch.Tensor:
    """Extract variables from an xarray Dataset as a stacked tensor.

    Args:
        dataset: xarray Dataset.
        variables: Variable names to extract.

    Returns:
        Tensor of shape ``(n_samples, n_variables)`` with time as the slow
        axis and ``nSCHISM_hgrid_node`` as the fast axis.
    """
    start = _time.time()
    da = dataset[variables].to_array(dim="variable")
    da = da.stack(sample=[d for d in da.dims if d != "variable"])
    da = da.transpose("sample", "variable")
    tensor = torch.from_numpy(np.asarray(da.data))
    log.info(f"Finished retrieving {len(variables)} variables in {_time.time() - start:.0f} seconds.")
    return tensor


def load_and_preload_data(
    ds: xr.Dataset,
    features: List[str],
    targets: List[str],
    preload_data: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Optionally preload features and targets into tensors.

    Args:
        ds: xarray Dataset.
        features: Feature variable names.
        targets: Target variable names.
        preload_data: Whether to actually load.

    Returns:
        ``(X, y)`` tensors, or ``(None, None)`` when *preload_data* is ``False``.
    """
    if preload_data:
        X = retrieve_data(ds, features)
        y = retrieve_data(ds, targets)
        return X, y
    log.info("Lazy loading enabled: features and targets will be loaded on-demand")
    return None, None


# ---------------------------------------------------------------------------
# Node indices
# ---------------------------------------------------------------------------


def load_node_indices(csv_path: str) -> np.ndarray:
    """Load node indices from a CSV file.

    Args:
        csv_path: Path to CSV with a ``nSCHISM_hgrid_node`` column.

    Returns:
        Array of node indices.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If the required column is missing.
    """
    node_idx_path = Path(csv_path)
    if not node_idx_path.exists():
        raise FileNotFoundError(f"Node index CSV not found: {node_idx_path}")
    node_idx_df = pd.read_csv(node_idx_path)
    if "nSCHISM_hgrid_node" not in node_idx_df.columns:
        raise ValueError("Node index CSV must have a 'nSCHISM_hgrid_node' column")
    return node_idx_df["nSCHISM_hgrid_node"].to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# Weighting and subsampling
# ---------------------------------------------------------------------------


def compute_weights(
    y: torch.Tensor,
    targets: List[str],
    strategy: Optional[str],
    power: float = 1.0,
) -> Optional[torch.Tensor]:
    """Compute and normalise per-sample weights.

    Weights are normalised so that their mean equals 1.0.

    Args:
        y: Target tensor of shape ``(n_samples, n_targets)``.
        targets: Target variable names (for logging).
        strategy: ``None`` or ``"z_score"``.
        power: Exponent controlling weighting intensity.

    Returns:
        Weight tensor of shape ``(n_samples, n_targets)``, or ``None``.

    Raises:
        ValueError: If *strategy* is not recognised.
    """
    if strategy is None:
        log.info("No sample weight strategy specified, skipping weight computation")
        return None

    log.info(f"Computing sample weights with strategy '{strategy}'...")

    n_samples = len(y)
    n_targets = len(targets)

    if strategy == "z_score":
        z_scores = (y - torch.mean(y)) / torch.std(y)
        weights = 1.0 - torch.exp(-(z_scores**2) / torch.tensor(power))
    else:
        raise ValueError(f"Unknown sample_weight_strategy: '{strategy}'. Supported: None, 'z_score'")

    # Normalise per target so mean == 1.0
    for i in range(n_targets):
        weight_sum = weights[:, i].sum()
        weights[:, i] = weights[:, i] * (n_samples / weight_sum)
        log.info(
            f"Target {targets[i]}: normalized weights, "
            f"(mean={weights[:, i].mean():.4f}, std={weights[:, i].std():.4f}, "
            f"min={weights[:, i].min():.4f}, max={weights[:, i].max():.4f})"
        )

    weights = weights.float()
    log.info(f"Sample weights computed and normalized for {n_samples} samples")
    return weights


def subsample_by_zscore(
    X: Optional[torch.Tensor],
    y: torch.Tensor,
    subsample_prob: float,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Subsample based on z-score-derived retention probabilities.

    Samples with higher absolute z-scores (outliers) are more likely to be
    retained. Retention probability: ``p = 1 - exp(-z² / 2)``, scaled so the
    expected retention rate equals *subsample_prob*.

    Args:
        X: Feature tensor ``(n_samples, n_features)`` or ``None``.
        y: Target tensor ``(n_samples, n_targets)``.
        subsample_prob: Target retention rate.

    Returns:
        ``(X, y)`` after subsampling. ``X`` may be ``None`` if the input was.
    """
    log.info(f"Subsampling dataset based on z-score with target rate {subsample_prob}...")

    n_samples_before = len(y)

    z_scores = (y - torch.mean(y, dim=0)) / torch.std(y, dim=0)
    max_abs_z = torch.abs(z_scores).max(dim=1)[0]

    retention_probs = 1.0 - torch.exp(-(max_abs_z**2) / 2.0)

    if retention_probs.mean() > 0:
        retention_probs = retention_probs * (subsample_prob / retention_probs.mean())
        retention_probs = torch.clamp(retention_probs, 0.0, 1.0)

    random_vals = torch.rand(n_samples_before)
    keep_mask = random_vals < retention_probs

    if X is not None:
        X = X[keep_mask]
    y = y[keep_mask]

    n_samples_after = len(y)
    retention_rate = n_samples_after / n_samples_before * 100
    log.info(f"Subsampling complete: kept {n_samples_after}/{n_samples_before} samples ({retention_rate:.2f}%)")

    return X, y


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def reconstruct_from_delta(
    delta_predictions: np.ndarray,
    dataset: xr.Dataset,
    target_name: str,
    features: List[str],
    scaling_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, xr.Dataset]:
    """Reconstruct absolute values from delta predictions.

    For targets like ``delta_sigWaveHeight``::

        reconstructed[t] = prev_value[t] + delta_prediction[t]

    Args:
        delta_predictions: Array of shape ``(n_time, n_nodes)``.
        dataset: xarray Dataset containing the source data.
        target_name: Delta target name (e.g. ``"delta_sigWaveHeight"``).
        features: Feature names used by the model.
        scaling_params: Optional scaling parameters for reverting scaling.

    Returns:
        ``(reconstructed_values, updated_dataset)``.

    Raises:
        ValueError: If *target_name* does not start with ``"delta_"``.
    """
    if not target_name.startswith("delta_"):
        raise ValueError(f"Target '{target_name}' is not a delta target")

    original_var = target_name.replace("delta_", "")
    prev_var = f"prev_{original_var}"

    if prev_var not in dataset:
        log.info(f"Computing {prev_var} for reconstruction")
        dataset = compute_prevT(dataset, [prev_var])

    if scaling_params and prev_var in features:
        dataset = revert_scaling(dataset, [prev_var], scaling_params)

    prev_values = dataset[prev_var].values
    reconstructed = prev_values + delta_predictions

    return reconstructed, dataset


# ---------------------------------------------------------------------------
# Prediction post-processing
# ---------------------------------------------------------------------------


def unscale_predictions(
    predictions: np.ndarray,
    target_name: str,
    features: List[str],
    scaling_params: Optional[Dict[str, Dict[str, float]]],
) -> np.ndarray:
    """Revert z-score scaling on a numpy prediction array.

    Only applies if *target_name* appears in both *features* and *scaling_params*.

    Args:
        predictions: Array of predictions (any shape).
        target_name: Name of the target variable.
        features: Feature names used by the model.
        scaling_params: ``{var: {mean: …, std: …}}`` or ``None``.

    Returns:
        Un-scaled predictions (same shape as input).
    """
    if not scaling_params or target_name not in features or target_name not in scaling_params:
        return predictions

    mean = scaling_params[target_name]["mean"]
    std = scaling_params[target_name]["std"]
    log.info(f"Reverting scaling for target '{target_name}' with mean={mean:.4f}, std={std:.4f}")
    return predictions * std + mean


def build_prediction_dataset(
    predictions: np.ndarray,
    target_names: List[str],
    ds: xr.Dataset,
    features: List[str],
    scaling_params: Optional[Dict[str, Dict[str, float]]],
    time_coords: xr.DataArray,
    extra_attrs: Optional[Dict[str, object]] = None,
) -> xr.Dataset:
    """Build an xarray Dataset from raw model predictions.

    Handles delta-target reconstruction, un-scaling, and coordinate
    attachment.  This is the single source of truth for converting a
    ``(n_time, n_nodes, n_targets)`` prediction buffer into the
    standard NetCDF output format.

    Args:
        predictions: Array of shape ``(n_time, n_nodes, n_targets)``.
        target_names: Ordered list of target variable names matching
            the last axis of *predictions*.
        ds: Source xarray Dataset (supplies coordinates and ``prev_*``
            variables for delta reconstruction).
        features: Feature names used by the model.
        scaling_params: Scaling parameters, or ``None``.
        time_coords: Time coordinate for the output (may be a slice of
            ``ds.time`` for sequence models).
        extra_attrs: Additional global attributes for the output dataset.

    Returns:
        xr.Dataset with standard dimensions ``(time, nSCHISM_hgrid_node)``
        and coordinates ``nSCHISM_hgrid_node_x`` / ``nSCHISM_hgrid_node_y``.
    """
    data_vars: Dict[str, object] = {}

    for i, target_name in enumerate(target_names):
        target_preds = predictions[:, :, i]  # (n_time, n_nodes)

        if target_name.startswith("delta_"):
            original_var = target_name.replace("delta_", "")
            log.info(f"Reconstructing {original_var} from delta predictions")
            reconstructed, ds = reconstruct_from_delta(
                delta_predictions=target_preds,
                dataset=ds,
                target_name=target_name,
                features=features,
                scaling_params=scaling_params,
            )
            data_vars[original_var] = (["time", "nSCHISM_hgrid_node"], reconstructed)
            data_vars[target_name] = (["time", "nSCHISM_hgrid_node"], target_preds)
        else:
            target_preds = unscale_predictions(target_preds, target_name, features, scaling_params)
            data_vars[target_name] = (["time", "nSCHISM_hgrid_node"], target_preds)

    output_target_names = [tn.replace("delta_", "") for tn in target_names]

    attrs: Dict[str, object] = {
        "description": f"Predictions for {', '.join(output_target_names)}",
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    ds_pred = xr.Dataset(
        data_vars,
        coords={
            "time": time_coords,
            "nSCHISM_hgrid_node": ds.nSCHISM_hgrid_node,
            "nSCHISM_hgrid_node_x": ds.SCHISM_hgrid_node_x,
            "nSCHISM_hgrid_node_y": ds.SCHISM_hgrid_node_y,
        },
        attrs=attrs,
    )

    return ds_pred
