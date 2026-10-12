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

from scipy.ndimage import gaussian_filter1d

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)

# ``compute_weights`` tuning constants (strategy="histogram").
_HISTOGRAM_N_BINS = 50  # number of bins for the histogram-based density estimate
_HISTOGRAM_SMOOTHING_SIGMA = 2  # gaussian_filter1d sigma for smoothing the histogram


# ---------------------------------------------------------------------------
# Zarr I/O helpers
# ---------------------------------------------------------------------------


def normalize_day_ranges(data_range) -> List[List[int]]:
    """Normalize a day-range specification to a list of ``[start, end]`` ranges.

    Accepts either a single range ``[start, end]`` or a list of ranges
    ``[[s1, e1], [s2, e2], ...]`` (all 1-indexed, inclusive).  Ranges must be
    in ascending order and non-overlapping.

    Args:
        data_range: Day-range specification.

    Returns:
        List of ``[start, end]`` ranges.

    Raises:
        ValueError: If a range is malformed, out of order, or overlapping.
    """
    if len(data_range) == 0:
        raise ValueError("Empty day-range specification.")

    if hasattr(data_range[0], "__len__"):
        raw_ranges = [list(r) for r in data_range]
    else:
        raw_ranges = [list(data_range)]

    ranges: List[List[int]] = []
    for r in raw_ranges:
        if len(r) != 2:
            raise ValueError(f"Invalid day range {r}: expected [start_day, end_day].")
        start, end = int(r[0]), int(r[1])
        if start < 1 or end < start:
            raise ValueError(f"Invalid day range [{start}, {end}]: expected 1 <= start <= end.")
        ranges.append([start, end])

    for prev, nxt in zip(ranges, ranges[1:]):
        if nxt[0] <= prev[1]:
            raise ValueError(f"Day ranges must be ascending and non-overlapping: {prev} followed by {nxt}.")

    return ranges


def compute_valid_sequence_starts(range_lengths: List[int], seq_len: int) -> np.ndarray:
    """Compute sequence window start indices that do not straddle range gaps.

    Args:
        range_lengths: Number of time steps per contiguous range, in the
            order they are concatenated along the time axis.
        seq_len: Window length in time steps.

    Returns:
        Array of valid start indices into the concatenated time axis.

    Raises:
        ValueError: If any range is shorter than *seq_len*.
    """
    starts: List[int] = []
    offset = 0
    for i, length in enumerate(range_lengths):
        if length < seq_len:
            raise ValueError(
                f"Day range {i} has only {length} time steps after preprocessing; seq_len={seq_len} does not fit."
            )
        starts.extend(range(offset, offset + length - seq_len + 1))
        offset += length
    return np.asarray(starts, dtype=np.int64)


def open_and_slice_dataset(
    data_dir: Path, data_range: List[int], chunks: Optional[dict] = None
) -> Tuple[xr.Dataset, int]:
    """Open a Zarr dataset and slice to the requested day range.

    Days are calendar days of the year of the store's first time stamp, not
    positions in the store: day 1 is January 1 and day 5 is January 5,
    regardless of which dates the store actually covers.  Both ends of the
    range are inclusive: ``[1, 2]`` selects January 1 and 2 (i.e.
    ``2 * samples_per_day`` time steps if both days are fully present).
    Days missing from the store are simply not returned, so a range without
    any data yields a dataset with zero time steps.

    Args:
        data_dir: Path to the Zarr store.
        data_range: Day range as ``[start_day, end_day]`` (1-indexed day of
            year, inclusive).
        chunks: If not ``None``, open the store dask-backed with these chunks
            (``{}`` uses the native zarr chunking). Keeps the whole
            preprocessing/scaling pipeline lazy so that only the slices that
            are actually read are ever materialized. If ``None`` (default) the
            store is opened eagerly as before.

    Returns:
        Tuple of ``(dataset, samples_per_day)``.
    """
    ds = xr.open_dataset(data_dir, engine="zarr", chunks=chunks)
    time_delta = ds.time.values[1] - ds.time.values[0]
    samples_per_day = int(pd.Timedelta(days=1) / time_delta)
    log.info(f"Dataset has {samples_per_day} samples per day.")
    # Half-open calendar window [start_time, end_time), located on the time
    # axis by value so that stores not starting on January 1 select correctly.
    time_index = ds.indexes["time"]
    year_start = pd.Timestamp(year=time_index[0].year, month=1, day=1)
    start_time = year_start + pd.Timedelta(days=data_range[0] - 1)
    end_time = year_start + pd.Timedelta(days=data_range[1])
    start_idx = time_index.searchsorted(start_time, side="left")
    end_idx = time_index.searchsorted(end_time, side="left")
    ds = ds.isel(time=slice(start_idx, end_idx))
    if ds.sizes["time"] == 0:
        log.warning(
            f"Day range {list(data_range)} ({start_time} to {end_time}) contains no data; "
            f"store covers {time_index[0]} to {time_index[-1]}."
        )
    else:
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


def exclude_boundary_nodes(ds: xr.Dataset, boundary_nodes_csv: Optional[str]) -> xr.Dataset:
    """Exclude open-sea boundary nodes from the dataset.

    Args:
        ds: xarray Dataset to filter.
        boundary_nodes_csv: Path to CSV with a ``nSCHISM_hgrid_node`` column
            listing nodes to exclude, or ``None``.

    Returns:
        Dataset with boundary nodes removed (or the original if *boundary_nodes_csv* is ``None``).
    """
    if boundary_nodes_csv is not None:
        boundary_ids = load_node_indices(boundary_nodes_csv)
        all_ids = ds.nSCHISM_hgrid_node.values
        keep_ids = all_ids[~np.isin(all_ids, boundary_ids)]
        n_removed = len(all_ids) - len(keep_ids)
        ds = ds.sel(nSCHISM_hgrid_node=keep_ids)
        log.info(f"Excluded {n_removed} boundary nodes; {len(keep_ids)} nodes remaining.")
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


def compute_sincos(dataset: xr.Dataset, variables: List[str]) -> xr.Dataset:
    """Compute ``sin_var`` and ``cos_var`` for variables prefixed with ``sin_`` or ``cos_``.

    Interprets the base variable as an angle in **degrees** and applies
    ``np.sin`` / ``np.cos`` after converting to radians.  This is the
    recommended encoding for circular quantities such as ``meanWaveDirection``
    to avoid the discontinuity at 0°/360°.

    Example for ``meanWaveDirection``::

        ds["sin_meanWaveDirection"] = sin(radians(ds["meanWaveDirection"]))
        ds["cos_meanWaveDirection"] = cos(radians(ds["meanWaveDirection"]))

    Call this **before** :func:`compute_prevT` so that prefixed variables like
    ``prev_sin_meanWaveDirection`` are created automatically by time-shifting.

    Args:
        dataset: xarray Dataset containing the data.
        variables: List of variable names.  Only names starting with ``sin_``
            or ``cos_`` are processed; all others are ignored.

    Returns:
        Dataset with sin/cos variables added (unscaled, in [-1, 1]).
    """
    for var in variables:
        if var.startswith("sin_"):
            base_var = var[len("sin_") :]
            dataset[var] = np.sin(np.radians(dataset[base_var]))
            log.info(f"Added variable {var} = sin(radians({base_var}))")
        elif var.startswith("cos_"):
            base_var = var[len("cos_") :]
            dataset[var] = np.cos(np.radians(dataset[base_var]))
            log.info(f"Added variable {var} = cos(radians({base_var}))")
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
    derive_vars: Optional[List[str]] = None,
) -> Tuple[xr.Dataset, int]:
    """Compute derived variables, select relevant ones, and drop NaN timestep.

    Args:
        ds: xarray Dataset.
        features: Feature variable names.
        targets: Target variable names.
        extra_keep_vars: Additional variable names to keep in the dataset
            (e.g. graph topology variables). Defaults to ``None``.
        derive_vars: Additional variables to derive and keep (e.g. the
            sample-weighting variable). ``delta_<var>`` entries are computed
            here — i.e. per contiguous day range, so their first-step NaN is
            dropped below and time differences never span range gaps. Plain
            entries already in the dataset are simply kept.

    Returns:
        Tuple of ``(dataset, n_nodes)``.
    """
    ds = compute_sincos(ds, features + targets)
    ds = compute_prevT(ds, features)
    ds = compute_deltaT(ds, targets)
    selected_targets = targets + [tt.replace("delta_", "") for tt in targets]
    keep = features + selected_targets
    # Keep the original angle variable for any sin/cos target pair so that
    # circular metrics can be computed against the true angle at eval time.
    for t in targets:
        for prefix in ("sin_", "cos_"):
            if t.startswith(prefix):
                base_var = t[len(prefix) :]
                if base_var in ds and base_var not in keep:
                    keep.append(base_var)
    if derive_vars:
        for var in derive_vars:
            if var in keep:
                continue
            if var not in ds and var.startswith("delta_"):
                base_var = var[len("delta_") :]
                if base_var in ds:
                    ds = compute_deltaT(ds, [var])
            if var in ds:
                keep.append(var)
    if extra_keep_vars:
        keep = keep + list(extra_keep_vars)
    ds = select_variables(ds, keep)
    ds = remove_first_timestep_if_nan(ds)
    n_nodes = len(ds.nSCHISM_hgrid_node)
    return ds, n_nodes


def open_and_preprocess_dataset(
    data_dir: Path,
    data_range,
    features: List[str],
    targets: List[str],
    node_idx_csv: Optional[str] = None,
    extra_keep_vars: Optional[List[str]] = None,
    derive_vars: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
) -> Tuple[xr.Dataset, int, int, List[int]]:
    """Open a Zarr store and slice + preprocess one or more day ranges.

    Each contiguous range is preprocessed independently — time-shifted
    variables (``prev_*``, ``delta_*``) never leak across the gap between
    ranges, and each range loses its own first time step — then the ranges
    are concatenated along the time axis.

    Args:
        data_dir: Path to the Zarr store.
        data_range: A single ``[start_day, end_day]`` range or a list of
            such ranges (1-indexed, inclusive, ascending, non-overlapping).
        features: Feature variable names.
        targets: Target variable names.
        node_idx_csv: Optional CSV with node indices to keep.
        extra_keep_vars: Additional variables to keep (e.g. graph topology).
        derive_vars: Additional variables to derive per range and keep
            (see :func:`preprocess_variables`).
        chunks: If not ``None``, open the store dask-backed (lazy) with these
            chunks (``{}`` = native zarr chunking). Preprocessing/scaling then
            stay lazy and only the slices that are read are materialized.

    Returns:
        Tuple ``(dataset, samples_per_day, n_nodes, range_lengths)`` where
        *range_lengths* holds the post-preprocessing number of time steps per
        range — sequence datasets need it so windows do not straddle gaps.
    """
    ranges = normalize_day_ranges(data_range)

    processed: List[xr.Dataset] = []
    lengths: List[int] = []
    samples_per_day = n_nodes = 0
    for r in ranges:
        ds_r, samples_per_day = open_and_slice_dataset(data_dir, r, chunks=chunks)
        ds_r = filter_nodes(ds_r, node_idx_csv)
        ds_r, n_nodes = preprocess_variables(
            ds_r, features, targets, extra_keep_vars=extra_keep_vars, derive_vars=derive_vars
        )
        processed.append(ds_r)
        lengths.append(ds_r.sizes["time"])

    if len(processed) == 1:
        ds = processed[0]
    else:
        # data_vars/coords "minimal" so static variables (mesh coordinates,
        # edge topology without a time dim) are not broadcast along time.
        ds = xr.concat(processed, dim="time", data_vars="minimal", coords="minimal", compat="override")
        log.info(f"Concatenated {len(processed)} day ranges with {lengths} time steps each.")

    return ds, samples_per_day, n_nodes, lengths


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


def build_feedback_map(
    features: List[str],
    targets: List[str],
    scaling_params: Optional[Dict[str, Dict[str, float]]],
) -> List[Tuple[int, int, float, float]]:
    """Map ``prev_<target>`` features to the prediction channels that feed them.

    Used to close the feedback loop for rollout evaluation (``src/rollout.py``):
    a prediction at step *t* is written back as the ``prev_<target>`` feature
    for step *t+1*.

    Args:
        features: Feature names in model input order.
        targets: Target names in model output order.
        scaling_params: ``{var: {mean, std}}`` used to re-scale physical
            predictions into feature space, or ``None``.

    Returns:
        List of ``(feature_idx, target_idx, mean, std)`` tuples.  Predictions
        are written back as ``(pred - mean) / std``.
    """
    feedback = []
    for f_idx, feat in enumerate(features):
        if not feat.startswith("prev_"):
            continue
        base = feat[len("prev_") :]
        if base not in targets:
            log.warning(f"Feature '{feat}' has no matching target '{base}' — stays at ground truth during rollout.")
            continue
        t_idx = targets.index(base)
        if scaling_params and feat in scaling_params:
            mean = float(scaling_params[feat]["mean"])
            std = float(scaling_params[feat]["std"])
        else:
            mean, std = 0.0, 1.0
        feedback.append((f_idx, t_idx, mean, std))
        log.info(f"Feedback channel: prediction '{base}' -> feature '{feat}' (mean={mean:.4f}, std={std:.4f})")
    return feedback


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


def retrieve_weight_data(dataset: xr.Dataset, weight_variable: str) -> torch.Tensor:
    """Extract the sample-weighting variable as a flat tensor.

    The variable is looked up in the dataset directly.  If it is absent but
    has the form ``delta_<var>`` and ``<var>`` is present, the time difference
    is derived on the fly (its first time step is NaN; :func:`compute_weights`
    assigns those samples zero weight).

    Args:
        dataset: xarray Dataset (after preprocessing).
        weight_variable: Name of the weighting variable, e.g. ``sigWaveHeight``
            or ``delta_sigWaveHeight``.

    Returns:
        Tensor of shape ``(n_samples,)`` with the same time-major sample
        ordering as :func:`retrieve_data`.

    Raises:
        ValueError: If the variable is not in the dataset and cannot be derived.
    """
    if weight_variable not in dataset:
        base_var = weight_variable[len("delta_") :] if weight_variable.startswith("delta_") else None
        if base_var is None or base_var not in dataset:
            raise ValueError(
                f"weight_variable '{weight_variable}' not found in dataset and cannot be "
                f"derived. Available variables: {list(dataset.data_vars)}"
            )
        log.info(f"Deriving weight variable '{weight_variable}' from '{base_var}' on the fly.")
        dataset = compute_deltaT(dataset.copy(), [weight_variable])

    return retrieve_data(dataset, [weight_variable])[:, 0]


def compute_weights(
    data: torch.Tensor,
    strategy: str = "z_score",
    power: float = 1.0,
) -> Optional[torch.Tensor]:
    """Compute and normalise per-sample weights from a single reference variable.

    A single scalar weight is computed per sample based on the z-score of
    *data* (typically obtained via :func:`retrieve_weight_data`).  Weights are
    normalised so that their mean equals 1.0 and returned with shape
    ``(n_samples, 1)`` so they broadcast uniformly over all targets in the
    loss.  NaN entries (e.g. the first time step of an on-the-fly delta
    variable) receive zero weight.

    Args:
        data: Weighting variable values, shape ``(n_samples,)`` or ``(n_samples, 1)``.
        strategy: ``"z_score"`` (legacy; kept for backward compatibility with older
            configs/checkpoints, emphasises samples far from the mean) or ``"histogram"``
            (preferred: reweights by inverse local density, so rare value ranges get more
            weight regardless of where they sit relative to the mean).
        power: Exponent controlling weighting intensity. For ``"histogram"``, lower values
            move weights toward uniform (unweighted); it does not reproduce ``"z_score"``
            behavior, which is a distinct (tail-emphasis) weighting scheme.

    Returns:
        Weight tensor of shape ``(n_samples, 1)``, or ``None``.

    Raises:
        ValueError: If *strategy* is not recognised.
    """

    log.info(f"Computing sample weights with strategy '{strategy}'...")

    data = data.reshape(-1)
    n_samples = len(data)

    if strategy == "z_score":
        # Legacy strategy, kept for backward compatibility. Prefer "histogram".
        valid = torch.isfinite(data)
        if not bool(valid.all()):
            log.info(f"Weight data contains {int((~valid).sum())} non-finite entries; they get zero weight.")
        mean = data[valid].mean()
        std = data[valid].std()
        z_scores = torch.zeros_like(data)
        z_scores[valid] = (data[valid] - mean) / std
        weights = 1.0 - torch.exp(-(z_scores**2) / power)
    elif strategy == "histogram":
        n_bins = _HISTOGRAM_N_BINS
        hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
        hist_smooth = gaussian_filter1d(hist, sigma=_HISTOGRAM_SMOOTHING_SIGMA)

        eps = 1e-6  # guard against empty bins
        bin_idx = np.clip(np.digitize(data, bin_edges[1:-1]), 0, n_bins - 1)
        density = hist_smooth[bin_idx] + eps
        weights = torch.from_numpy(density ** (-power))
    else:
        raise ValueError(
            f"Unknown sample_weight_strategy: '{strategy}'. Supported: None, 'z_score' (legacy), 'histogram'"
        )

    # Normalise so mean == 1.0
    weight_sum = weights.sum()
    weights = weights * (n_samples / weight_sum)
    log.info(
        f"Sample weights normalised: mean={weights.mean():.4f}, std={weights.std():.4f}, "
        f"min={weights.min():.4f}, max={weights.max():.4f}"
    )

    # Return shape (n_samples, 1) so it broadcasts over all targets in the loss
    return weights.float().unsqueeze(-1)


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


def reconstruct_from_sincos(
    sin_predictions: np.ndarray,
    cos_predictions: np.ndarray,
) -> np.ndarray:
    """Reconstruct an angle in degrees from sin/cos component predictions.

    Uses ``np.degrees(np.arctan2(sin, cos)) % 360`` to map back to [0, 360).

    Args:
        sin_predictions: Predicted sin component (any shape).
        cos_predictions: Predicted cos component (same shape).

    Returns:
        Angle in degrees, in the range [0, 360).
    """
    return np.degrees(np.arctan2(sin_predictions, cos_predictions)) % 360.0


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

    # Collect raw (possibly unscaled) predictions per target name first,
    # so that sin/cos pairs can be detected and reconstructed together.
    raw_preds: Dict[str, np.ndarray] = {}

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
            raw_preds[target_name] = target_preds

    # Reconstruct angle (degrees) for any sin/cos target pairs present.
    # E.g. if both "sin_meanWaveDirection" and "cos_meanWaveDirection" are in
    # target_names, reconstruct "meanWaveDirection" via atan2.
    sin_targets = {n[len("sin_") :]: n for n in raw_preds if n.startswith("sin_")}
    cos_targets = {n[len("cos_") :]: n for n in raw_preds if n.startswith("cos_")}
    for base_var in sin_targets:
        if base_var in cos_targets:
            sin_name = sin_targets[base_var]
            cos_name = cos_targets[base_var]
            log.info(f"Reconstructing {base_var} from {sin_name} and {cos_name} via atan2")
            reconstructed_angle = reconstruct_from_sincos(raw_preds[sin_name], raw_preds[cos_name])
            data_vars[base_var] = (["time", "nSCHISM_hgrid_node"], reconstructed_angle)

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
