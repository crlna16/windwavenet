# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import Optional

import re

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..data import dataset_utils as du


def _reconstruct_wave_direction(ds: xr.Dataset) -> xr.Dataset:
    """Reconstruct meanWaveDirection from its sin/cos encoding, if present, and drop sin/cos.

    Args:
        ds: Dataset that may contain 'sin_meanWaveDirection' and 'cos_meanWaveDirection'.

    Returns:
        Dataset with 'meanWaveDirection' added and the sin/cos variables removed. Returned
        unchanged if either sin/cos variable is missing.
    """
    if "sin_meanWaveDirection" in ds.data_vars and "cos_meanWaveDirection" in ds.data_vars:
        ds["meanWaveDirection"] = (
            np.degrees(np.arctan2(ds["sin_meanWaveDirection"], ds["cos_meanWaveDirection"])) % 360.0
        )
        ds = ds.drop_vars(["sin_meanWaveDirection", "cos_meanWaveDirection"])
    return ds


def open_experiments(
    experiments_csv: str,
    ground_truth: str = "/work/ka1176/shared_data/2025-wwm-emulator-schism/processed_RUN04b_JUWELS_wpile.zarr",
    boundary_nodes_csv: Optional[str] = None,
    chunks="auto",
) -> xr.Dataset:
    """
    Open all experiments from a csv table. The table must contain the column "predictions" with absolute paths to
    generated netcdf files.

    All datasets are opened lazily as dask-backed arrays (rather than eagerly loaded, cached numpy
    arrays), so no prediction/ground-truth data is actually read from disk until it is needed
    (e.g. via `.values` or `.compute()` in downstream functions such as `compute_binned_scores`).

    Args:
        experiments_csv (str): path to csv with experiment table
        ground_truth (str): zarr directory with ground truth
        boundary_nodes_csv: Optional CSV file with open-sea boundary nodes to exclude.
        chunks: Dask chunking passed to `xr.open_zarr`/`xr.open_dataset` to keep data lazy. Set to
            `None` to disable dask and load eagerly instead.

    Returns: Dataset with all predictions concatenated along a new dimension (experiment). Data
        variables remain dask-backed and lazy; nothing is loaded into memory until computed.

    """

    ds_true = xr.open_zarr(ground_truth, chunks=chunks)

    # reconstruct meanWaveDirection from its sin/cos encoding, if present, and discard the sin/cos
    # variables so downstream code only ever sees the plain angle variable
    ds_true = _reconstruct_wave_direction(ds_true)

    experiments = pd.read_csv(experiments_csv, skipinitialspace=True, comment="#")

    # construct a key exp_name from {backbone}-{seq_len}-{graph nodes}
    exp_names = [
        f"{row['backbone']}-{row['sequence length']:.0f}-{row['graph nodes']:.0f}" for _, row in experiments.iterrows()
    ]
    exp_names = [ep.replace("-nan", "") for ep in exp_names]

    # open predictions for rows that have a valid path
    datasets = []
    valid_exp_names = []
    for exp_name, (_, row) in zip(exp_names, experiments.iterrows()):
        pred_path = row["predictions"]
        if pd.isna(pred_path) or not str(pred_path).strip():
            continue
        ds_pred = xr.open_dataset(str(pred_path).strip(), engine="netcdf4", chunks=chunks)
        datasets.append(ds_pred)
        valid_exp_names.append(exp_name)
        print(f"Added {exp_name} from {pred_path} ({ds_pred.time.values[0]} - {ds_pred.time.values[-1]})")

    # keep only the time steps that are present in all datasets
    common_times = set(datasets[0]["time"].values)
    for ds in datasets[1:]:
        common_times &= set(ds["time"].values)
    common_times = sorted(common_times)
    datasets = [ds.sel(time=common_times) for ds in datasets]

    # keep only the nodes (nSCHISM_hgrid_node) that are present in all predictions
    common_nodes = set(datasets[0]["nSCHISM_hgrid_node"].values)
    for ds in datasets[1:]:
        common_nodes &= set(ds["nSCHISM_hgrid_node"].values)
    common_nodes = sorted(common_nodes)
    datasets = [ds.sel(nSCHISM_hgrid_node=common_nodes) for ds in datasets]

    # concatenate everything along a new "experiment" dimension, identifiable by experiment names
    ds = xr.concat(datasets, dim=pd.Index(valid_exp_names, name="experiment"))

    # reconstruct meanWaveDirection from its sin/cos encoding, if predictions were made in that
    # encoding, so that pred_vars below lines up with ds_true (which also no longer has sin/cos)
    ds = _reconstruct_wave_direction(ds)

    # drop all coordinates except the spatial and index ones
    keep_coords = {"SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y", "time", "experiment"}
    coords_to_drop = [c for c in ds_true.coords if c not in keep_coords]
    ds_true = ds_true.drop_vars(coords_to_drop)
    coords_to_drop = [c for c in ds.coords if c not in keep_coords]
    ds = ds.drop_vars(coords_to_drop)

    # align ground truth: same time length, same nodes, same variables as predictions
    pred_vars = list(ds.data_vars)
    pred_times = ds.time.values
    ds_true = (
        ds_true.sel(time=pred_times)
        .sel(nSCHISM_hgrid_node=common_nodes)[pred_vars]
        .expand_dims(experiment=["ground_truth"])
    )
    ds = xr.concat([ds, ds_true], dim="experiment")

    # --- Remove open-sea boundary nodes if CSV is provided ---
    print(f"Dropping boundary nodes from {boundary_nodes_csv}")
    ds = du.exclude_boundary_nodes(ds, boundary_nodes_csv)

    return ds


def compute_binned_scores(
    ds: xr.Dataset,
    variable: str,
    bins: list,
) -> pd.DataFrame:
    """
    Compute RMSE and bias of each experiment vs ground truth, overall and per bin of the variable.

    Args:
        ds: Dataset from open_experiments with 'experiment' dimension.
        variable: Variable name to evaluate.
        bins: Bin edges applied to the ground truth values.

    Returns:
        DataFrame with columns [experiment, bin, rmse, bias].
        'bin' is a pd.Interval for binned rows or the string 'overall'.
    """
    gt = ds[variable].sel(experiment="ground_truth")  # (time, nSCHISM_hgrid_node)
    gt_flat = gt.values.ravel()

    if np.isnan(gt_flat).any():
        raise ValueError(f"Ground truth contains NaN values for variable '{variable}'.")

    bin_labels = pd.cut(pd.Series(gt_flat), bins=bins)

    experiments = [str(e) for e in ds.experiment.values if e != "ground_truth"]
    records = []

    for exp in experiments:
        pred_flat = ds[variable].sel(experiment=exp).values.ravel()

        if np.isnan(pred_flat).any():
            raise ValueError(f"Predictions for experiment '{exp}' contain NaN values for variable '{variable}'.")

        error = pred_flat - gt_flat

        # overall scores
        records.append(
            {
                "experiment": exp,
                "bin": "overall",
                "rmse": np.sqrt(np.mean(error**2)),
                "bias": np.mean(error),
            }
        )

        # per-bin scores
        for interval in bin_labels.cat.categories:
            mask = (bin_labels == interval).values
            if mask.sum() == 0:
                records.append({"experiment": exp, "bin": interval, "rmse": np.nan, "bias": np.nan})
            else:
                records.append(
                    {
                        "experiment": exp,
                        "bin": interval,
                        "rmse": np.sqrt(np.mean(error[mask] ** 2)),
                        "bias": np.mean(error[mask]),
                    }
                )

    return pd.DataFrame(records)


def compute_binned_change_scores(
    ds: xr.Dataset,
    variable: str,
    bins: list,
) -> pd.DataFrame:
    """
    Compute RMSE and bias of the timestep-to-timestep change in a variable for each experiment
    vs ground truth, overall and per bin of the ground-truth change.

    Args:
        ds: Dataset from open_experiments with 'experiment' dimension.
        variable: Variable name to evaluate.
        bins: Bin edges applied to the ground truth change values (variable diffed along time).

    Returns:
        DataFrame with columns [experiment, bin, rmse, bias].
        'bin' is a pd.Interval for binned rows or the string 'overall'.
    """
    gt_change = ds[variable].sel(experiment="ground_truth").diff(dim="time")  # (time - 1, nSCHISM_hgrid_node)
    gt_change_flat = gt_change.values.ravel()

    if np.isnan(gt_change_flat).any():
        raise ValueError(f"Ground truth change contains NaN values for variable '{variable}'.")

    bin_labels = pd.cut(pd.Series(gt_change_flat), bins=bins)

    experiments = [str(e) for e in ds.experiment.values if e != "ground_truth"]
    records = []

    for exp in experiments:
        pred_change_flat = ds[variable].sel(experiment=exp).diff(dim="time").values.ravel()

        if np.isnan(pred_change_flat).any():
            raise ValueError(f"Predictions for experiment '{exp}' contain NaN values for variable '{variable}'.")

        error = pred_change_flat - gt_change_flat

        # overall scores
        records.append(
            {
                "experiment": exp,
                "bin": "overall",
                "rmse": np.sqrt(np.mean(error**2)),
                "bias": np.mean(error),
            }
        )

        # per-bin scores
        for interval in bin_labels.cat.categories:
            mask = (bin_labels == interval).values
            if mask.sum() == 0:
                records.append({"experiment": exp, "bin": interval, "rmse": np.nan, "bias": np.nan})
            else:
                records.append(
                    {
                        "experiment": exp,
                        "bin": interval,
                        "rmse": np.sqrt(np.mean(error[mask] ** 2)),
                        "bias": np.mean(error[mask]),
                    }
                )

    return pd.DataFrame(records)


def plot_overall_scores(
    df: pd.DataFrame,
    metric: str = "rmse",
    fig: plt.Figure = None,
) -> plt.Figure:
    """
    Bar plot of the overall RMSE or bias across all experiments.

    Args:
        df: DataFrame returned by compute_binned_scores.
        metric: Either 'rmse' or 'bias'.
        fig: Optional existing Figure to draw on. A new figure is created if not provided.

    Returns:
        The matplotlib Figure with the plot.
    """
    if metric not in ("rmse", "bias"):
        raise ValueError(f"metric must be 'rmse' or 'bias', got '{metric}'.")

    overall = df[df["bin"] == "overall"][["experiment", metric]].copy()

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    sns.barplot(data=overall, x="experiment", y=metric, ax=ax)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    return fig


def plot_architecture_scores(
    df: pd.DataFrame,
    architecture: str,
    metric: str = "rmse",
    fig: plt.Figure = None,
) -> plt.Figure:
    """
    Plot overall RMSE or bias for a single architecture, varying over sequence length
    and/or graph nodes.

    The experiment name format is '{backbone}-{seq_len}-{graph_nodes}'.  The
    function detects which of the two hyperparameters actually vary and selects
    the appropriate plot type:

    * Only one varies  → line plot with that parameter on the x-axis.
    * Both vary        → 2-D heatmap (seq_len × graph_nodes) with the metric
                         as the colour map.

    Args:
        df: DataFrame returned by compute_binned_scores.
        architecture: Backbone name, e.g. 'lstm' or 'lstm-gat-vn'.
        metric: Either 'rmse' or 'bias'.
        fig: Optional existing Figure to draw on.

    Returns:
        The matplotlib Figure with the plot.
    """
    if metric not in ("rmse", "bias"):
        raise ValueError(f"metric must be 'rmse' or 'bias', got '{metric}'.")

    # filter to overall rows for the requested architecture
    # architecture name may itself contain "-"; match trailing int or int-int suffix
    overall = df[df["bin"] == "overall"].copy()
    arch_re = re.compile(rf"^{re.escape(architecture)}(?:-(\d+)(?:-(\d+))?)?$")

    def _parse_exp(exp: str):
        m = arch_re.match(str(exp))
        if not m:
            return None
        seq = int(m.group(1)) if m.group(1) is not None else np.nan
        nodes = int(m.group(2)) if m.group(2) is not None else np.nan
        return seq, nodes

    parsed = overall["experiment"].apply(_parse_exp)
    overall = overall[parsed.notna()].copy()

    if overall.empty:
        raise ValueError(f"No experiments found for architecture '{architecture}'.")

    overall[["seq_len", "graph_nodes"]] = pd.DataFrame(parsed[parsed.notna()].tolist(), index=overall.index)

    seq_vals = overall["seq_len"].dropna().unique()
    node_vals = overall["graph_nodes"].dropna().unique()
    seq_varies = len(seq_vals) > 1
    nodes_varies = len(node_vals) > 1

    if not seq_varies and not nodes_varies:
        raise ValueError(f"Architecture '{architecture}' has only one experiment; nothing to plot.")

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    if seq_varies and nodes_varies:
        # 2-D heatmap: rows = seq_len, columns = graph_nodes
        pivot = overall.pivot(index="seq_len", columns="graph_nodes", values=metric)
        pivot = pivot.sort_index().sort_index(axis=1)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cbar_kws={"label": metric.upper()})
        ax.set_xlabel("Graph nodes")
        ax.set_ylabel("Sequence length")
        ax.set_title(f"{architecture} — {metric.upper()}")

    elif seq_varies:
        plot_data = overall.sort_values("seq_len")
        sns.lineplot(data=plot_data, x="seq_len", y=metric, marker="o", ax=ax)
        ax.set_xlabel("Sequence length")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{architecture} — {metric.upper()}")

    else:
        plot_data = overall.sort_values("graph_nodes")
        sns.lineplot(data=plot_data, x="graph_nodes", y=metric, marker="o", ax=ax)
        ax.set_xlabel("Graph nodes")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{architecture} — {metric.upper()}")

    fig.tight_layout()
    return fig


def plot_binned_scores(
    df: pd.DataFrame,
    metric: str = "rmse",
    fig: plt.Figure = None,
) -> plt.Figure:
    """
    Line plot of RMSE or bias across target-variable bins, one line per experiment.

    Args:
        df: DataFrame returned by compute_binned_scores.
        metric: Either 'rmse' or 'bias'.
        fig: Optional existing Figure to draw on.

    Returns:
        The matplotlib Figure with the plot.
    """
    if metric not in ("rmse", "bias"):
        raise ValueError(f"metric must be 'rmse' or 'bias', got '{metric}'.")

    binned = df[df["bin"] != "overall"].copy()

    # represent each bin by its midpoint for a numeric x-axis
    binned["bin_mid"] = binned["bin"].apply(lambda iv: iv.mid)
    binned = binned.sort_values("bin_mid")

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    sns.lineplot(
        data=binned,
        x="bin_mid",
        y=metric,
        hue="experiment",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Bin midpoint")
    ax.set_ylabel(metric.upper())
    ax.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    return fig


def plot_spatial_error(
    ds: xr.Dataset,
    experiment,
    variable: str,
    time_indices,
    metric: str = "rmse",
    unit: str = "m",
    resolution: float = 10000,
    x_range: Optional[tuple] = None,
    y_range: Optional[tuple] = None,
    fig: plt.Figure = None,
) -> plt.Figure:
    """
    Spatial heatmap of RMSE or bias for one or more experiments against ground
    truth, interpolated onto a regular grid and drawn on a Cartopy map.

    When multiple experiments are provided all subplots share a common
    vmin/vmax.  Subplots are arranged in rows of at most 3 columns.

    Args:
        ds: Dataset returned by open_experiments.
        experiment: A single experiment key or a list of experiment keys
            (must be values on the 'experiment' dimension, not 'ground_truth').
        variable: Variable name to evaluate.
        time_indices: Either a single integer (one time step) or a
            ``(start, end)`` tuple selecting ``ds.isel(time=slice(start, end))``.
            The metric is aggregated over all selected time steps.
        metric: 'rmse' or 'bias'.
        unit: Optional unit string appended to the colorbar label and title,
            default ``'m'``.
        resolution: Bin length in the same units as the node coordinates.
            The number of grid points along each axis is derived as
            ``ceil((max - min) / resolution)``.  Default is 10 000.
        x_range: Optional (x_min, x_max) tuple to restrict the plotted domain
            (in the same units as SCHISM_hgrid_node_x).
        y_range: Optional (y_min, y_max) tuple to restrict the plotted domain
            (in the same units as SCHISM_hgrid_node_y).
        fig: Optional existing Figure to draw on (only used for a single
            experiment).  A new figure is always created for multiple
            experiments.

    Returns:
        The matplotlib Figure with the plot(s).
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError("cartopy is required for plot_spatial_error.") from exc
    try:
        from scipy.interpolate import griddata
    except ImportError as exc:
        raise ImportError("scipy is required for plot_spatial_error.") from exc

    if metric not in ("rmse", "bias"):
        raise ValueError(f"metric must be 'rmse' or 'bias', got '{metric}'.")

    # Normalise inputs to lists.
    if isinstance(experiment, str):
        experiments = [experiment]
    else:
        experiments = list(experiment)

    # Normalise time_indices: scalar → single step, (start, end) → range.
    if np.isscalar(time_indices):
        t_slice = [int(time_indices)]
        t0_label = str(ds.time.isel(time=int(time_indices)).values)
        t1_label = None
    else:
        t0, t1 = int(time_indices[0]), int(time_indices[1])
        t_slice = slice(t0, t1)
        t0_label = str(ds.time.isel(time=t0).values)
        t1_label = str(ds.time.isel(time=t1 - 1).values)

    # --- Node positions (shared across experiments) --------------------------
    node_x_da = ds["SCHISM_hgrid_node_x"]
    if "experiment" in node_x_da.dims:
        node_x_da = node_x_da.isel(experiment=0)
    node_x = node_x_da.values

    node_y_da = ds["SCHISM_hgrid_node_y"]
    if "experiment" in node_y_da.dims:
        node_y_da = node_y_da.isel(experiment=0)
    node_y = node_y_da.values

    # Optional sub-domain mask (same for all experiments).
    mask = np.ones(len(node_x), dtype=bool)
    if x_range is not None:
        mask &= (node_x >= x_range[0]) & (node_x <= x_range[1])
    if y_range is not None:
        mask &= (node_y >= y_range[0]) & (node_y <= y_range[1])

    node_x = node_x[mask]
    node_y = node_y[mask]

    # Regular grid (shared across experiments).
    nx = max(2, int(np.ceil((node_x.max() - node_x.min()) / resolution)))
    ny = max(2, int(np.ceil((node_y.max() - node_y.min()) / resolution)))
    xi = np.linspace(node_x.min(), node_x.max(), nx)
    yi = np.linspace(node_y.min(), node_y.max(), ny)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    gt = ds[variable].sel(experiment="ground_truth").isel(time=t_slice).values  # (n_time, n_nodes)

    # --- Compute per-experiment interpolated grids ---------------------------
    grids = []
    for exp in experiments:
        pred = ds[variable].sel(experiment=exp).isel(time=t_slice).values
        error = pred - gt

        if metric == "rmse":
            node_error = np.sqrt(np.nanmean(error**2, axis=0))[mask]
        else:
            node_error = np.nanmean(error, axis=0)[mask]

        zi = griddata((node_x, node_y), node_error, (xi_grid, yi_grid), method="linear")
        grids.append(zi)

    # --- Common color range --------------------------------------------------
    if metric == "rmse":
        cmap = "YlOrRd"
        vmin = 0.0
        vmax = float(max(np.nanmax(z) for z in grids))
    else:
        cmap = "RdBu_r"
        max_abs = float(max(np.nanmax(np.abs(z)) for z in grids))
        vmin, vmax = -max_abs, max_abs

    # --- Layout --------------------------------------------------------------
    n_exp = len(experiments)
    ncols = min(n_exp, 3)
    nrows = int(np.ceil(n_exp / ncols))
    projection = ccrs.Orthographic(central_longitude=4.5, central_latitude=53.5)

    if n_exp == 1 and fig is not None:
        axes = [fig.gca()]
        _own_fig = False
    else:
        fig, axes_arr = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols + 1, 6 * nrows),
            subplot_kw={"projection": projection},
            squeeze=False,
        )
        axes = axes_arr.ravel().tolist()
        _own_fig = True
        fig.subplots_adjust(top=0.88, right=0.87, hspace=0.35, wspace=0.1)

    pcm = None
    for ax, exp, zi in zip(axes, experiments, grids):
        pcm = ax.pcolormesh(
            xi,
            yi,
            zi,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.UTM(32),
            zorder=1,
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=3)
        ax.coastlines(zorder=4)
        ax.set_title(exp)

    # Hide unused axes (when n_exp is not a multiple of ncols).
    for ax in axes[n_exp:]:
        ax.set_visible(False)

    # Single shared colorbar in a dedicated strip on the right.
    unit_str = f" ({unit})" if unit else ""
    cbar_label = f"{metric.upper()} ({variable}){unit_str}"
    if pcm is not None:
        if _own_fig:
            cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.65])
        else:
            cbar_ax = None
        fig.colorbar(pcm, cax=cbar_ax, ax=None if cbar_ax else axes[:n_exp], label=cbar_label)

    # Figure-wide title.
    time_str = t0_label if t1_label is None else f"{t0_label} … {t1_label}"
    fig.suptitle(f"{metric.upper()} ({variable}){unit_str}\n{time_str}", y=0.98)

    return fig
