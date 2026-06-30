# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def open_experiments(
    experiments_csv: str,
    ground_truth: str = "/work/ka1176/shared_data/2025-wwm-emulator-schism/processed_RUN04b_JUWELS_wpile.zarr",
    test_days=[335, 365],
) -> xr.Dataset:
    """
    Open all experiments from a csv table. The table must contain the column "predictions" with absolute paths to
    generated netcdf files.

    Args:
        experiments_csv (str): path to csv with experiment table
        ground_truth (str): zarr directory with ground truth
        test_days: test set days (defaults to 335 - 365).

    Returns: Dataset with all predictions concatenated along a new dimension (experiment).

    """

    ds_true = xr.open_zarr(ground_truth).isel(time=slice(test_days[0], test_days[1] + 1))
    experiments = pd.read_csv(experiments_csv, skipinitialspace=True)

    # construct a key exp_name from {backbone}-{seq_len}-{graph nodes}
    exp_names = [
        f"{row['backbone']}-{row['sequence length']}-{row['graph nodes']}" for _, row in experiments.iterrows()
    ]

    # open predictions for rows that have a valid path
    datasets = []
    valid_exp_names = []
    for exp_name, (_, row) in zip(exp_names, experiments.iterrows()):
        pred_path = row["predictions"]
        if pd.isna(pred_path) or not str(pred_path).strip():
            continue
        ds_pred = xr.open_dataset(str(pred_path).strip(), engine="netcdf4")
        datasets.append(ds_pred)
        valid_exp_names.append(exp_name)

    # some predictions do not cover the full length (because of sequence length), cut the whole set accordingly
    min_time = min(ds.sizes["time"] for ds in datasets)
    datasets = [ds.isel(time=slice(None, min_time)) for ds in datasets]

    # keep only the nodes (nSCHISM_hgrid_node) that are present in all predictions
    common_nodes = set(datasets[0]["nSCHISM_hgrid_node"].values)
    for ds in datasets[1:]:
        common_nodes &= set(ds["nSCHISM_hgrid_node"].values)
    common_nodes = sorted(common_nodes)
    datasets = [ds.sel(nSCHISM_hgrid_node=common_nodes) for ds in datasets]

    # concatenate everything along a new "experiment" dimension, identifiable by experiment names
    ds = xr.concat(datasets, dim=pd.Index(valid_exp_names, name="experiment"))

    # align ground truth: same time length, same nodes, same variables as predictions
    pred_vars = list(ds.data_vars)
    pred_times = ds.time.values
    ds_true = (
        ds_true.sel(time=pred_times)
        .sel(nSCHISM_hgrid_node=common_nodes)[pred_vars]
        .expand_dims(experiment=["ground_truth"])
    )
    ds = xr.concat([ds, ds_true], dim="experiment")

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

    bin_labels = pd.cut(gt_flat, bins=bins)

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
