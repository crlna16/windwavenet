# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Precompute 10 km-resolution spatial RMSE + ground-truth maps for ONE experiment.

Unlike compute_experiment_metrics.py, this does not scan all experiments (the
griddata interpolation is too expensive to run for every one) -- pass the
specific experiment subfolder to process via --experiment-name.

For summer and winter separately, and for each of sigWaveHeight, meanWavePeriod
and meanWaveDirection that is actually present in the experiment's prediction
files, computes:
  - the per-node RMSE of the experiment's predictions vs ground truth,
    interpolated onto a regular grid.
  - the ground-truth time-mean, interpolated onto the same grid.
meanWaveDirection is circular (degrees), so its RMSE and mean use wraparound-aware
arithmetic instead of plain subtraction/averaging. sigWaveHeight is required;
meanWavePeriod/meanWaveDirection are simply skipped if absent, so this still
works when only sigWaveHeight is available.

All grids are saved to a single NetCDF file (dims: variable, season, y, x) that
can be loaded and rendered with `src.utils.plot_functions.plot_spatial_grids`
without redoing the interpolation.

Usage: python compute_spatial_metrics.py --experiments-dir DIR --experiment-name NAME
       [--output-dir DIR] [--resolution METERS] [--variables VAR1,VAR2,...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from compute_experiment_metrics import (  # noqa: E402
    CIRCULAR_VARIABLES,
    GROUND_TRUTH_ZARR,
    PREDICT_FINAL_DIR,
    SUMMER_FILE,
    SUMMER_START,
    VARIABLE,
    VARIABLES,
    WINTER_FILE,
    WINTER_START,
    build_period_dataset,
)
from src.utils.plot_functions import _circular_error, compute_spatial_grid  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "spatial_metrics"
RESOLUTION = 10000  # meters

SEASON_FILES = {"summer": (SUMMER_FILE, SUMMER_START), "winter": (WINTER_FILE, WINTER_START)}


def _node_coords(ds: xr.Dataset) -> tuple:
    """Node x/y coordinates for ds's nodes, read from the ground-truth zarr.

    Prediction NetCDF files store node coordinates under a differently-named
    variable (`nSCHISM_hgrid_node_x`/`y`), so this looks them up directly from
    the ground-truth zarr instead of relying on `ds`'s own coordinates.
    """
    ds_coords = xr.open_zarr(GROUND_TRUTH_ZARR, chunks={})[["SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y"]]
    ds_coords = ds_coords.sel(nSCHISM_hgrid_node=ds.nSCHISM_hgrid_node.values)
    return ds_coords["SCHISM_hgrid_node_x"].values, ds_coords["SCHISM_hgrid_node_y"].values


def compute_variable_grids(
    ds: xr.Dataset, node_x: np.ndarray, node_y: np.ndarray, exp_name: str, variable: str, resolution: float
) -> tuple:
    """RMSE grid (`exp_name` vs ground truth) and ground-truth time-mean grid for one variable."""
    gt = ds[variable].sel(experiment="ground_truth").values  # (time, node)
    pred = ds[variable].sel(experiment=exp_name).values

    if variable in CIRCULAR_VARIABLES:
        node_rmse = np.sqrt(np.nanmean(_circular_error(pred, gt) ** 2, axis=0))
        # vector (sin/cos) mean, since a plain average of angles is meaningless across 0/360 wraparound
        gt_rad = np.radians(gt)
        node_gt_mean = np.degrees(
            np.arctan2(np.nanmean(np.sin(gt_rad), axis=0), np.nanmean(np.cos(gt_rad), axis=0))
        ) % 360.0
    else:
        node_rmse = np.sqrt(np.nanmean((pred - gt) ** 2, axis=0))
        node_gt_mean = np.nanmean(gt, axis=0)

    rmse_grid = compute_spatial_grid(node_x, node_y, node_rmse, resolution=resolution)
    gt_grid = compute_spatial_grid(node_x, node_y, node_gt_mean, resolution=resolution)
    return rmse_grid, gt_grid


def compute_season_grids(
    exp_name: str, nc_path: Path, period_start: str, resolution: float, variables: list
) -> tuple:
    """Per-variable (rmse_grid, gt_grid) grids for one season, plus the variables actually available."""
    ds, available = build_period_dataset(exp_name, nc_path, period_start)
    available = [v for v in variables if v in available]
    node_x, node_y = _node_coords(ds)

    grids = {
        variable: compute_variable_grids(ds, node_x, node_y, exp_name, variable, resolution)
        for variable in available
    }
    return grids, available


def main(
    experiments_dir: Path = PREDICT_FINAL_DIR,
    experiment_name: str = None,
    output_dir: Path = OUTPUT_DIR,
    resolution: float = RESOLUTION,
    variables: list = VARIABLES,
):
    exp_dir = experiments_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    season_grids = {}
    season_variables = {}
    for season, (nc_file, period_start) in SEASON_FILES.items():
        nc_path = exp_dir / nc_file
        if not nc_path.exists():
            raise FileNotFoundError(f"Missing '{nc_file}' for experiment '{experiment_name}' in {exp_dir}")

        print(f"Processing '{experiment_name}' ({season}) ...")
        season_grids[season], season_variables[season] = compute_season_grids(
            experiment_name, nc_path, period_start, resolution, variables
        )

    # only keep variables available in both seasons, so the output stays consistent across "season"
    available_variables = [v for v in variables if all(v in season_variables[s] for s in SEASON_FILES)]
    if VARIABLE not in available_variables:
        raise RuntimeError(f"Required variable '{VARIABLE}' not found for experiment '{experiment_name}'")

    variable_datasets = {}
    for variable in available_variables:
        season_datasets = {}
        for season in SEASON_FILES:
            rmse_grid, gt_grid = season_grids[season][variable]
            # rmse_grid/gt_grid share xi/yi: same node coords, same resolution, no x/y_range.
            season_datasets[season] = xr.Dataset(
                {
                    "rmse": (["y", "x"], rmse_grid["zi"]),
                    "ground_truth_mean": (["y", "x"], gt_grid["zi"]),
                },
                coords={"x": rmse_grid["xi"], "y": rmse_grid["yi"]},
            )
        variable_datasets[variable] = xr.concat(
            [season_datasets[s] for s in SEASON_FILES], dim=pd.Index(list(SEASON_FILES), name="season")
        )

    ds_out = xr.concat(
        [variable_datasets[v] for v in available_variables], dim=pd.Index(available_variables, name="variable")
    )
    ds_out.attrs["variables"] = ",".join(available_variables)
    ds_out.attrs["experiment"] = experiment_name
    ds_out.attrs["resolution_m"] = resolution

    output_path = output_dir / f"{experiment_name}_spatial_metrics.nc"
    ds_out.to_netcdf(output_path)
    print(f"Saved spatial metrics for '{experiment_name}' to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=PREDICT_FINAL_DIR)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resolution", type=float, default=RESOLUTION)
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help=f"Comma-separated list of variables to process (default: all of {', '.join(VARIABLES)})",
    )
    args = parser.parse_args()
    main(
        experiments_dir=args.experiments_dir,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        resolution=args.resolution,
        variables=args.variables.split(",") if args.variables else VARIABLES,
    )
