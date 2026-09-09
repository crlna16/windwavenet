# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Precompute 10 km-resolution spatial RMSE + ground-truth maps for ONE experiment.

Unlike compute_experiment_metrics.py, this does not scan all experiments (the
griddata interpolation is too expensive to run for every one) -- pass the
specific experiment subfolder to process via --experiment-name.

For summer and winter separately, computes:
  - the per-node RMSE of the experiment's sigWaveHeight predictions vs ground
    truth, interpolated onto a regular grid.
  - the ground-truth sigWaveHeight time-mean, interpolated onto the same grid.

Both grids are saved to a single NetCDF file (dims: season, y, x) that can be
loaded and rendered with `src.utils.plot_functions.plot_spatial_grids` without
redoing the interpolation.

Usage: python compute_spatial_metrics.py --experiments-dir DIR --experiment-name NAME
       [--output-dir DIR] [--resolution METERS]
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
    GROUND_TRUTH_ZARR,
    PREDICT_FINAL_DIR,
    SUMMER_FILE,
    SUMMER_START,
    VARIABLE,
    WINTER_FILE,
    WINTER_START,
    build_period_dataset,
)
from src.utils.plot_functions import compute_spatial_grid  # noqa: E402

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


def compute_season_grids(exp_name: str, nc_path: Path, period_start: str, resolution: float) -> tuple:
    """RMSE grid (`exp_name` vs ground truth) and ground-truth time-mean grid for one season."""
    ds = build_period_dataset(exp_name, nc_path, period_start)
    node_x, node_y = _node_coords(ds)

    gt = ds[VARIABLE].sel(experiment="ground_truth").values  # (time, node)
    pred = ds[VARIABLE].sel(experiment=exp_name).values

    node_rmse = np.sqrt(np.nanmean((pred - gt) ** 2, axis=0))
    node_gt_mean = np.nanmean(gt, axis=0)

    rmse_grid = compute_spatial_grid(node_x, node_y, node_rmse, resolution=resolution)
    gt_grid = compute_spatial_grid(node_x, node_y, node_gt_mean, resolution=resolution)
    return rmse_grid, gt_grid


def main(
    experiments_dir: Path = PREDICT_FINAL_DIR,
    experiment_name: str = None,
    output_dir: Path = OUTPUT_DIR,
    resolution: float = RESOLUTION,
):
    exp_dir = experiments_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    season_datasets = {}
    for season, (nc_file, period_start) in SEASON_FILES.items():
        nc_path = exp_dir / nc_file
        if not nc_path.exists():
            raise FileNotFoundError(f"Missing '{nc_file}' for experiment '{experiment_name}' in {exp_dir}")

        print(f"Processing '{experiment_name}' ({season}) ...")
        rmse_grid, gt_grid = compute_season_grids(experiment_name, nc_path, period_start, resolution)

        # rmse_grid/gt_grid share xi/yi: same node coords, same resolution, no x/y_range.
        season_datasets[season] = xr.Dataset(
            {
                "rmse": (["y", "x"], rmse_grid["zi"]),
                "ground_truth_mean": (["y", "x"], gt_grid["zi"]),
            },
            coords={"x": rmse_grid["xi"], "y": rmse_grid["yi"]},
        )

    ds_out = xr.concat(
        [season_datasets[s] for s in SEASON_FILES], dim=pd.Index(list(SEASON_FILES), name="season")
    )
    ds_out.attrs["variable"] = VARIABLE
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
    args = parser.parse_args()
    main(
        experiments_dir=args.experiments_dir,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        resolution=args.resolution,
    )
