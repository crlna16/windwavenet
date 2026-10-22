# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Precompute RMSE/bias metrics for all experiments under logs/predict_final.

Scans experiment subfolders, skips ones that don't yet have both the summer
(predictions_183-213.nc) and winter (predictions_336-366.nc) prediction files,
and for the rest computes total/summer/winter RMSE + bias plus binned scores
(by variable value and by its timestep-to-timestep change) against the
ground-truth zarr, for sigWaveHeight, meanWavePeriod and meanWaveDirection.
Not every experiment's netcdf necessarily contains meanWavePeriod/meanWaveDirection;
those are simply skipped for experiments/periods where they are absent, while
sigWaveHeight (required) is still processed normally. Results are saved to CSV so
the Evaluation notebook can just load them instead of recomputing on every run.

Usage: python compute_experiment_metrics.py [--predict-final-dir DIR] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from src.data.dataset_utils import exclude_boundary_nodes  # noqa: E402
from src.utils.plot_functions import (  # noqa: E402
    _circular_error,
    _reconstruct_wave_direction,
    compute_binned_scores,
    compute_binned_change_scores,
)

VARIABLE = "sigWaveHeight"  # required variable; experiments missing it are an error, not a skip

# bin edges per variable, applied to ground-truth values (VALUE_BINS) or their
# timestep-to-timestep change (CHANGE_BINS). None = placeholder, binning is skipped
# for that variable until real edges are filled in.
VALUE_BINS = {
    "sigWaveHeight": [0, 0.5, 1.0, 2.0, 4.0, 100],
    "meanWavePeriod": [0, 3, 5, 7, 100],
    "meanWaveDirection": [0, 45, 90, 135, 180, 225, 270, 315, 360],
}
CHANGE_BINS = {
    "sigWaveHeight": [-2, -1, -0.5, 0, 0.5, 1, 2],
    "meanWavePeriod": [-10, -1, 0, 1, 100],
    "meanWaveDirection": [-180, -90, -45, -15, 0, 15, 45, 90, 180],
}
VARIABLES = list(VALUE_BINS)
# meanWaveDirection is an angle: use wraparound-aware error instead of plain subtraction
CIRCULAR_VARIABLES = {"meanWaveDirection"}

PREDICT_FINAL_DIR = Path("/work/ka1176/caroline/gitlab/wind-wave-model-emulator/logs/predict_final")
GROUND_TRUTH_ZARR = "/work/ka1176/shared_data/2025-wwm-emulator-schism/processed_RUN04b_JUWELS_wpile.zarr"
BOUNDARY_NODES_CSV = REPO_ROOT / "src/data/node_selection/boundary_nodes.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "metrics"

SUMMER_FILE = "predictions_183-213.nc"
WINTER_FILE = "predictions_336-366.nc"

# start 2 days into each period, past every model's LSTM warm-up lag, so all
# experiments are compared over the exact same timestamps
SUMMER_START = "2020-07-03"
WINTER_START = "2020-12-03"
CHUNKS = "auto"


def find_complete_experiments(predict_final_dir: Path, summer_file=SUMMER_FILE, winter_file=WINTER_FILE):
    """Yield (experiment_name, summer_path, winter_path) for folders that have both prediction files."""
    for exp_dir in sorted(p for p in predict_final_dir.iterdir() if p.is_dir()):
        summer_path = exp_dir / summer_file
        winter_path = exp_dir / winter_file
        if summer_path.exists() and winter_path.exists():
            yield exp_dir.name, summer_path, winter_path
        else:
            print(f"Skipping '{exp_dir.name}': missing {summer_file} and/or {winter_file}")


def build_period_dataset(exp_name: str, nc_path: Path, period_start: str, chunks=CHUNKS) -> "tuple[xr.Dataset, list]":
    """Build a (experiment=[exp_name, 'ground_truth'], time, node) Dataset for one prediction file.

    Only the variables in VARIABLES that are actually present (after reconstructing
    meanWaveDirection from its sin/cos encoding, if that's how it was stored) in both the
    prediction file and the ground truth are kept, so experiments/periods missing
    meanWavePeriod/meanWaveDirection still work for sigWaveHeight.

    Returns:
        Tuple of (dataset, list of variable names available in this dataset).
    """
    ds_pred = xr.open_dataset(nc_path, engine="netcdf4", chunks=chunks)
    ds_pred = _reconstruct_wave_direction(ds_pred)

    # chunks={} keeps the zarr store's native chunking instead of "auto" picking
    # misaligned chunk boundaries (which xarray otherwise warns about).
    ds_true = xr.open_zarr(GROUND_TRUTH_ZARR, chunks={})
    ds_true = _reconstruct_wave_direction(ds_true)

    available = [v for v in VARIABLES if v in ds_pred.data_vars and v in ds_true.data_vars]
    missing = [v for v in VARIABLES if v not in available]
    if missing:
        print(f"  '{exp_name}' ({nc_path.name}): variable(s) not found, skipping for this file: {missing}")

    ds_pred = ds_pred[available].sel(time=slice(period_start, None))
    ds_pred = ds_pred.expand_dims(experiment=[exp_name])

    ds_true = ds_true[available].sel(time=ds_pred.time.values).expand_dims(experiment=["ground_truth"])

    ds = xr.concat([ds_pred, ds_true], dim="experiment")
    ds = exclude_boundary_nodes(ds, str(BOUNDARY_NODES_CSV))
    return ds, available


def rmse_bias(ds: xr.Dataset, exp_name: str, variable: str, circular: bool = False) -> dict:
    """RMSE and bias of experiment `exp_name` vs 'ground_truth' over the whole dataset."""
    gt = ds[variable].sel(experiment="ground_truth").values
    pred = ds[variable].sel(experiment=exp_name).values
    error = _circular_error(pred, gt) if circular else pred - gt
    return {"rmse": float(np.sqrt(np.nanmean(error**2))), "bias": float(np.nanmean(error))}


def main(predict_final_dir: Path = PREDICT_FINAL_DIR, output_dir: Path = OUTPUT_DIR, summer_file: str = SUMMER_FILE, winter_file: str=WINTER_FILE, summer_start: str=SUMMER_START, winter_start: str=WINTER_START):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    binned_value_dfs = []
    binned_change_dfs = []

    for exp_name, summer_path, winter_path in find_complete_experiments(predict_final_dir, summer_file=summer_file, winter_file=winter_file):
        print(f"Processing '{exp_name}' ...")
        ds_summer, summer_vars = build_period_dataset(exp_name, summer_path, summer_start)
        ds_winter, winter_vars = build_period_dataset(exp_name, winter_path, winter_start)
        # only keep variables available in both periods, so ds_total stays consistent
        variables = [v for v in VARIABLES if v in summer_vars and v in winter_vars]
        #if VARIABLE not in variables:
        #    raise RuntimeError(f"Required variable '{VARIABLE}' not found for experiment '{exp_name}'")
        ds_summer = ds_summer[variables]
        ds_winter = ds_winter[variables]
        # NOTE: ds_total joins two non-contiguous periods (summer/winter), so the single
        # timestep-to-timestep diff() at their boundary is not a physically meaningful change.
        # This affects one out of ~1500 samples per node in the binned-change metrics below,
        # which is negligible for RMSE/bias aggregated over ~5e5 nodes.
        ds_total = xr.concat([ds_summer, ds_winter], dim="time").sortby("time")

        for variable in variables:
            circular = variable in CIRCULAR_VARIABLES
            total = rmse_bias(ds_total, exp_name, variable, circular)
            summer = rmse_bias(ds_summer, exp_name, variable, circular)
            winter = rmse_bias(ds_winter, exp_name, variable, circular)
            summary_rows.append(
                {
                    "experiment": exp_name,
                    "variable": variable,
                    "total_rmse": total["rmse"],
                    "total_bias": total["bias"],
                    "summer_rmse": summer["rmse"],
                    "summer_bias": summer["bias"],
                    "winter_rmse": winter["rmse"],
                    "winter_bias": winter["bias"],
                }
            )

            if VALUE_BINS[variable] is not None:
                df = compute_binned_scores(ds_total, variable, VALUE_BINS[variable], circular=circular)
                binned_value_dfs.append(df.assign(variable=variable))
            if CHANGE_BINS[variable] is not None:
                df = compute_binned_change_scores(ds_total, variable, CHANGE_BINS[variable], circular=circular)
                binned_change_dfs.append(df.assign(variable=variable))

    if not summary_rows:
        raise RuntimeError(f"No complete experiments found under {predict_final_dir}")

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.concat(binned_value_dfs, ignore_index=True).to_csv(output_dir / "binned_value_metrics.csv", index=False)
    pd.concat(binned_change_dfs, ignore_index=True).to_csv(output_dir / "binned_change_metrics.csv", index=False)
    print(f"Saved metrics for {len(summary_rows)} experiments to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-final-dir", type=Path, default=PREDICT_FINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--summer-file", type=str, default=SUMMER_FILE)
    parser.add_argument("--winter-file", type=str, default=WINTER_FILE)
    parser.add_argument("--summer-start", type=str, default=SUMMER_START)
    parser.add_argument("--winter-start", type=str, default=WINTER_START)
    args = parser.parse_args()
    main(predict_final_dir=args.predict_final_dir, output_dir=args.output_dir, summer_file=args.summer_file, winter_file=args.winter_file, summer_start=args.summer_start, winter_start=args.winter_start)
