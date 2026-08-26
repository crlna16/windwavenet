# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Precompute RMSE/bias metrics for all experiments under logs/predict_final.

Scans experiment subfolders, skips ones that don't yet have both the summer
(predictions_183-213.nc) and winter (predictions_336-366.nc) prediction files,
and for the rest computes total/summer/winter RMSE + bias plus binned scores
(by sigWaveHeight value and by its timestep-to-timestep change) against the
ground-truth zarr. Results are saved to CSV so the Evaluation notebook can just
load them instead of recomputing on every run.

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
from src.utils.plot_functions import compute_binned_scores, compute_binned_change_scores  # noqa: E402

VARIABLE = "sigWaveHeight"

PREDICT_FINAL_DIR = Path("/work/ka1176/caroline/gitlab/wave-wind-model-emulator/logs/predict_2020")
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

VALUE_BINS = [0, 0.5, 1.0, 2.0, 4.0]
CHANGE_BINS = [-2, -1, -0.5, 0, 0.5, 1, 2]


def find_complete_experiments(predict_final_dir: Path):
    """Yield (experiment_name, summer_path, winter_path) for folders that have both prediction files."""
    for exp_dir in sorted(p for p in predict_final_dir.iterdir() if p.is_dir()):
        summer_path = exp_dir / SUMMER_FILE
        winter_path = exp_dir / WINTER_FILE
        if summer_path.exists() and winter_path.exists():
            yield exp_dir.name, summer_path, winter_path
        else:
            print(f"Skipping '{exp_dir.name}': missing {SUMMER_FILE} and/or {WINTER_FILE}")


def build_period_dataset(exp_name: str, nc_path: Path, period_start: str, chunks=CHUNKS) -> xr.Dataset:
    """Build a (experiment=[exp_name, 'ground_truth'], time, node) Dataset for one prediction file."""
    ds_pred = xr.open_dataset(nc_path, engine="netcdf4", chunks=chunks)[[VARIABLE]]
    ds_pred = ds_pred.sel(time=slice(period_start, None))
    ds_pred = ds_pred.expand_dims(experiment=[exp_name])

    # chunks={} keeps the zarr store's native chunking instead of "auto" picking
    # misaligned chunk boundaries (which xarray otherwise warns about).
    ds_true = xr.open_zarr(GROUND_TRUTH_ZARR, chunks={})[[VARIABLE]]
    ds_true = ds_true.sel(time=ds_pred.time.values).expand_dims(experiment=["ground_truth"])

    ds = xr.concat([ds_pred, ds_true], dim="experiment")
    ds = exclude_boundary_nodes(ds, str(BOUNDARY_NODES_CSV))
    return ds


def rmse_bias(ds: xr.Dataset, exp_name: str) -> dict:
    """RMSE and bias of experiment `exp_name` vs 'ground_truth' over the whole dataset."""
    gt = ds[VARIABLE].sel(experiment="ground_truth").values
    pred = ds[VARIABLE].sel(experiment=exp_name).values
    error = pred - gt
    return {"rmse": float(np.sqrt(np.nanmean(error**2))), "bias": float(np.nanmean(error))}


def main(predict_final_dir: Path = PREDICT_FINAL_DIR, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    binned_value_dfs = []
    binned_change_dfs = []

    for exp_name, summer_path, winter_path in find_complete_experiments(predict_final_dir):
        print(f"Processing '{exp_name}' ...")
        ds_summer = build_period_dataset(exp_name, summer_path, SUMMER_START)
        ds_winter = build_period_dataset(exp_name, winter_path, WINTER_START)
        # NOTE: ds_total joins two non-contiguous periods (summer/winter), so the single
        # timestep-to-timestep diff() at their boundary is not a physically meaningful change.
        # This affects one out of ~1500 samples per node in the binned-change metrics below,
        # which is negligible for RMSE/bias aggregated over ~5e5 nodes.
        ds_total = xr.concat([ds_summer, ds_winter], dim="time").sortby("time")

        total = rmse_bias(ds_total, exp_name)
        summer = rmse_bias(ds_summer, exp_name)
        winter = rmse_bias(ds_winter, exp_name)
        summary_rows.append(
            {
                "experiment": exp_name,
                "total_rmse": total["rmse"],
                "total_bias": total["bias"],
                "summer_rmse": summer["rmse"],
                "summer_bias": summer["bias"],
                "winter_rmse": winter["rmse"],
                "winter_bias": winter["bias"],
            }
        )

        binned_value_dfs.append(compute_binned_scores(ds_total, VARIABLE, VALUE_BINS))
        binned_change_dfs.append(compute_binned_change_scores(ds_total, VARIABLE, CHANGE_BINS))

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
    args = parser.parse_args()
    main(predict_final_dir=args.predict_final_dir, output_dir=args.output_dir)
