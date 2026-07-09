# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Profile RSS memory usage at each stage of dataset construction.

Run this on the machine/node where training actually OOMs, with the SAME
Hydra config you use for training, to find out which stage of dataset
construction is responsible before changing anything. Compares resident
memory before/after each step of the loading pipeline (open+slice+preprocess,
scale, retrieve_data/preload, weight computation, dataloader construction)
and reports the mesh/data volume so the numbers are interpretable.

Usage::

    python scripts/profile_dataset_memory.py experiment=swh_1h_densenet
    python scripts/profile_dataset_memory.py experiment=swh_1h_lstm data.train_days=[1,304]
    python scripts/profile_dataset_memory.py experiment=swh_1h_multi-head-gat-vn +stage=test

Note the ``+`` prefix on ``stage`` and ``report_zarr_chunks`` below: these
are keys this script adds on top of train.yaml, not overrides of existing
ones, so Hydra requires ``+key=value`` on the CLI (plain ``key=value``
would fail with "could not override").

Pass ``+stage=test`` to profile the (usually much larger) full-mesh test
dataset instead of train+val, and ``+report_zarr_chunks=false`` to skip
printing the on-disk zarr chunk shapes (on by default), which matter for
whether per-sample access in the graph datasets is cheap or expensive.

Requires ``psutil`` (add to the project's dev/test dependencies if missing).
"""

import gc
import os
from typing import Optional

import hydra
import psutil
import xarray as xr
from omegaconf import DictConfig

process = psutil.Process(os.getpid())


def rss_gib() -> float:
    """Current resident set size of this process, in GiB."""
    return process.memory_info().rss / 1024**3


class Stage:
    """Context manager that logs the RSS delta across a named step."""

    def __init__(self, name: str):
        self.name = name
        self.before = 0.0

    def __enter__(self):
        gc.collect()
        self.before = rss_gib()
        print(f"[{self.name}] start — RSS={self.before:.3f} GiB")
        return self

    def __exit__(self, *exc):
        gc.collect()
        after = rss_gib()
        print(f"[{self.name}] done  — RSS={after:.3f} GiB (Δ={after - self.before:+.3f} GiB)")
        return False


def report_zarr_chunks(data_dir: str, variables: list) -> None:
    """Print on-disk zarr chunk shapes for the given variables.

    Chunk shape matters a lot for the graph datasets' per-sample
    ``.isel(time=t, nSCHISM_hgrid_node=subgraph_nodes)`` reads: if a variable
    is chunked with many time steps per chunk, a single-timestep read still
    pulls the whole chunk off disk. If it's chunked with few nodes per
    chunk, a scattered-node read touches many chunks. Neither is a memory
    bottleneck per se, but both make "lazy" access slow rather than cheap.
    """
    print("\n--- On-disk zarr chunk shapes ---")
    ds = xr.open_dataset(data_dir, engine="zarr")
    for var in variables:
        if var not in ds:
            print(f"  {var}: not in dataset (likely derived, e.g. prev_/delta_/sin_/cos_)")
            continue
        da = ds[var]
        chunks = da.encoding.get("chunks") or da.encoding.get("preferred_chunks")
        print(f"  {var}: dims={da.dims}, shape={da.shape}, on-disk chunks={chunks}")
    ds.close()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    stage: str = cfg.get("stage", "fit")
    report_chunks: bool = bool(cfg.get("report_zarr_chunks", True))

    print(f"Profiling data._target_={cfg.data._target_}, stage={stage}")
    print(f"Baseline RSS before touching anything: {rss_gib():.3f} GiB\n")

    if report_chunks:
        report_zarr_chunks(cfg.data.data_dir, list(cfg.data.features) + list(cfg.data.targets))
        print()

    with Stage("instantiate datamodule (no data touched yet)"):
        datamodule = hydra.utils.instantiate(cfg.data)

    with Stage(f"datamodule.setup(stage='{stage}')"):
        datamodule.setup(stage=stage)

    ds = None
    for attr in ("ds_train", "ds_val", "ds_test", "ds_pred"):
        candidate = getattr(datamodule, attr, None)
        if candidate is not None:
            ds = candidate
            print(f"\nInspecting {attr}: len={len(ds)}")
            n_nodes = getattr(ds, "n_nodes", getattr(ds, "n_center_nodes", None))
            if n_nodes is not None:
                print(f"  n_nodes / n_center_nodes = {n_nodes}")
            if hasattr(ds, "X"):
                print(f"  X: shape={tuple(ds.X.shape)}, dtype={ds.X.dtype}, nbytes={ds.X.nbytes / 1024**3:.3f} GiB")
            if hasattr(ds, "y"):
                print(f"  y: shape={tuple(ds.y.shape)}, dtype={ds.y.dtype}, nbytes={ds.y.nbytes / 1024**3:.3f} GiB")
            if getattr(ds, "weights", None) is not None:
                w = ds.weights
                print(f"  weights: shape={tuple(w.shape)}, nbytes={w.nbytes / 1024**3:.3f} GiB")
            if hasattr(ds, "_targets_ram"):
                t = ds._targets_ram
                print(f"  _targets_ram: shape={tuple(t.shape)}, nbytes={t.nbytes / 1024**3:.3f} GiB")
            if hasattr(ds, "graph_cache"):
                print(f"  graph_cache: {len(ds.graph_cache)} cached subgraph topologies")

    with Stage("build dataloader + fetch one batch"):
        loader_name = {"fit": "train_dataloader", "test": "test_dataloader", "predict": "predict_dataloader"}[stage]
        loader = getattr(datamodule, loader_name)()
        batch: Optional[object] = next(iter(loader))
        del batch

    print(f"\nFinal RSS: {rss_gib():.3f} GiB")
    print(
        "\nIf 'setup' shows the big jump: that's eager retrieve_data()/preload — "
        "see the lazy-loading strategy for SimpleWWMDataset/SequenceWWMDataset.\n"
        "If RSS is fine here but blows up later mid-training: suspect DataLoader "
        "worker duplication (check num_workers) or train+val+test coexisting "
        "(run this script with stage=test in a FRESH process to isolate it)."
    )


if __name__ == "__main__":
    main()
