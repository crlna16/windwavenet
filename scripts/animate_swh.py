# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Animate significant wave height (SWH) over time on a Cartopy map.

Produces an MP4 video where each frame scatter-plots mesh nodes
coloured by SWH at the corresponding time step.

Usage
-----
    python scripts/animate_swh.py \
        --netcdf-path /path/to/data.netcdf \
        --output outputs/swh_animation.mp4 \
        --time-start 0 --time-end 100 \
        --fps 10 --dpi 150 \
        --cmap viridis --subsample 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate significant wave height on a Cartopy map.")
    parser.add_argument(
        "--netcdf-path",
        type=str,
        required=True,
        help="Path to the NetCDF dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/swh_animation.mp4",
        help="Output video file path (default: outputs/swh_animation.mp4).",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10).")
    parser.add_argument(
        "--time-start",
        type=int,
        default=0,
        help="Start time index (0-based, default: 0).",
    )
    parser.add_argument(
        "--time-end",
        type=int,
        default=None,
        help="End time index (exclusive). Default: all time steps.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution (default: 150).")
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name (default: viridis).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Plot every Nth node for faster rendering (default: 1 = all).",
    )
    return parser.parse_args(argv)


def validate_dataset(ds: xr.Dataset) -> None:
    """Check that required variables and dimensions exist."""
    required_vars = ["sigWaveHeight", "SCHISM_hgrid_node_x", "SCHISM_hgrid_node_y"]
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise ValueError(f"Dataset is missing required variables: {missing}")

    if "time" not in ds.dims:
        raise ValueError("Dataset has no 'time' dimension.")
    if "nSCHISM_hgrid_node" not in ds.dims:
        raise ValueError("Dataset has no 'nSCHISM_hgrid_node' dimension.")

    swh = ds["sigWaveHeight"]
    if set(swh.dims) != {"time", "nSCHISM_hgrid_node"}:
        raise ValueError(f"Expected sigWaveHeight dims (time, nSCHISM_hgrid_node), got {swh.dims}")
    print(f"Dataset OK: {len(ds.time)} time steps, {len(ds.nSCHISM_hgrid_node)} nodes.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- Load dataset ---
    print(f"Loading dataset: {args.netcdf_path}")
    ds = xr.open_dataset(args.netcdf_path, engine="netcdf4")
    validate_dataset(ds)

    # --- Extract coordinates (subsample) ---
    step = args.subsample
    node_x = ds["SCHISM_hgrid_node_x"].values[::step]
    node_y = ds["SCHISM_hgrid_node_y"].values[::step]
    n_nodes = len(node_x)
    print(f"Plotting {n_nodes} nodes (subsample={step}).")

    # --- Time range ---
    t_start = args.time_start
    t_end = args.time_end if args.time_end is not None else len(ds.time)
    t_end = min(t_end, len(ds.time))
    n_frames = t_end - t_start
    print(f"Rendering frames {t_start}..{t_end - 1} ({n_frames} frames).")

    # --- Set up figure ---
    data_crs = ccrs.UTM(32)
    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.stock_img()  # Natural Earth shaded relief (green lowlands, brown mountains)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "ocean",
            "10m",
            facecolor="white",
            edgecolor="none",
        ),
        zorder=2,
    )
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=4)
    ax.coastlines(resolution="10m", linewidth=0.5, zorder=4)

    # Auto-extent from node bounding box (with padding)
    # Transform a few corner points to get lon/lat extent
    plate = ccrs.PlateCarree()
    corners_xy = np.array(
        [
            [node_x.min(), node_y.min()],
            [node_x.max(), node_y.max()],
        ]
    )
    corners_lonlat = plate.transform_points(data_crs, corners_xy[:, 0], corners_xy[:, 1])
    lon_min, lon_max = corners_lonlat[0, 0], corners_lonlat[1, 0]
    lat_min, lat_max = corners_lonlat[0, 1], corners_lonlat[1, 1]
    pad_lon = 0.02 * (lon_max - lon_min)
    pad_lat = 0.02 * (lat_max - lat_min)
    ax.set_extent(
        [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat],
        crs=plate,
    )

    # Initial scatter (first frame data)
    swh_0 = ds["sigWaveHeight"].isel(time=t_start).values[::step]
    scatter = ax.scatter(
        node_x,
        node_y,
        c=swh_0,
        cmap=args.cmap,
        vmin=0,
        vmax=7,
        s=5,
        edgecolors="none",
        transform=data_crs,
        zorder=3,
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02)
    cbar.set_label("Significant wave height (m)")

    # Timestamp text
    time_0 = str(ds.time.values[t_start])[:16].replace("T", " ")
    timestamp_text = ax.set_title(time_0, fontsize=11)

    fig.tight_layout()

    # --- Animation update function ---
    progress = tqdm(total=n_frames, desc="Rendering", unit="frame")

    def update(frame_idx: int) -> tuple:
        t_idx = t_start + frame_idx
        swh = ds["sigWaveHeight"].isel(time=t_idx).values[::step]
        scatter.set_array(swh)

        time_str = str(ds.time.values[t_idx])[:16].replace("T", " ")
        timestamp_text.set_text(time_str)

        progress.update(1)
        return (scatter, timestamp_text)

    # --- Write video ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    # Choose writer: FFmpeg if available, otherwise Pillow (GIF)
    suffix = output_path.suffix.lower()
    try:
        from matplotlib.animation import FFMpegWriter

        plt.rcParams["animation.ffmpeg_path"] = "/sw/spack-levante/ffmpeg-4.3.2-rmywep/bin/ffmpeg"

        writer = FFMpegWriter(
            fps=args.fps,
            metadata={"title": "SWH Animation"},
            extra_args=["-vcodec", "mpeg4", "-q:v", "5"],
        )
        if suffix not in (".mp4", ".mov", ".avi"):
            output_path = output_path.with_suffix(".mp4")
        print(f"Writing video (FFmpeg) to: {output_path}")
    except Exception:
        from matplotlib.animation import PillowWriter

        writer = PillowWriter(fps=args.fps)
        if suffix != ".gif":
            output_path = output_path.with_suffix(".gif")
        print(f"FFmpeg unavailable. Writing GIF to: {output_path}")

    anim.save(str(output_path), writer=writer, dpi=args.dpi)

    progress.close()
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
