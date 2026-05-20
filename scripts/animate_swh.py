# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Animate significant wave height (SWH) over time as a two-part video.

Part 1: Full domain with regridded heatmap (pcolormesh).
Part 2: German Bight zoom with graph nodes (scatter) and edges.

Both parts run over the same time range and are concatenated into one MP4.

Usage
-----
    python scripts/animate_swh.py \
        --netcdf-path /path/to/data.netcdf \
        --output outputs/swh_animation.mp4 \
        --time-start 0 --time-end 100 \
        --fps 10 --dpi 150 \
        --cmap viridis --resolution 5000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
from tqdm import tqdm

from src.data.graph_utils import build_adjacency


# --- Worker globals for parallel griddata ---
_worker_points = None
_worker_xi = None
_worker_yi = None


def _init_griddata_worker(points, xi, yi):
    """Initializer for multiprocessing workers: store shared arrays."""
    global _worker_points, _worker_xi, _worker_yi
    _worker_points = points
    _worker_xi = xi
    _worker_yi = yi


def _regrid_one_frame(swh_values):
    """Worker function: interpolate one SWH field onto the regular grid."""
    return griddata(_worker_points, swh_values, (_worker_xi, _worker_yi), method="linear")


# --- German Bight extent (lon/lat) ---
GERMAN_BIGHT_EXTENT = [5.5, 10.0, 53.0, 56.0]  # [lon_min, lon_max, lat_min, lat_max]

# --- Wadden Sea / coastal zoom extent (lon/lat) ---
# Bounded by: Groningen (west), Sylt (east), Helgoland (south), Rømø (north)
WADDEN_SEA_EXTENT = [6.5, 8.7, 54.1, 55.2]  # [lon_min, lon_max, lat_min, lat_max]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate SWH: full-domain heatmap + German Bight graph view.")
    parser.add_argument(
        "--netcdf-path",
        type=str,
        required=True,
        help="Path to the netcdf (or NetCDF) dataset.",
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
        "--resolution",
        type=float,
        default=5000.0,
        help="Regridding resolution in metres for full-domain heatmap (default: 5000).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Plot every Nth node in the German Bight view (default: 1 = all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for griddata interpolation (default: 1).",
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
    print(f"Dataset OK: {len(ds.time)} time steps, {len(ds.nSCHISM_hgrid_node)} nodes.")


def build_regular_grid(node_x: np.ndarray, node_y: np.ndarray, resolution: float):
    """Build a regular grid covering the node bounding box."""
    x_reg = np.arange(node_x.min(), node_x.max() + resolution, resolution)
    y_reg = np.arange(node_y.min(), node_y.max() + resolution, resolution)
    xi, yi = np.meshgrid(x_reg, y_reg)
    return x_reg, y_reg, xi, yi


def get_region_mask(node_x: np.ndarray, node_y: np.ndarray, data_crs, extent_lonlat):
    """Return boolean mask for nodes within a lon/lat bounding box."""
    plate = ccrs.PlateCarree()
    lon_min, lon_max, lat_min, lat_max = extent_lonlat
    corners_lonlat = np.array([[lon_min, lat_min], [lon_max, lat_max]])
    corners_utm = data_crs.transform_points(plate, corners_lonlat[:, 0], corners_lonlat[:, 1])
    x_min, x_max = corners_utm[0, 0], corners_utm[1, 0]
    y_min, y_max = corners_utm[0, 1], corners_utm[1, 1]
    mask = (node_x >= x_min) & (node_x <= x_max) & (node_y >= y_min) & (node_y <= y_max)
    return mask


def get_german_bight_mask(node_x: np.ndarray, node_y: np.ndarray, data_crs):
    """Return boolean mask for nodes within the German Bight extent."""
    return get_region_mask(node_x, node_y, data_crs, GERMAN_BIGHT_EXTENT)


def get_edges_for_subset(adjacency: dict, subset_indices: np.ndarray):
    """Get edge pairs (as local indices) for a subset of nodes.

    Uses the pre-built adjacency dict (from graph_utils.build_adjacency)
    exactly as the notebook VisualiseGraphSample does.

    Parameters
    ----------
    adjacency : dict
        Dict mapping global node position -> list of neighbour positions.
    subset_indices : np.ndarray
        Global 0-based positional indices of the subset nodes.

    Returns
    -------
    edge_src, edge_dst : np.ndarray
        Local indices into the subset array for each edge.
    """
    subset_set = set(subset_indices.tolist())
    global_to_local = {g: l for l, g in enumerate(subset_indices)}  # noqa: E741

    edge_src_list = []
    edge_dst_list = []
    for node_pos in subset_indices:
        for nb_pos in adjacency.get(int(node_pos), []):
            if nb_pos in subset_set:
                edge_src_list.append(global_to_local[int(node_pos)])
                edge_dst_list.append(global_to_local[nb_pos])

    return np.array(edge_src_list), np.array(edge_dst_list)


def setup_map_ax(ax, extent_lonlat=None, data_crs=None, node_x=None, node_y=None):
    """Add land, borders, coastlines and set extent."""
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "10m",
            facecolor="darkkhaki",
            edgecolor="none",
        ),
        zorder=2,
    )
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=4)
    ax.coastlines(resolution="10m", linewidth=0.5, zorder=4)

    plate = ccrs.PlateCarree()
    if extent_lonlat is not None:
        ax.set_extent(extent_lonlat, crs=plate)
    elif node_x is not None and node_y is not None and data_crs is not None:
        corners_xy = np.array([[node_x.min(), node_y.min()], [node_x.max(), node_y.max()]])
        corners_lonlat = plate.transform_points(data_crs, corners_xy[:, 0], corners_xy[:, 1])
        lon_min, lon_max = corners_lonlat[0, 0], corners_lonlat[1, 0]
        lat_min, lat_max = corners_lonlat[0, 1], corners_lonlat[1, 1]
        pad_lon = 0.02 * (lon_max - lon_min)
        pad_lat = 0.02 * (lat_max - lat_min)
        ax.set_extent(
            [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat],
            crs=plate,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- Load dataset ---
    print(f"Loading dataset: {args.netcdf_path}")
    ds = xr.open_dataset(args.netcdf_path, engine="netcdf4")
    validate_dataset(ds)

    # --- Coordinates ---
    data_crs = ccrs.UTM(32)
    node_x = ds["SCHISM_hgrid_node_x"].values
    node_y = ds["SCHISM_hgrid_node_y"].values

    # --- Time range ---
    t_start = args.time_start
    t_end = args.time_end if args.time_end is not None else len(ds.time)
    t_end = min(t_end, len(ds.time))
    n_frames = t_end - t_start
    print(f"Time range: {t_start}..{t_end - 1} ({n_frames} frames per part).")

    # --- Prepare regridding for Part 1 ---
    print(f"Building regular grid (resolution={args.resolution:.0f} m)...")
    x_reg, y_reg, xi, yi = build_regular_grid(node_x, node_y, args.resolution)
    points = np.column_stack([node_x, node_y])

    # --- Prepare German Bight subset for Part 2 ---
    print("Selecting German Bight nodes...")
    gb_mask = get_german_bight_mask(node_x, node_y, data_crs)
    gb_indices = np.where(gb_mask)[0]
    step = args.subsample
    gb_indices = gb_indices[::step]
    gb_x = node_x[gb_indices]
    gb_y = node_y[gb_indices]
    print(f"German Bight: {len(gb_indices)} nodes (subsample={step}).")

    # --- Prepare Wadden Sea subset for Part 3 ---
    print("Selecting Wadden Sea nodes...")
    ws_mask = get_region_mask(node_x, node_y, data_crs, WADDEN_SEA_EXTENT)
    ws_indices = np.where(ws_mask)[0]
    ws_indices = ws_indices[::step]
    ws_x = node_x[ws_indices]
    ws_y = node_y[ws_indices]
    print(f"Wadden Sea: {len(ws_indices)} nodes (subsample={step}).")

    # Pre-transform subset coordinates to lon/lat (avoids repeated UTM->PlateCarree
    # transforms during rendering, which is the main bottleneck for LineCollection)
    plate = ccrs.PlateCarree()
    gb_lonlat = plate.transform_points(data_crs, gb_x, gb_y)
    gb_lon, gb_lat = gb_lonlat[:, 0], gb_lonlat[:, 1]
    ws_lonlat = plate.transform_points(data_crs, ws_x, ws_y)
    ws_lon, ws_lat = ws_lonlat[:, 0], ws_lonlat[:, 1]

    print("Building adjacency from zarr (shared data)...")
    ds_zarr = xr.open_dataset(
        "/work/ka1176/shared_data/2025-wwm-emulator-schism/processed_RUN04b_JUWELS_wpile.zarr",
        engine="zarr",
    )
    adjacency = build_adjacency(ds_zarr)
    ds_zarr.close()
    print("Building edge list for German Bight subset...")
    edge_src, edge_dst = get_edges_for_subset(adjacency, gb_indices)
    print(f"German Bight edges: {len(edge_src)}")

    print("Building edge list for Wadden Sea subset...")
    ws_edge_src, ws_edge_dst = get_edges_for_subset(adjacency, ws_indices)
    print(f"Wadden Sea edges: {len(ws_edge_src)}")

    # Pre-compute edge line segments in lon/lat (avoids UTM transform at render time)
    segments = np.stack(
        [np.column_stack([gb_lon[edge_src], gb_lat[edge_src]]), np.column_stack([gb_lon[edge_dst], gb_lat[edge_dst]])],
        axis=1,
    )
    ws_segments = np.stack(
        [
            np.column_stack([ws_lon[ws_edge_src], ws_lat[ws_edge_src]]),
            np.column_stack([ws_lon[ws_edge_dst], ws_lat[ws_edge_dst]]),
        ],
        axis=1,
    )

    # --- Output path ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in (".mp4", ".mov", ".avi"):
        output_path = output_path.with_suffix(".mp4")

    # ===================================================================
    # Pre-load all SWH data into memory (single bulk read)
    # ===================================================================
    print(f"\nLoading SWH data for frames {t_start}..{t_end - 1} into memory...")
    swh_all = ds["sigWaveHeight"].isel(time=slice(t_start, t_end)).values  # (n_frames, n_nodes)
    print(f"SWH block loaded: {swh_all.shape}, {swh_all.nbytes / 1e6:.1f} MB")

    # ===================================================================
    # Pre-compute regridded fields for Part 1 (parallelizable)
    # ===================================================================
    n_workers = min(args.workers, n_frames)
    print(f"\n--- Pre-computing {n_frames} regridded fields ({n_workers} workers) ---")
    swh_arrays = [swh_all[i] for i in range(n_frames)]

    if n_workers > 1:
        with mp.Pool(n_workers, initializer=_init_griddata_worker, initargs=(points, xi, yi)) as pool:
            grids = list(
                tqdm(
                    pool.imap(_regrid_one_frame, swh_arrays),
                    total=n_frames,
                    desc="Regridding",
                    unit="frame",
                )
            )
    else:
        grids = []
        for swh in tqdm(swh_arrays, desc="Regridding", unit="frame"):
            grids.append(griddata(points, swh, (xi, yi), method="linear"))

    print(f"\nTotal frames: {3 * n_frames} (Part 1: {n_frames}, Part 2: {n_frames}, Part 3: {n_frames})")

    # ===================================================================
    # Render frames sequentially via FFmpeg pipe
    # ===================================================================
    ffmpeg_bin = "/sw/spack-levante/ffmpeg-4.3.2-rmywep/bin/ffmpeg"
    time_strings = [str(ds.time.values[t_start + i])[:16].replace("T", " ") for i in range(n_frames)]

    try:
        from matplotlib.animation import FFMpegWriter

        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_bin
        writer = FFMpegWriter(
            fps=args.fps,
            metadata={"title": "SWH Animation"},
            extra_args=["-vcodec", "mpeg4", "-q:v", "5"],
        )
        if output_path.suffix.lower() not in (".mp4", ".mov", ".avi"):
            output_path = output_path.with_suffix(".mp4")
    except Exception:
        from matplotlib.animation import PillowWriter

        writer = PillowWriter(fps=args.fps)
        output_path = output_path.with_suffix(".gif")
        print("FFmpeg unavailable, using PillowWriter (GIF).")

    print(f"Output: {output_path}")

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    with writer.saving(fig, str(output_path), dpi=args.dpi):
        # --- Part 1: Full domain heatmap ---
        print("\n--- Part 1: Full domain (writing frames) ---")
        ax.clear()
        setup_map_ax(ax, data_crs=data_crs, node_x=node_x, node_y=node_y)

        mesh = ax.pcolormesh(
            x_reg,
            y_reg,
            grids[0],
            cmap=args.cmap,
            vmin=0,
            vmax=7,
            transform=data_crs,
            shading="auto",
            zorder=1,
        )
        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
        cbar.set_label("Signifikante Wellenhöhe (m)")
        title = ax.set_title(f"\u2014 {time_strings[0]}", fontsize=11)
        fig.tight_layout()

        for i in tqdm(range(n_frames), desc="Part 1", unit="frame"):
            mesh.set_array(grids[i].ravel())
            title.set_text(f"Full domain \u2014 {time_strings[i]}")
            writer.grab_frame()

        cbar.remove()
        mesh.remove()

        # --- Part 2: German Bight scatter + edges ---
        print("\n--- Part 2: German Bight (scatter + edges) ---")
        ax.clear()
        setup_map_ax(ax, extent_lonlat=GERMAN_BIGHT_EXTENT)

        lc = LineCollection(
            segments,
            colors="grey",
            linewidths=0.3,
            alpha=0.4,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        ax.add_collection(lc)

        scatter = ax.scatter(
            gb_lon,
            gb_lat,
            c=swh_all[0][gb_indices],
            cmap=args.cmap,
            vmin=0,
            vmax=7,
            s=3,
            edgecolors="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        cbar2 = fig.colorbar(scatter, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
        cbar2.set_label("Signifikante Wellenhöhe (m)")
        title2 = ax.set_title(f"\u2014 {time_strings[0]}", fontsize=11)
        fig.tight_layout()

        for i in tqdm(range(n_frames), desc="Part 2", unit="frame"):
            scatter.set_array(swh_all[i][gb_indices])
            title2.set_text(f"German Bight \u2014 {time_strings[i]}")
            writer.grab_frame()

        cbar2.remove()

        # --- Part 3: Wadden Sea zoom (scatter + edges) ---
        print("\n--- Part 3: Wadden Sea zoom (scatter + edges) ---")
        ax.clear()
        setup_map_ax(ax, extent_lonlat=WADDEN_SEA_EXTENT)

        lc3 = LineCollection(
            ws_segments,
            colors="grey",
            linewidths=0.3,
            alpha=0.4,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )
        ax.add_collection(lc3)

        scatter3 = ax.scatter(
            ws_lon,
            ws_lat,
            c=swh_all[0][ws_indices],
            cmap=args.cmap,
            vmin=0,
            vmax=7,
            s=5,
            edgecolors="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        cbar3 = fig.colorbar(scatter3, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
        cbar3.set_label("Signifikante Wellenhöhe (m)")
        title3 = ax.set_title(f"\u2014 {time_strings[0]}", fontsize=11)
        fig.tight_layout()

        for i in tqdm(range(n_frames), desc="Part 3", unit="frame"):
            scatter3.set_array(swh_all[i][ws_indices])
            title3.set_text(f"Wadden Sea \u2014 {time_strings[i]}")
            writer.grab_frame()

    plt.close(fig)
    print(f"\nDone. Video saved to: {output_path}")


if __name__ == "__main__":
    main()
