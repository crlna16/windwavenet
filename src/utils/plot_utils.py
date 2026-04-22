# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.figure import Figure


def regrid_to_regular(
    da: xr.DataArray,
    resolution: float,
    method: str = "linear",
    x_coord: str = "SCHISM_hgrid_node_x",
    y_coord: str = "SCHISM_hgrid_node_y",
    node_dim: str = "nSCHISM_hgrid_node",
) -> xr.DataArray:
    """Regrid an unstructured-mesh DataArray to a regular x/y grid.

    Parameters
    ----------
    da : xr.DataArray
        DataArray on ``(time, nSCHISM_hgrid_node)`` or ``(nSCHISM_hgrid_node,)``
        with coordinates ``SCHISM_hgrid_node_x`` / ``SCHISM_hgrid_node_y``
        (projected, in metres).
    resolution : float
        Grid spacing in metres for the regular output grid.
    method : str, optional
        Interpolation method passed to ``scipy.interpolate.griddata``.
        One of ``"nearest"``, ``"linear"`` (default), ``"cubic"``.
    x_coord : str, optional
        Name of the x-coordinate variable in *da*.
    y_coord : str, optional
        Name of the y-coordinate variable in *da*.
    node_dim : str, optional
        Name of the unstructured node dimension in *da*.

    Returns
    -------
    xr.DataArray
        DataArray with dimensions ``(time, y, x)`` or ``(y, x)`` on a
        regular grid.  Points outside the convex hull of the mesh are
        set to NaN (for ``method="linear"`` or ``"cubic"``).
    """
    node_x = da.coords[x_coord].values
    node_y = da.coords[y_coord].values

    # Build regular grid aligned to the mesh bounding box
    x_min, x_max = node_x.min(), node_x.max()
    y_min, y_max = node_y.min(), node_y.max()

    x_reg = np.arange(x_min, x_max + resolution, resolution)
    y_reg = np.arange(y_min, y_max + resolution, resolution)
    xi, yi = np.meshgrid(x_reg, y_reg)

    points = np.column_stack([node_x, node_y])
    values = da.values

    has_time = "time" in da.dims

    if has_time:
        n_time = values.shape[0]
        regridded = np.empty((n_time, *xi.shape), dtype=np.float32)
        for t in range(n_time):
            regridded[t] = griddata(points, values[t], (xi, yi), method=method)
        dims = ["time", "y", "x"]
        coords: dict[str, object] = {"time": da.coords["time"], "x": x_reg, "y": y_reg}
    else:
        regridded = griddata(points, values, (xi, yi), method=method).astype(np.float32)
        dims = ["y", "x"]
        coords = {"x": x_reg, "y": y_reg}

    da_reg = xr.DataArray(regridded, dims=dims, coords=coords, name=da.name)
    da_reg.coords["x"].attrs = {"units": "m", "long_name": "projection_x_coordinate"}
    da_reg.coords["y"].attrs = {"units": "m", "long_name": "projection_y_coordinate"}

    return da_reg


def plot_on_map(
    da: xr.DataArray,
    utm_zone: int = 32,
    northern_hemisphere: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    label: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    ax: GeoAxes | None = None,
) -> Figure:
    """Plot a regridded DataArray on a cartopy map.

    Parameters
    ----------
    da : xr.DataArray
        2-D array with dimensions ``(y, x)`` in UTM metre coordinates
        (as produced by :func:`regrid_to_regular`).
    utm_zone : int, optional
        UTM zone number (default 32).
    northern_hemisphere : bool, optional
        Whether the UTM zone is in the northern hemisphere (default True).
    vmin, vmax : float, optional
        Colour-scale limits.  Passed to ``pcolormesh``.
    cmap : str, optional
        Matplotlib colour-map name (default ``"viridis"``).
    title : str, optional
        Plot title.  Falls back to ``da.name`` if not given.
    label : str, optional
        Colour-bar label.  Falls back to ``da.name`` if not given.
    figsize : tuple, optional
        Figure size in inches (default ``(10, 8)``).
    ax : matplotlib Axes, optional
        Existing GeoAxes to plot into.  If *None*, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    data_crs = ccrs.UTM(utm_zone, southern_hemisphere=not northern_hemisphere)

    if ax is None:
        fig, ax = plt.subplots(  # type: ignore[assignment]
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    assert ax is not None
    assert fig is not None

    mesh = ax.pcolormesh(
        da.coords["x"].values,
        da.coords["y"].values,
        np.ma.masked_invalid(da.values),
        transform=data_crs,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        zorder=1,
    )

    ax.add_feature(cfeature.LAND, facecolor="white", zorder=2)
    ax.coastlines(resolution="10m", zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=3)
    # ax.gridlines(draw_labels=False)

    fig.colorbar(mesh, ax=ax, label=label or str(da.name) or "", orientation="horizontal", shrink=0.7)  # type: ignore[union-attr]

    ax.set_title(title or str(da.name) or "")

    fig.tight_layout()

    return fig  # type: ignore[return-value]


def plot_bias_on_map(
    da_a: xr.DataArray,
    da_b: xr.DataArray,
    resolution: float,
    utm_zone: int = 32,
    northern_hemisphere: bool = True,
    vlim: float | None = None,
    cmap: str = "RdBu_r",
    title: str | None = None,
    label: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    ax: GeoAxes | None = None,
) -> Figure:
    """Plot the bias (da_a − da_b) on a cartopy map with a diverging colourmap.

    The difference is computed on the original unstructured mesh, then
    regridded once (linear interpolation) to avoid double-interpolation
    error.

    Parameters
    ----------
    da_a, da_b : xr.DataArray
        DataArrays on the unstructured mesh (same shape and coordinates).
        Typically *da_a* is the prediction and *da_b* is the reference.
    resolution : float
        Regular-grid spacing in metres passed to :func:`regrid_to_regular`.
    utm_zone : int, optional
        UTM zone number (default 32).
    northern_hemisphere : bool, optional
        Whether the UTM zone is in the northern hemisphere (default True).
    vlim : float, optional
        Symmetric colour-scale limit (``vmin=-vlim, vmax=+vlim``).
        If *None*, derived from the data.
    cmap : str, optional
        Matplotlib colour-map name (default ``"RdBu_r"``).
    title : str, optional
        Plot title.
    label : str, optional
        Colour-bar label.
    figsize : tuple, optional
        Figure size in inches (default ``(10, 8)``).
    ax : GeoAxes, optional
        Existing GeoAxes to plot into.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bias = da_a - da_b
    bias.name = title or "bias"

    bias_reg = regrid_to_regular(bias, resolution=resolution)

    if vlim is None:
        vlim = float(np.nanmax(np.abs(bias_reg.values)))

    return plot_on_map(
        bias_reg,
        utm_zone=utm_zone,
        northern_hemisphere=northern_hemisphere,
        vmin=-vlim,
        vmax=vlim,
        cmap=cmap,
        title=title,
        label=label,
        figsize=figsize,
        ax=ax,
    )
