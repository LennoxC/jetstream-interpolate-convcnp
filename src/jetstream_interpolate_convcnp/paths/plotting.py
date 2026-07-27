"""
plotting.py
-----------
Visualisation of an optimized (or initial) flight path: a plan view (lon/lat)
with a wind quiver at a chosen pressure level, plus altitude and ground-speed
profiles along the route.

Terrain/coastline outlines are drawn via cartopy if it's installed
(`pip install cartopy`); if it isn't, plotting silently falls back to plain
matplotlib axes with no basemap, so the module still works everywhere.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from .losses import compute_losses, LossWeights

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False


def _add_terrain(ax):
    """Draw coastlines/land/ocean/borders on `ax` if cartopy is available
    and `ax` is a GeoAxes. No-op otherwise."""
    if not _HAS_CARTOPY:
        return
    ax.add_feature(cfeature.LAND, facecolor="#e8e4d8", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe3f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=1)


def _wind_quiver_layer(ax, windfield, plevel_index: int, time_index: int, transform=None):
    lons = windfield.lons
    lats = windfield.lats
    u = windfield.u_grid[0, plevel_index, time_index].cpu().numpy()
    v = windfield.v_grid[0, plevel_index, time_index].cpu().numpy()
    speed = np.sqrt(u ** 2 + v ** 2)
    step = max(1, len(lons) // 25)
    LON, LAT = np.meshgrid(lons, lats)
    kwargs = {"transform": transform} if transform is not None else {}
    cf = ax.contourf(LON, LAT, speed, levels=20, cmap="viridis", alpha=0.6, zorder=2, **kwargs)
    ax.quiver(
        LON[::step, ::step], LAT[::step, ::step],
        u[::step, ::step], v[::step, ::step],
        color="white", scale=800, width=0.002, zorder=3, **kwargs,
    )
    return cf


def plot_trajectory(
    trajectory,
    windfield,
    aircraft,
    weights: Optional[LossWeights] = None,
    num_samples: int = 200,
    plevel_index: Optional[int] = None,
    time_index: int = 0,
    title: str = "Optimized flight path",
    show_terrain: bool = True,
):
    """Produce a 3-panel figure: plan view with wind field + path (+ terrain
    outline, if cartopy is installed and show_terrain=True), altitude
    profile, and ground-speed profile along the path."""
    weights = weights or LossWeights()
    with torch.no_grad():
        result = compute_losses(trajectory, windfield, aircraft, weights, num_samples)
    path = result.path.cpu().numpy()
    t_sec = result.t_sec.cpu().numpy()
    lat, lon, alt = path[:, 0], path[:, 1], path[:, 2]

    if plevel_index is None:
        # pick the level closest to the median cruise altitude
        med_alt = np.median(alt)
        alts_at_levels = windfield.z_grid[0, :, time_index].mean(dim=(-1, -2)).cpu().numpy()
        plevel_index = int(np.argmin(np.abs(alts_at_levels - med_alt)))

    use_geo = show_terrain and _HAS_CARTOPY
    transform = ccrs.PlateCarree() if use_geo else None

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])

    if use_geo:
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        lon_span = max(lon.max() - lon.min(), 1e-3)
        lat_span = max(lat.max() - lat.min(), 1e-3)
        pad_lon = max(0.15 * lon_span, 0.5)
        pad_lat = max(0.15 * lat_span, 0.5)
        ax_map.set_extent(
            [max(lon.min() - pad_lon, windfield.lons.min()),
             min(lon.max() + pad_lon, windfield.lons.max()),
             max(lat.min() - pad_lat, windfield.lats.min()),
             min(lat.max() + pad_lat, windfield.lats.max())],
            crs=transform,
        )
        _add_terrain(ax_map)
    else:
        if show_terrain and not _HAS_CARTOPY:
            print("plot_trajectory: cartopy not installed, skipping terrain outline "
                  "(pip install cartopy to enable it).")
        ax_map = fig.add_subplot(gs[0, :])

    cf = _wind_quiver_layer(ax_map, windfield, plevel_index, time_index, transform=transform)
    fig.colorbar(cf, ax=ax_map, label="wind speed (m/s)")
    plot_kwargs = {"transform": transform} if use_geo else {}
    ax_map.plot(lon, lat, color="red", linewidth=2, label="flight path", zorder=4, **plot_kwargs)
    ax_map.scatter([lon[0], lon[-1]], [lat[0], lat[-1]], color="black", zorder=5, **plot_kwargs)
    ax_map.annotate("origin", (lon[0], lat[0]))
    ax_map.annotate("destination", (lon[-1], lat[-1]))
    cps = trajectory.control_points_numpy()
    ax_map.scatter(cps[:, 1], cps[:, 0], color="orange", marker="x", s=40,
                   label="control points", zorder=5, **plot_kwargs)
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.set_title(f"{title}  |  level={windfield.plevels[plevel_index]:.0f} hPa")
    ax_map.legend(loc="upper right")

    dist_km = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(lat) ** 2 + np.diff(lon) ** 2))]) * 111.0

    ax_alt = fig.add_subplot(gs[1, 0])
    ax_alt.plot(dist_km, alt)
    ax_alt.set_xlabel("distance along path (approx km)")
    ax_alt.set_ylabel("altitude (m)")
    ax_alt.set_title("Altitude profile")

    ax_speed = fig.add_subplot(gs[1, 1])
    dt = np.diff(t_sec)
    seg_dist_m = np.sqrt(np.diff(lat) ** 2 + np.diff(lon) ** 2) * 111000.0
    gs_mps = seg_dist_m / np.clip(dt, 1e-3, None)
    ax_speed.plot(dist_km[1:], gs_mps)
    ax_speed.set_xlabel("distance along path (approx km)")
    ax_speed.set_ylabel("ground speed (m/s)")
    ax_speed.set_title("Ground speed profile")

    fig.tight_layout()
    stats = result.as_dict()
    fig.suptitle(
        f"J={stats['total']:.2f}  T={stats['time_s']/60:.1f} min  "
        f"F={stats['fuel_kg']:.1f} kg  C={stats['curvature']:.4f}  "
        f"S={stats['smoothness']:.4f}  H={stats['altitude']:.1f}",
        y=1.02,
    )
    return fig