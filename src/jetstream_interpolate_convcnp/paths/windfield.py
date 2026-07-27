"""
windfield.py
------------
Wraps an xarray Dataset with dims (valid_time, pressure_level, latitude, longitude)
and data variables u, v, z (geopotential) into a fully differentiable PyTorch
lookup table, so that wind at an arbitrary (time, lat, lon, altitude) can be
queried and back-propagated through.

Horizontal + time interpolation is done with `torch.nn.functional.grid_sample`
(trilinear). Vertical interpolation is done separately: z (converted to
geopotential *height* in metres) is sampled at every pressure level for the
query point, and u/v are then linearly interpolated between the two pressure
levels whose altitude brackets the requested altitude. This correctly handles
the fact that the height of a given pressure level varies in space and time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

G0 = 9.80665  # standard gravity, m/s^2
EPOCH = np.datetime64("1970-01-01T00:00:00")


def datetime_to_seconds(times) -> np.ndarray:
    """Convert any array-like of numpy datetime64 / python datetimes to
    float seconds since the unix epoch."""
    arr = np.asarray(times, dtype="datetime64[ns]").astype("datetime64[s]")
    return (arr - EPOCH).astype(np.int64).astype(np.float64)


class WindField:
    """Differentiable 4D (time, lat, lon, alt) wind field built from an
    xarray Dataset with variables u, v, z on (valid_time, pressure_level,
    latitude, longitude)."""

    def __init__(
        self,
        ds: xr.Dataset,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype

        # Normalise ordering: ascending lat/lon/time, and pressure levels
        # ordered from HIGH pressure (low altitude) to LOW pressure (high
        # altitude), i.e. 700 -> 200 hPa, so that altitude is ascending
        # along the pressure_level axis.
        ds = ds.sortby("valid_time").sortby("latitude").sortby("longitude")
        ds = ds.sortby("pressure_level", ascending=False)

        self.times_sec = datetime_to_seconds(ds.valid_time.values)  # (T,)
        self.plevels = ds.pressure_level.values.astype(np.float64)  # (P,)
        self.lats = ds.latitude.values.astype(np.float64)
        self.lons = ds.longitude.values.astype(np.float64)

        u = ds["u"].values  # (T, P, Lat, Lon)
        v = ds["v"].values
        z = ds["z"].values / G0  # geopotential height in metres

        self.t0, self.t1 = float(self.times_sec[0]), float(self.times_sec[-1])
        self.lat0, self.lat1 = float(self.lats.min()), float(self.lats.max())
        self.lon0, self.lon1 = float(self.lons.min()), float(self.lons.max())
        self.P = len(self.plevels)

        # grid_sample wants shape (N=1, C, D, H, W). We put pressure level on
        # the channel axis (C=P) so a single call returns all levels at once,
        # and (T, Lat, Lon) -> (D, H, W).
        def to_grid(arr):
            t = torch.tensor(arr, dtype=dtype, device=device)  # (T,P,Lat,Lon)
            t = t.permute(1, 0, 2, 3).unsqueeze(0)  # (1,P,T,Lat,Lon)
            return t.contiguous()

        self.u_grid = to_grid(u)
        self.v_grid = to_grid(v)
        self.z_grid = to_grid(z)

        self.alt_min = float(self.z_grid.min().item())
        self.alt_max = float(self.z_grid.max().item())

    @classmethod
    def from_netcdf(cls, path: str, **kwargs) -> "WindField":
        return cls(xr.open_dataset(path), **kwargs)

    def to(self, device):
        self.device = device
        self.u_grid = self.u_grid.to(device)
        self.v_grid = self.v_grid.to(device)
        self.z_grid = self.z_grid.to(device)
        return self

    def _normalize(self, t_sec, lat, lon):
        """Map raw coordinates into grid_sample's [-1, 1] convention.
        grid_sample grid order is (x -> W/lon, y -> H/lat, z -> D/time)."""
        x = 2 * (lon - self.lon0) / (self.lon1 - self.lon0 + 1e-12) - 1
        y = 2 * (lat - self.lat0) / (self.lat1 - self.lat0 + 1e-12) - 1
        zc = 2 * (t_sec - self.t0) / (self.t1 - self.t0 + 1e-12) - 1
        return x, y, zc

    def sample(self, t_sec: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor, alt: torch.Tensor):
        """Sample (u, v) wind components at N query points.

        All inputs are 1-D torch tensors of the same length N. lat/lon/alt
        may require grad (they typically come from the optimized trajectory);
        t_sec may be a plain tensor (it's usually derived, not optimized, but
        gradients will flow through it too if it does require grad).

        Returns: (u, v) each shape (N,), in m/s.
        """
        N = lat.shape[0]
        device = lat.device
        u_grid = self.u_grid.to(device)
        v_grid = self.v_grid.to(device)
        z_grid = self.z_grid.to(device)

        x, y, zc = self._normalize(t_sec, lat, lon)
        # clamp normalized coords so points slightly outside the domain
        # (e.g. during optimization overshoot) still get a sensible border value
        x = torch.clamp(x, -1.0, 1.0)
        y = torch.clamp(y, -1.0, 1.0)
        zc = torch.clamp(zc, -1.0, 1.0)

        grid = torch.stack([x, y, zc], dim=-1).view(1, N, 1, 1, 3)

        u_p = F.grid_sample(u_grid, grid, align_corners=True, mode="bilinear", padding_mode="border")
        v_p = F.grid_sample(v_grid, grid, align_corners=True, mode="bilinear", padding_mode="border")
        z_p = F.grid_sample(z_grid, grid, align_corners=True, mode="bilinear", padding_mode="border")

        # (1,P,N,1,1) -> (N,P)
        u_p = u_p.view(self.P, N).transpose(0, 1)
        v_p = v_p.view(self.P, N).transpose(0, 1)
        z_p = z_p.view(self.P, N).transpose(0, 1)

        z_min = z_p[:, 0]
        z_max = z_p[:, -1]
        alt_c = torch.clamp(alt, min=z_min, max=z_max)

        # Find bracketing level index per point (non-differentiable index
        # selection is fine -- gradient flows through the interpolation
        # fraction and the bracket values themselves).
        idx = torch.sum((z_p[:, :-1] <= alt_c.unsqueeze(1)).to(torch.int64), dim=1) - 1
        idx = torch.clamp(idx, 0, self.P - 2)
        ar = torch.arange(N, device=device)
        lo, hi = idx, idx + 1

        z_lo, z_hi = z_p[ar, lo], z_p[ar, hi]
        u_lo, u_hi = u_p[ar, lo], u_p[ar, hi]
        v_lo, v_hi = v_p[ar, lo], v_p[ar, hi]

        frac = (alt_c - z_lo) / (z_hi - z_lo + 1e-6)
        u = u_lo + frac * (u_hi - u_lo)
        v = v_lo + frac * (v_hi - v_lo)
        return u, v

    def height_range(self):
        """Approximate (min, max) altitude in metres covered by the dataset."""
        return self.alt_min, self.alt_max

    def time_range(self):
        """(start, end) as numpy datetime64[s]."""
        return (
            (EPOCH + np.timedelta64(int(self.t0), "s")),
            (EPOCH + np.timedelta64(int(self.t1), "s")),
        )
