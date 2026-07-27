"""
trajectory.py
-------------
Trajectory is parametrized by K control points (lat, lon, alt). A Catmull-Rom
spline (implemented directly in torch, fully differentiable, no external
spline solver needed) is used to turn the control points into a dense path of
M samples for loss evaluation / plotting.

By default the first and last control points are kept fixed (the flight's
origin and destination) and only the interior control points are optimized,
which is almost always what you want for point-to-point route optimization.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def catmull_rom(cp: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Evaluate a Catmull-Rom spline through control points `cp` (K, D) at
    `num_samples` evenly spaced parameter values in [0, 1]. Returns (num_samples, D).
    Differentiable w.r.t. cp."""
    K = cp.shape[0]
    device, dtype = cp.device, cp.dtype
    if K < 2:
        raise ValueError("Need at least 2 control points")
    if K == 2:
        u = torch.linspace(0, 1, num_samples, device=device, dtype=dtype).unsqueeze(1)
        return cp[0].unsqueeze(0) * (1 - u) + cp[1].unsqueeze(0) * u

    # Pad control points so every real segment has 4 neighbours.
    cp_ext = torch.cat([cp[0:1], cp, cp[-1:]], dim=0)  # (K+2, D)
    n_seg = K - 1

    u = torch.linspace(0, 1, num_samples, device=device, dtype=dtype)
    seg_f = u * n_seg
    seg_f = torch.clamp(seg_f, max=n_seg - 1e-6)
    seg_idx = torch.floor(seg_f).long()
    t = (seg_f - seg_idx).unsqueeze(1)  # (M,1)

    p0 = cp_ext[seg_idx]
    p1 = cp_ext[seg_idx + 1]
    p2 = cp_ext[seg_idx + 2]
    p3 = cp_ext[seg_idx + 3]

    t2 = t * t
    t3 = t2 * t
    pos = 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )
    return pos


class Trajectory(nn.Module):
    """Optimizable flight path: K control points of (lat, lon, alt).

    Parameters
    ----------
    waypoints : array-like (K, 3)
        Initial guess for [lat, lon, alt_m] at each control point, in flight
        order (first = origin, last = destination).
    start_time : np.datetime64 or float
        Departure time. Used to seed the wind lookup along the path; total
        flight time is *estimated* from the wind field, not fixed.
    fixed_endpoints : bool
        If True (default), the first and last control points are held fixed
        and only interior points are optimized.
    """

    def __init__(
        self,
        waypoints,
        start_time,
        fixed_endpoints: bool = True,
        device="cpu",
        dtype=torch.float32,
    ):
        super().__init__()
        wp = torch.as_tensor(np.asarray(waypoints, dtype=np.float64), dtype=dtype, device=device)
        assert wp.ndim == 2 and wp.shape[1] == 3, "waypoints must be (K,3): lat, lon, alt_m"
        self.fixed_endpoints = fixed_endpoints
        self.device = device
        self.dtype = dtype

        if fixed_endpoints and wp.shape[0] > 2:
            self.register_buffer("p_start", wp[0:1].clone())
            self.register_buffer("p_end", wp[-1:].clone())
            self.interior = nn.Parameter(wp[1:-1].clone())
        else:
            self.p_start = None
            self.p_end = None
            self.interior = nn.Parameter(wp.clone())

        if isinstance(start_time, (np.datetime64,)):
            from .windfield import datetime_to_seconds
            self.start_time_sec = float(datetime_to_seconds([start_time])[0])
        else:
            self.start_time_sec = float(start_time)

    def control_points(self) -> torch.Tensor:
        if self.p_start is not None:
            return torch.cat([self.p_start, self.interior, self.p_end], dim=0)
        return self.interior

    def sample_path(self, num_samples: int = 200) -> torch.Tensor:
        """Returns (num_samples, 3) tensor of [lat, lon, alt_m] along the path."""
        return catmull_rom(self.control_points(), num_samples)

    def as_numpy(self, num_samples: int = 200) -> np.ndarray:
        with torch.no_grad():
            return self.sample_path(num_samples).cpu().numpy()

    def control_points_numpy(self) -> np.ndarray:
        with torch.no_grad():
            return self.control_points().cpu().numpy()
