"""
losses.py
---------
Differentiable loss components for flight-path optimization:

    J = lambda_t * T + lambda_f * F + lambda_c * C + lambda_s * S + lambda_h * H

  T : estimated travel time, from integrating ground speed (airspeed + wind
      component along track) over the path.
  F : fuel burn, from a regime-based fuel-flow proxy (climb/cruise/descent)
      integrated over the same estimated time.
  C : curvature penalty, from heading change per unit path length (enforces
      realistic turn rates).
  S : smoothness penalty, from the discrete second derivative of the spline
      path.
  H : altitude-change penalty, discouraging excessive climbs/descents.

Everything here operates on the *dense sampled path* from Trajectory.sample_path(),
and is written with plain torch ops so gradients flow back to the control points.

NOTE on ground-speed approximation: to keep the model differentiable and
simple, ground speed along the path is approximated as
    gs = airspeed_TAS + (wind vector) . (unit tangent of the path)
i.e. we assume the aircraft's heading approximately follows the path tangent
and ignore wind-correction (crab) angle. This is a standard simplification
for optimization-time cost estimates; swap in a proper wind-triangle solve
in `_ground_speed` if you need exact heading correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .aircraft import AircraftProfile

EARTH_RADIUS_M = 6371000.0


def haversine(lat1, lon1, lat2, lon2, R=EARTH_RADIUS_M):
    """Great-circle distance in metres between paired points (torch tensors, degrees)."""
    to_rad = torch.pi / 180.0
    phi1, phi2 = lat1 * to_rad, lat2 * to_rad
    dphi = (lat2 - lat1) * to_rad
    dlambda = (lon2 - lon1) * to_rad
    a = torch.sin(dphi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dlambda / 2) ** 2
    return 2 * R * torch.asin(torch.clamp(torch.sqrt(a + 1e-12), max=1.0))


def _wrap_angle(angle):
    """Wrap radians to (-pi, pi]."""
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


@dataclass
class LossWeights:
    time: float = 1.0
    fuel: float = 1.0
    curvature: float = 1.0
    smoothness: float = 1.0
    altitude: float = 1.0


@dataclass
class LossResult:
    total: torch.Tensor
    time_s: torch.Tensor
    fuel_kg: torch.Tensor
    curvature: torch.Tensor
    smoothness: torch.Tensor
    altitude: torch.Tensor
    path: torch.Tensor            # (M,3) dense sampled path [lat,lon,alt]
    t_sec: torch.Tensor           # (M,) estimated time at each sample, seconds since epoch
    components: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "time_s": float(self.time_s.detach().cpu()),
            "fuel_kg": float(self.fuel_kg.detach().cpu()),
            "curvature": float(self.curvature.detach().cpu()),
            "smoothness": float(self.smoothness.detach().cpu()),
            "altitude": float(self.altitude.detach().cpu()),
        }


def _estimate_time_and_fuel(path: torch.Tensor, start_time_sec: float, windfield, aircraft: AircraftProfile):
    """Sequentially integrate ground speed along the dense path to estimate
    a time-of-arrival at every sample and a fuel-flow regime per segment.

    path: (M,3) [lat, lon, alt_m]
    Returns: t_sec (M,), dt (M-1,), fuel_rate (M-1,) kg/s, ds (M-1,) metres
    """
    device = path.device
    M = path.shape[0]
    lat, lon, alt = path[:, 0], path[:, 1], path[:, 2]

    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    dalt = alt[1:] - alt[:-1]

    dh = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    ds = torch.sqrt(dh ** 2 + dalt ** 2 + 1e-9)

    mid_lat = (lat[:-1] + lat[1:]) / 2
    mid_lon = (lon[:-1] + lon[1:]) / 2
    mid_alt = (alt[:-1] + alt[1:]) / 2

    # local East/North components of the path tangent (equirectangular approx,
    # fine for regional-scale domains)
    coslat = torch.cos(mid_lat * torch.pi / 180.0)
    east = dlon * coslat
    north = dlat
    tangent_norm = torch.sqrt(east ** 2 + north ** 2 + 1e-12)
    east_u = east / tangent_norm
    north_u = north / tangent_norm

    t_sec = torch.zeros(M, dtype=path.dtype, device=device)
    t_sec_list = [torch.as_tensor(start_time_sec, dtype=path.dtype, device=device)]

    dt_list = []
    fuel_rate_list = []

    t_prev = t_sec_list[0]
    for i in range(M - 1):
        u, v = windfield.sample(
            t_prev.unsqueeze(0), mid_lat[i : i + 1], mid_lon[i : i + 1], mid_alt[i : i + 1]
        )
        wind_along = u[0] * east_u[i] + v[0] * north_u[i]
        gs = aircraft.cruise_tas_mps + wind_along
        gs = torch.clamp(gs, min=1.0)  # guard against pathological headwinds
        dt = ds[i] / gs
        dt_list.append(dt)

        vrate = dalt[i] / torch.clamp(dt, min=1e-3)
        fuel_rate = torch.where(
            vrate > aircraft.vertical_rate_threshold_mps,
            torch.as_tensor(aircraft.fuel_flow_climb_kgps, dtype=path.dtype, device=device),
            torch.where(
                vrate < -aircraft.vertical_rate_threshold_mps,
                torch.as_tensor(aircraft.fuel_flow_descent_kgps, dtype=path.dtype, device=device),
                torch.as_tensor(aircraft.fuel_flow_cruise_kgps, dtype=path.dtype, device=device),
            ),
        )
        fuel_rate_list.append(fuel_rate)

        t_next = t_prev + dt
        t_sec_list.append(t_next)
        t_prev = t_next

    t_sec = torch.stack(t_sec_list)
    dt = torch.stack(dt_list)
    fuel_rate = torch.stack(fuel_rate_list)
    return t_sec, dt, fuel_rate, ds


def compute_losses(
    trajectory,
    windfield,
    aircraft: AircraftProfile,
    weights: Optional[LossWeights] = None,
    num_samples: int = 200,
) -> LossResult:
    """Evaluate J = lambda_t*T + lambda_f*F + lambda_c*C + lambda_s*S + lambda_h*H
    for `trajectory` against `windfield` using `aircraft`'s performance profile.

    `trajectory` may be evaluated against any WindField instance with the same
    coordinate conventions -- this is how you compare a path across two
    different weather datasets.
    """
    weights = weights or LossWeights()
    path = trajectory.sample_path(num_samples)  # (M,3)

    t_sec, dt, fuel_rate, ds = _estimate_time_and_fuel(path, trajectory.start_time_sec, windfield, aircraft)

    T = t_sec[-1] - t_sec[0]
    F_ = torch.sum(fuel_rate * dt)

    lat, lon, alt = path[:, 0], path[:, 1], path[:, 2]
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    coslat = torch.cos(((lat[:-1] + lat[1:]) / 2) * torch.pi / 180.0)
    heading = torch.atan2(dlon * coslat, dlat)  # (M-1,) radians, 0=north

    dtheta = _wrap_angle(heading[1:] - heading[:-1])  # (M-2,)
    seg_len = torch.clamp((ds[1:] + ds[:-1]) / 2, min=1.0)
    curvature = dtheta / seg_len  # rad/m
    C = torch.sum(curvature ** 2 * seg_len)

    d2 = path[2:] - 2 * path[1:-1] + path[:-2]  # (M-2,3), discrete 2nd derivative
    S = torch.sum(d2 ** 2)

    dalt = alt[1:] - alt[:-1]
    H = torch.sum(dalt ** 2)

    total = weights.time * T + weights.fuel * F_ + weights.curvature * C + weights.smoothness * S + weights.altitude * H

    return LossResult(
        total=total,
        time_s=T,
        fuel_kg=F_,
        curvature=C,
        smoothness=S,
        altitude=H,
        path=path,
        t_sec=t_sec,
    )
