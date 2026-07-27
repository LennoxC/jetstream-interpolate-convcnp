"""
example.py
----------
End-to-end usage example. Replace `ds` with your real xarray Dataset
(e.g. xr.open_dataset("era5_slice.nc")).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from .windfield import WindField
    from .aircraft import PROFILES
    from .trajectory import Trajectory
    from .losses import LossWeights
    from .optimizer import TrajectoryOptimizer
    from .plotting import plot_trajectory
except ImportError:  # allow running as `python example.py` from inside the package dir
    from windfield import WindField
    from aircraft import PROFILES
    from trajectory import Trajectory
    from losses import LossWeights
    from optimizer import TrajectoryOptimizer
    from plotting import plot_trajectory


def build_synthetic_dataset():
    """Small synthetic dataset matching your schema, for a runnable demo."""
    import xarray as xr

    valid_time = np.arange("2019-07-01", "2019-07-02", np.timedelta64(6, "h"), dtype="datetime64[ns]")
    pressure_level = np.array([200, 250, 300, 500, 700])
    latitude = np.arange(-28.0, -46.25, -0.25)
    longitude = np.arange(147.0, 173.25, 0.25)

    rng = np.random.default_rng(0)
    shape = (len(valid_time), len(pressure_level), len(latitude), len(longitude))
    # rough jet-stream-like structure: stronger westerlies at high altitude
    plev_factor = (700 - pressure_level) / 500.0  # bigger at low pressure (high alt)
    u = 15 + 25 * plev_factor[None, :, None, None] * np.ones(shape) + rng.normal(0, 2, shape)
    v = rng.normal(0, 5, shape)
    # geopotential height increasing as pressure decreases (rough US std atmosphere)
    z_by_level = {200: 11800, 250: 10400, 300: 9200, 500: 5700, 700: 3000}
    z = np.stack([np.full(shape[0:1] + shape[2:], z_by_level[p]) for p in pressure_level], axis=1).astype(float)
    z = z + rng.normal(0, 30, z.shape)
    z = z * 9.80665  # dataset stores geopotential, not height

    ds = xr.Dataset(
        {
            "u": (("valid_time", "pressure_level", "latitude", "longitude"), u),
            "v": (("valid_time", "pressure_level", "latitude", "longitude"), v),
            "z": (("valid_time", "pressure_level", "latitude", "longitude"), z),
        },
        coords={
            "valid_time": valid_time,
            "pressure_level": pressure_level,
            "latitude": latitude,
            "longitude": longitude,
            "number": 0,
        },
    )
    return ds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_synthetic_dataset()
    windfield = WindField(ds, device=device)

    aircraft = PROFILES["narrowbody_jet"]

    # Origin/destination roughly spanning the domain (e.g. Sydney-ish -> Christchurch-ish)
    origin = (-28.0, 153.0, 10000.0)
    destination = (-43.5, 172.6, 10000.0)
    n_control_points = 6
    lats = np.linspace(origin[0], destination[0], n_control_points)
    lons = np.linspace(origin[1], destination[1], n_control_points)
    alts = np.full(n_control_points, 10000.0)
    waypoints = np.stack([lats, lons, alts], axis=1)

    start_time = ds.valid_time.values[0]
    trajectory = Trajectory(waypoints, start_time=start_time, fixed_endpoints=True, device=device)

    weights = LossWeights(time=1.0, fuel=0.05, curvature=5e4, smoothness=1e3, altitude=0.02)

    optimizer = TrajectoryOptimizer(trajectory, windfield, aircraft, weights=weights, num_samples=120, lr=0.05)
    optimizer.run(num_iters=200, verbose=True, log_every=20)

    fig = plot_trajectory(trajectory, windfield, aircraft, weights=weights, num_samples=200)
    plt.show()

    # Example: evaluate the same optimized path against a *different* dataset
    ds2 = build_synthetic_dataset()  # stand-in for e.g. a forecast vs. reanalysis comparison
    windfield2 = WindField(ds2, device=device)
    result_on_ds2 = optimizer.evaluate(windfield=windfield2)
    print("Loss of the same path evaluated on dataset #2:", result_on_ds2.as_dict())


if __name__ == "__main__":
    main()
