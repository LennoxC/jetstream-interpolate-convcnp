import torch
from jetstream_interpolate_convcnp.utils.device import device
from jetstream_interpolate_convcnp.processing.amdar.AMDARInterface import AMDARInterface
from jetstream_interpolate_convcnp.processing.ecmwf.ECMWFInterface import ECMWFInterface
from jetstream_interpolate_convcnp.utils.constants import ALTITUDE, TIME, LATITUDE, LONGITUDE, WIND_U, WIND_V
from jetstream_interpolate_convcnp.utils.conversions import metres_to_degrees
import numpy as np

"""
sample code for reference

x, y, z = torch.meshgrid(
    torch.linspace(-1, 1, x_size),
    torch.linspace(-1, 1, y_size),
    torch.linspace(-1, 1, z_size),
    indexing='ij'
)

xc_grid = torch.stack((x, y, z), dim=0)  
xc_grid = xc_grid.unsqueeze(0).repeat(b_size, 1, 1, 1, 1)
xt = xc_grid

yc_grid = torch.tensor(velocity_field_batch, dtype=torch.float32)  # shape: (x_size, y_size, z_size, 2)
yc_grid = yc_grid.permute(0, 4, 1, 2, 3)

# mask some inputs
mask = torch.ones(b_size, x_size, y_size, z_size, dtype=torch.float32)

# mask out 50% of the points randomly
mask = mask.flatten(1)  # shape: (b_size, x_size * y_size * z_size)
for i in range(mask.shape[0]):
    indices = torch.randperm(mask.shape[1])[:mask.shape[1] // 2]
    mask[i, indices] = 0
    
mask = mask.view(b_size, x_size, y_size, z_size)  # shape: (b_size, x_size, y_size, z_size)
mask = mask.unsqueeze(1) # shape: (b_size, 1, x_size, y_size, z_size) to match yc_grid's shape

"""

class TaskBuilder:
    def __init__(self, settings):
        self.settings = settings
        self.amdar = AMDARInterface(settings)
        self.ecmwf = ECMWFInterface(settings)
        self.device = torch.device(device)

    def build_tasks(self, sample_idx_batch):
        # build tasks for a batch of samples
        # this might be multithreded eventually
        # sample_idx_batch is a list of tuples of (date, sample_idx) for each sample in the batch
        x_size = self.settings['training']['xy_resolution']
        y_size = self.settings['training']['xy_resolution']
        z_size = self.settings['training']['z_resolution']
        b_size = len(sample_idx_batch)

        amdar_batch = torch.zeros(b_size, 2, x_size, y_size, z_size, dtype=torch.float32, device=self.device)
        ecmwf_batch = torch.zeros(b_size, 2, x_size, y_size, z_size, dtype=torch.float32, device=self.device)

        for b_idx, ((year, month, day), sample_idx) in enumerate(sample_idx_batch):
            amdar_tensor, ecmwf_tensor = self.build_task(year, month, day, sample_idx, x_size, y_size, z_size)
            amdar_batch[b_idx] = amdar_tensor
            ecmwf_batch[b_idx] = ecmwf_tensor

        return amdar_batch, ecmwf_batch

        

    def build_task(self, year, month, day, sample_idx, x_size, y_size, z_size):
        # build a task for a given sample, which includes the context and target points and values for that sample.
        
        # centre the coordinates for the sample based on the lat/lon/alt of the sample
        # add a random shift to the centre so the model doesn't rely on a centered sample

        sample = self.amdar.fetch_one(year, month, day, sample_idx)
        lat, lon, time = sample[LATITUDE], sample[LONGITUDE], sample[TIME]

        time_window_seconds = self.settings['training']['time_window_secs']

        lat += metres_to_degrees(np.random.normal(0, self.settings['training']['random_shift_variance_km']), lat)[0]
        lon += metres_to_degrees(np.random.normal(0, self.settings['training']['random_shift_variance_km']), lat)[1]

        # we have the centre of the obs, at a time. Now form the bounds of the grid.
        xy_window_size_m = self.settings['training']['xy_window_size_km'] * 1000
        z_window_size_m = self.settings['training']['z_window_size_km'] * 1000
        alt_min_m = self.settings['training']['z_min_km'] * 1000
        alt_max_m = alt_min_m + z_window_size_m

        lat_min = lat - metres_to_degrees(xy_window_size_m / 2, lat)[0]
        lat_max = lat + metres_to_degrees(xy_window_size_m / 2, lat)[0]
        lon_min = lon - metres_to_degrees(xy_window_size_m / 2, lat)[1]
        lon_max = lon + metres_to_degrees(xy_window_size_m / 2, lat)[1]
        
        # now fetch all the data within the bounds and time window
        # the bug is in this function - empty arrays are being returned.
        print("Fetching amdar")
        df = self.amdar.fetch_for_batch((lat_min, lat_max), (lon_min, lon_max), (alt_min_m, alt_max_m), time, time_window_seconds)
        print(f"Fetched {len(df)} amdar samples")
        print("Fetching ecmwf")
        ecmwf = self.ecmwf.fetch_for_batch((lat_min, lat_max), (lon_min, lon_max), time, time_window_seconds)
        print(f"Fetched ecmwf data with shape {ecmwf[WIND_U].shape} for u and {ecmwf[WIND_V].shape} for v")
        
        # convert amdar lat/lon/alt to grid coordinates
        amdar_x_grid, amdar_y_grid, amdar_z_grid = self.offgrid_coords_to_mesh(
            df[LATITUDE],
            df[LONGITUDE],
            df[ALTITUDE],
            (lat_min, lat_max),
            (lon_min, lon_max),
            (alt_min_m, alt_max_m),
            x_size,
            y_size,
            z_size,
        )
        ecmwf_x_grid, ecmwf_y_grid, ecmwf_z_grid = self.gridded_coords_to_mesh(
            ecmwf,
            (lat_min, lat_max),
            (lon_min, lon_max),
            (alt_min_m, alt_max_m),
            x_size,
            y_size,
            z_size,
        )

        amdar_tensor = torch.zeros(2, x_size, y_size, z_size, dtype=torch.float32, device=self.device)
        if len(df) > 0:
            amdar_u = torch.as_tensor(df[WIND_U].to_numpy(dtype=np.float32), device=self.device)
            amdar_v = torch.as_tensor(df[WIND_V].to_numpy(dtype=np.float32), device=self.device)
            x_idx = torch.as_tensor(amdar_x_grid, dtype=torch.long, device=self.device)
            y_idx = torch.as_tensor(amdar_y_grid, dtype=torch.long, device=self.device)
            z_idx = torch.as_tensor(amdar_z_grid, dtype=torch.long, device=self.device)

            amdar_tensor[0, x_idx, y_idx, z_idx] = amdar_u
            amdar_tensor[1, x_idx, y_idx, z_idx] = amdar_v

        # second are the gridded ecmwf points in the xarray ds
        ecmwf_tensor = torch.zeros(2, x_size, y_size, z_size, dtype=torch.float32, device=self.device)
        ecmwf_u = ecmwf[WIND_U]
        ecmwf_v = ecmwf[WIND_V]
        if TIME in ecmwf_u.dims:
            ecmwf_u = ecmwf_u.mean(dim=TIME)
            ecmwf_v = ecmwf_v.mean(dim=TIME)
        if all(dim in ecmwf_u.dims for dim in (ALTITUDE, LATITUDE, LONGITUDE)):
            ecmwf_u = ecmwf_u.transpose(ALTITUDE, LATITUDE, LONGITUDE)
            ecmwf_v = ecmwf_v.transpose(ALTITUDE, LATITUDE, LONGITUDE)

        ecmwf_u_vals = torch.as_tensor(np.ravel(ecmwf_u.values).astype(np.float32), device=self.device)
        ecmwf_v_vals = torch.as_tensor(np.ravel(ecmwf_v.values).astype(np.float32), device=self.device)
        ex_idx = torch.as_tensor(np.ravel(ecmwf_x_grid), dtype=torch.long, device=self.device)
        ey_idx = torch.as_tensor(np.ravel(ecmwf_y_grid), dtype=torch.long, device=self.device)
        ez_idx = torch.as_tensor(np.ravel(ecmwf_z_grid), dtype=torch.long, device=self.device)

        n = min(ecmwf_u_vals.numel(), ex_idx.numel(), ey_idx.numel(), ez_idx.numel())
        if n > 0:
            ecmwf_tensor[0, ex_idx[:n], ey_idx[:n], ez_idx[:n]] = ecmwf_u_vals[:n]
            ecmwf_tensor[1, ex_idx[:n], ey_idx[:n], ez_idx[:n]] = ecmwf_v_vals[:n]

        return amdar_tensor, ecmwf_tensor

    def offgrid_coords_to_mesh(self, lat, lon, alt, lat_bounds, lon_bounds, alt_bounds, x_size, y_size, z_size):
        # convert lat/lon/alt to x/y/z in the grid coordinates based on the centre of the grid and the window size
        # lat, lon, alt are columns from the dataframe.
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        alt_min, alt_max = alt_bounds

        lat_vals = np.asarray(lat, dtype=np.float64)
        lon_vals = np.asarray(lon, dtype=np.float64)
        alt_vals = np.asarray(alt, dtype=np.float64)

        x = (lon_vals - lon_min) / max(lon_max - lon_min, 1e-12)
        y = (lat_vals - lat_min) / max(lat_max - lat_min, 1e-12)
        z = (alt_vals - alt_min) / max(alt_max - alt_min, 1e-12)

        x_grid = np.clip(np.rint(x * (x_size - 1)), 0, x_size - 1).astype(np.int64)
        y_grid = np.clip(np.rint(y * (y_size - 1)), 0, y_size - 1).astype(np.int64)
        z_grid = np.clip(np.rint(z * (z_size - 1)), 0, z_size - 1).astype(np.int64)

        return x_grid, y_grid, z_grid
    
    def gridded_coords_to_mesh(self, ds, lat_bounds, lon_bounds, alt_bounds, x_size, y_size, z_size):
        # convert an xarray dataset with lat/lon/alt coordinates to x/y/z in the grid coordinates based on the centre of the grid and the window size
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        alt_min, alt_max = alt_bounds

        lat = ds[LATITUDE].values
        lon = ds[LONGITUDE].values
        alt = ds[ALTITUDE].values

        x_1d = np.clip(np.rint((lon - lon_min) / max(lon_max - lon_min, 1e-12) * (x_size - 1)), 0, x_size - 1).astype(np.int64)
        y_1d = np.clip(np.rint((lat - lat_min) / max(lat_max - lat_min, 1e-12) * (y_size - 1)), 0, y_size - 1).astype(np.int64)
        z_1d = np.clip(np.rint((alt - alt_min) / max(alt_max - alt_min, 1e-12) * (z_size - 1)), 0, z_size - 1).astype(np.int64)

        z_grid, y_grid, x_grid = np.meshgrid(z_1d, y_1d, x_1d, indexing='ij')

        return x_grid, y_grid, z_grid
        
