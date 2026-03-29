import xarray as xr
import numpy as np

class ECMWFInterface:
    def __init__(self, settings):
        self.settings = settings
        self.ecmwf_path = settings['paths']['process_ecmwf_path_base']

    def fetch_for_batch(self, lat_range, lon_range, timestamp_end, time_window_secs):
        # fetch the ECMWF data for the given lat/lon box and time window
        # this is a placeholder implementation that just returns an empty dataset
        # eventually this will read from the ECMWF data on disk and return the relevant subset of the data
        
        ds = xr.load_dataset(self.ecmwf_path)
        timestamp_start = timestamp_end - np.timedelta64(time_window_secs, 's')
        ds = ds.sel(time=slice(timestamp_start, timestamp_end), lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
        
        return ds