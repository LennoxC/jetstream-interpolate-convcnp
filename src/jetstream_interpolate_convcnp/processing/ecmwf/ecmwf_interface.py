from jetstream_interpolate_convcnp.utils.constants import TIME, TIME, LONGITUDE, LATITUDE, ALTITUDE

import pandas as pd
import xarray as xr
import numpy as np
import dask.dataframe as dd

class ECMWFInterface:
    def __init__(self, settings):
        self.settings = settings
        self.ecmwf_path = settings['paths']['process_ecmwf_path_base'] + "altitude_band=*/coarse_lat=*/coarse_lon=*/part.*.parquet"

    def fetch_for_batch(self, lat_range, lon_range, timestamp_end, time_window_secs):
        # fetch the ECMWF data for the given lat/lon box and time window
        # follow the inspiration from AMDARInterface.fetch_for_batch to minimize the amount of data read from disk
        timestamp_end = pd.to_datetime(timestamp_end)
        timestamp_start = timestamp_end - np.timedelta64(time_window_secs, 's')

        date_end = timestamp_end.date()
        date_start = timestamp_start.date()

        lat_partitions = range(int(np.floor(lat_range[0])), int(np.floor(lat_range[1])) + 1)
        lon_partitions = range(int(np.floor(lon_range[0])), int(np.floor(lon_range[1])) + 1)

        paths = []
        for date in pd.date_range(date_start, date_end):
            date = pd.to_datetime(date)
            for lat in lat_partitions:
                for lon in lon_partitions:
                    paths.append(f"{self.settings['paths']['process_ecmwf_path_base']}altitude_band=*/coarse_lat={lat}/coarse_lon={lon}/part.*.parquet")
        
        df = dd.read_parquet(paths)

        if (len(df) == 0):
            raise ValueError(f"No ECMWF data found for the given lat/lon box and time window. Paths searched: {paths}")
        
        df = df[
            (df[LATITUDE] >= lat_range[0]) & (df[LATITUDE] <= lat_range[1]) &
            (df[LONGITUDE] >= lon_range[0]) & (df[LONGITUDE] <= lon_range[1]) &
            (df[TIME] >= timestamp_start) & (df[TIME] <= timestamp_end)
        ]

        df = df.compute()

        return df