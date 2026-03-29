from jetstream_interpolate_convcnp.utils.constants import LONGITUDE, LATITUDE, DATE, TIME, ALTITUDE, WIND_U, WIND_V
import dask.dataframe as dd
import numpy as np
import pandas as pd

class AMDARInterface:
    def __init__(self, settings):
        self.settings = settings
        self.save_path = self.settings['paths']['process_amdar_path_base']

    def fetch_one(self, date, index):
        date = pd.to_datetime(date)
        # fetch the partition for the given date, then return the row at the given index
        partition_path = f"{self.save_path}date={date.strftime('%Y-%m-%d')} 00:00:00"
        df = dd.read_parquet(partition_path).compute()
        if index >= len(df):
            raise IndexError(f"Index {index} out of range for date {date} with {len(df)} samples")
        return df.iloc[index]
    
    def fetch_partition(self, date, lat, lon):
        # fetch the partition for the given date, lat, and lon
        date = pd.to_datetime(date)
        partition_path = f"{self.save_path}date={date.strftime('%Y-%m-%d')} 00:00:00/{LATITUDE}_int={int(np.floor(lat))}/{LONGITUDE}_int={int(np.floor(lon))}"
        df = dd.read_parquet(partition_path).compute()
        return df
    
    def fetch_for_batch(self, lat_range, lon_range, alt_range, timestamp_end, time_window_secs):
        # fetch all samples within the given lat/lon box and time window.
        # use the partitioning to minimize the amount of data read from disk.
        # greacefully handle when lon crosses from 359 to 0
        timestamp_end = pd.to_datetime(timestamp_end)
        timestamp_start = timestamp_end - np.timedelta64(time_window_secs, 's')
        
        # the timestamps we are fetching over could span multiple dates
        date_end = timestamp_end.date()
        date_start = timestamp_start.date()

        lat_partitions = range(int(np.floor(lat_range[0])), int(np.floor(lat_range[1])) + 1)
        lon_partitions = range(int(np.floor(lon_range[0])), int(np.floor(lon_range[1])) + 1)
        alt_partitions = range(int(np.floor(alt_range[0])), int(np.floor(alt_range[1])) + 1)

        # build all the paths we need to read from
        paths = []
        for date in pd.date_range(date_start, date_end):
            date = pd.to_datetime(date)
            for lat in lat_partitions:
                for lon in lon_partitions:
                    paths.append(f"{self.save_path}date={date.strftime('%Y-%m-%d')} 00:00:00/{LATITUDE}_int={lat}/{LONGITUDE}_int={lon}/part.*.parquet")
        
        # read all the data in a dask dataframe
        df = dd.read_parquet(paths)
        # filter to the exact lat/lon and time range we want
        df = df[
            (df[LATITUDE] >= lat_range[0]) & (df[LATITUDE] <= lat_range[1]) &
            (df[LONGITUDE] >= lon_range[0]) & (df[LONGITUDE] <= lon_range[1]) &
            (df[TIME] >= timestamp_start) & (df[TIME] <= timestamp_end) &
            (df[ALTITUDE] >= alt_range[0]) & (df[ALTITUDE] <= alt_range[1])
        ]

        return df.compute()
