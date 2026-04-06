from jetstream_interpolate_convcnp.utils.constants import LONGITUDE, LATITUDE, DATE, TIME, ALTITUDE, WIND_U, WIND_V
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr

class AMDARInterface:
    def __init__(self, settings, normalize=True):
        self.settings = settings
        self.save_path = self.settings['paths']['process_amdar_path_base']
        self.normalize = normalize

    def fetch_one(self, year, month, day, index):
        # fetch the partition for the given date, then return the row at the given index
        partition_path = f"{self.save_path}year={year}/month={month}/day={day}"
        df = dd.read_parquet(partition_path).compute()
        if index >= len(df):
            raise IndexError(f"Index {index} out of range for date {year}-{month}-{day} with {len(df)} samples")
        return df.iloc[index]
    
    def fetch_partition(self, year, month, day, lat, lon):
        # fetch the partition for the given date, lat, and lon
        partition_path = f"{self.save_path}year={year}/month={month}/day={day}/{LATITUDE}_int={int(np.floor(lat))}/{LONGITUDE}_int={int(np.floor(lon))}"
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

        # build all the paths we need to read from
        paths = []
        for date in pd.date_range(date_start, date_end):
            date = pd.to_datetime(date)
            for lat in lat_partitions:
                for lon in lon_partitions:
                    paths.append(f"{self.save_path}year={date.year}/month={date.month}/day={date.day}/{LATITUDE}_int={lat}/{LONGITUDE}_int={lon}/part.*.parquet")
        
        # read all the data in a dask dataframe
        df = dd.read_parquet(paths)
        # filter to the exact lat/lon and time range we want
        df = df[
            (df[LATITUDE] >= lat_range[0]) & (df[LATITUDE] <= lat_range[1]) &
            (df[LONGITUDE] >= lon_range[0]) & (df[LONGITUDE] <= lon_range[1]) &
            (df[TIME] >= timestamp_start) & (df[TIME] <= timestamp_end) &
            (df[ALTITUDE] >= alt_range[0]) & (df[ALTITUDE] <= alt_range[1])
        ]

        # then we need to normalize the amdar data to the same scale as the ecmwf data using the precomputed norm parameters
        if self.normalize:
            df = self.normalize_to_ecmwf(df, lat_range, lon_range, timestamp_end)
        else:
            df = df.compute()

        return df
    
    def normalize_to_ecmwf(self, df, lat_range, lon_range, timestamp_end):
        # amdar data is not normalized when saved

        # find the pressure levels from ecmwf observation data. ECMWF has an altitude column and a pressure index. 
        # find the closest pressure level for each amdar observation and add it as a column to the amdar dataframe
        ecmwf_data_path = self.settings['paths']['process_ecmwf_path_base']
        ecmwf_data_df = (xr.open_dataset(ecmwf_data_path)
                            .sel(time=timestamp_end, method='pad')
                            .sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
                            [['time', 'pressure_level', 'lat', 'lon', 'altitude', 'u', 'v']]
                            .to_dataframe()
                            .reset_index())[['pressure_level', 'lat', 'lon', 'altitude']]

        # load the ecmwf norm parameters and apply them to the amdar data so it's on the same scale as the ecmwf data

        ecmwf_norm_params = f"{self.settings['paths']['ecmwf_norm_params_path']}params.nc"
        ecmwf_ds = (xr.open_dataset(ecmwf_norm_params)
                    .sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
                    [['u_mean', 'u_std', 'v_mean', 'v_std']])

        ecmwf_norm_df = (ecmwf_ds.sel(lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
                            [['u_mean', 'u_std', 'v_mean', 'v_std']]
                            .to_dataframe()
                            .reset_index())[['pressure_level', 'lat', 'lon', 'u_mean', 'u_std', 'v_mean', 'v_std']]

        # merge df and find the nearest ecmwf pressure level for each amdar obs based on nearest altitude. Altitude will not match exactly.

        # df: time, date, lat, lon, altitude, u, v, year, month, day, lat_int, lon_int
        # ecmwf_data_df: pressure_level, lat, lon, altitude
        # ecmwf_norm_df: pressure_level, lat, lon, u_mean, u_std, v_mean, v_std

        # desired:
        # df: time, date, lat, lon, altitude, u, v | where u and v are normalized using the nearest ecmwf_norm_df params based on nearest pressure level, nearest lat, and nearest lon. lat/lon is every 0.5 of a degree

        df = df.compute()

        ecmwf_levels = ecmwf_data_df[['pressure_level', 'lat', 'lon', 'altitude']].copy()
        df['lat_grid'] = np.round(df['lat'] * 2) / 2
        df['lon_grid'] = np.round(df['lon'] * 2) / 2
        ecmwf_levels['lat_grid'] = np.round(ecmwf_levels['lat'] * 2) / 2
        ecmwf_levels['lon_grid'] = np.round(ecmwf_levels['lon'] * 2) / 2
        
        df = df.reset_index(drop=True)
        df['obs_id'] = np.arange(len(df))

        merged = df.merge(
            ecmwf_levels,
            on=['lat_grid', 'lon_grid'],
            suffixes=('', '_ecmwf')
        )
        merged['alt_diff'] = np.abs(merged['altitude'] - merged['altitude_ecmwf'])

        # pick closest level per observation
        idx = merged.groupby('obs_id')['alt_diff'].idxmin()
        merged = merged.loc[idx].copy()

        ecmwf_norm_df['lat_grid'] = np.round(ecmwf_norm_df['lat'] * 2) / 2
        ecmwf_norm_df['lon_grid'] = np.round(ecmwf_norm_df['lon'] * 2) / 2

        merged = merged.merge(
            ecmwf_norm_df,
            on=['pressure_level', 'lat_grid', 'lon_grid'],
            how='left'
        )

        merged['u'] = (merged['u'] - merged['u_mean']) / merged['u_std']
        merged['v'] = (merged['v'] - merged['v_mean']) / merged['v_std']

        return merged[['time', 'date', 'lat_x', 'lon_x', 'altitude', 'u', 'v', 'year', 'month', 'day', 'lat_int', 'lon_int']].rename(columns={'lat_x': 'lat', 'lon_x': 'lon'})
