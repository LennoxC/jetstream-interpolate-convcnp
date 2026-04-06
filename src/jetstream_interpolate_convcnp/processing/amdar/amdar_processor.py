import dask.dataframe as dd
import numpy as np
from jetstream_interpolate_convcnp.utils.constants import (
    ALTITUDE, LATITUDE, LONGITUDE, TIME, WIND_U, WIND_V, DATE
)
from jetstream_interpolate_convcnp.processing.ecmwf.ecmwf_processor import ECMWFProcessor
from jetstream_interpolate_convcnp.utils.settings import settings

class AMDARProcessor:
    def __init__(self, partition_cols=None, reduce_time=False, **kwargs):
        self.dataset_path = settings['paths']['amdar_load_path']
        self.save_path = settings['paths']['process_amdar_path_base']

        self.partition_cols = partition_cols
        self.reduce_time = reduce_time
        self.pandas_kwargs = kwargs
        
    def load(self):
        self.ds = dd.read_csv(self.dataset_path, 
                              header=None,
                              names=[str(i) for i in range(1, 36)],
                              dtype='object',
                              assume_missing=True,
                              on_bad_lines='skip',
                              blocksize="64MB",
                              **self.pandas_kwargs)

    def preprocess(self):
        # Clean column names
        self.ds.columns = [str(c).strip() for c in self.ds.columns]

        source_to_target = {
            '1': 'year',
            '2': 'month',
            '3': 'day',
            '4': 'hour',
            '5': 'minute',
            '6': 'second',
            '14': LATITUDE,
            '15': LONGITUDE,
            '16': ALTITUDE,
            '25': 'wind_direction',
            '26': 'wind_speed',
        }

        self.ds = self.ds.rename(columns=source_to_target)

        required_cols = ['year', 'month', 'day', 'hour', LATITUDE, LONGITUDE, ALTITUDE, 'wind_speed', 'wind_direction']

        # Convert to numeric FIRST
        for col in required_cols + ['minute', 'second']:
            self.ds[col] = dd.to_numeric(self.ds[col], errors='coerce')

        time_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']

        for col in time_cols:
            self.ds[col] = dd.to_numeric(self.ds[col], errors='coerce')

        for col in ['minute', 'second']:
            self.ds[col] = self.ds[col].mask(self.ds[col] == -9999999, 0)  # replace sentinel with 0 for time components
            self.ds[col] = self.ds[col].mask(self.ds[col] == -9999999, np.nan)

        self.ds['minute'] = self.ds['minute'].fillna(0)
        self.ds['second'] = self.ds['second'].fillna(0)
        
        # Drop invalid rows
        self.ds = self.ds.dropna(subset=required_cols)

        for col in time_cols:
            self.ds[col] = self.ds[col].astype('int64')

        timestamp_str = (self.ds['year'].astype(str) + '-' +
                         self.ds['month'].astype(str).str.zfill(2) + '-' +
                         self.ds['day'].astype(str).str.zfill(2) + ' ' +
                         self.ds['hour'].astype(str).str.zfill(2) + ':' +
                         self.ds['minute'].astype(str).str.zfill(2) + ':' +
                         self.ds['second'].astype(str).str.zfill(2))

        self.ds[TIME] = dd.to_datetime(timestamp_str)
        self.ds[DATE] = self.ds[TIME].dt.floor('D')

        # Wind components
        self.ds[WIND_U] = -self.ds['wind_speed'] * np.sin(np.radians(self.ds['wind_direction']))
        self.ds[WIND_V] = -self.ds['wind_speed'] * np.cos(np.radians(self.ds['wind_direction']))

        # Normalize longitude
        self.ds[LONGITUDE] = self.ds[LONGITUDE] % 360

        # cast lat and lon to int for partitioning. These cols are for partitioning only
        # coarsen lat/lon for bigger partitions
        grid_coarsening = 2

        self.ds[f"coarse_{LATITUDE}"] = (np.floor(self.ds[LATITUDE] / grid_coarsening)*grid_coarsening).astype('int64')
        self.ds[f"coarse_{LONGITUDE}"] = (np.floor(self.ds[LONGITUDE] / grid_coarsening)*grid_coarsening).astype('int64')

        # further feature engineering
        self.ds['log_altitude'] = np.log(self.ds[ALTITUDE])

        altitude_band_width_m = 2000
        self.ds['altitude_band'] = np.floor(self.ds[ALTITUDE] / altitude_band_width_m).astype('int64')
        
        # normalize to ecmwf normalization params
        self.ds = self.normalize(self.ds, normalize=True)

        target_cols = [TIME, DATE, 'year', 'month', 'day', LATITUDE, LONGITUDE, f"coarse_{LATITUDE}", f"coarse_{LONGITUDE}", ALTITUDE, WIND_U, WIND_V, 'altitude_band', 'log_altitude', f"{WIND_U}_normed", f"{WIND_V}_normed"]

        self.ds = self.ds[target_cols]

        if self.ds.map_partitions(len).sum().compute() == 0:
            raise ValueError("No data left after preprocessing — check filters.")
    
    def normalize(self, df, normalize=True):
        group_cols = ['altitude_band', 'coarse_lat', 'coarse_lon']
        
        ecmwf = ECMWFProcessor()
        df_norm_pd = ecmwf.load_norm_params().compute()

        def normalize_partition(pdf, norm_df):
            pdf = pdf.merge(norm_df, on=group_cols, how='left')

            if normalize:
                pdf['u_normed'] = (pdf['u'] - pdf['u_mean']) / pdf['u_std']
                pdf['v_normed'] = (pdf['v'] - pdf['v_mean']) / pdf['v_std']

            if not normalize:
                pdf['u'] = pdf[f"{WIND_U}_normed"] * pdf['u_std'] + pdf['u_mean']
                pdf['v'] = pdf[f"{WIND_V}_normed"] * pdf['v_std'] + pdf['v_mean']

            return pdf
        
        meta = df._meta.copy()
        meta['u_mean'] = np.float64()
        meta['u_std'] = np.float64()
        meta['v_mean'] = np.float64()
        meta['v_std'] = np.float64()
        meta['u_normed'] = np.float32()
        meta['v_normed'] = np.float32()

        return df.map_partitions(normalize_partition, df_norm_pd, meta=meta)
    
    # def normalize(self, df):
    #     params = self.load_ecmwf_norm_params()
    #     
    #     df = df.merge(params, left_on=['altitude_band', f"coarse_{LATITUDE}", f"coarse_{LONGITUDE}"], right_on=['altitude_band', 'coarse_lat', 'coarse_lon'], how='left')
    # 
    #     df[f"{WIND_U}_normed"] = (df[WIND_U] - df['u_mean']) / df['u_std']
    #     df[f"{WIND_V}_normed"] = (df[WIND_V] - df['v_mean']) / df['v_std']
    # 
    #     return df

    # def unnormalize(self, df):
    #     params = self.load_ecmwf_norm_params()
    #  
    #     df = df.merge(params, left_on=['altitude_band', f"coarse_{LATITUDE}", f"coarse_{LONGITUDE}"], right_on=['altitude_band', 'coarse_lat', 'coarse_lon'], how='left')
    # 
    #     df[WIND_U] = df[f"{WIND_U}_normed"] * df['u_std'] + df['u_mean']
    #     df[WIND_V] = df[f"{WIND_V}_normed"] * df['v_std'] + df['v_mean']
    # 
    #     return df

    def run(self):
        self.load()
        self.preprocess()

        # Ensure execution happens
        self.ds.to_parquet(
            self.save_path,
            partition_on=self.partition_cols if self.partition_cols else None,
            schema={
                TIME: 'timestamp[s]',
                DATE: 'timestamp[s]',
                'year': 'int32',
                'month': 'int32',
                'day': 'int32',
                LATITUDE: 'double',
                LONGITUDE: 'double',
                f"coarse_{LATITUDE}": 'int32',
                f"coarse_{LONGITUDE}": 'int32',
                ALTITUDE: 'double',
                WIND_U: 'double',
                WIND_V: 'double',
                'altitude_band': 'int32',
                'log_altitude': 'double',
                f"{WIND_U}_normed": 'double',
                f"{WIND_V}_normed": 'double'
            },
            write_index=False,
            compute=True
        )

    def output_dataset_path(self):
        return self.save_path