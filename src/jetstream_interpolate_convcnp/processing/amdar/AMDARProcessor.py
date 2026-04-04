import dask.dataframe as dd
import numpy as np
from jetstream_interpolate_convcnp.utils.constants import (
    ALTITUDE, LATITUDE, LONGITUDE, TIME, WIND_U, WIND_V, DATE
)

class AMDARProcessor:
    def __init__(self, dataset_path, partition_cols=None, reduce_time=False, **kwargs):
        self.dataset_path = dataset_path
        self.partition_cols = partition_cols
        self.reduce_time = reduce_time
        self.pandas_kwargs = kwargs
        self.save_path = None

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
            '26': 'wind_speed',
            '27': 'wind_direction'
        }

        self.ds = self.ds.rename(columns=source_to_target)

        required_cols = [
            'year', 'month', 'day', 'hour',
            LATITUDE, LONGITUDE, ALTITUDE,
            'wind_speed', 'wind_direction'
        ]

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

        self.ds[TIME] = dd.to_datetime(
            timestamp_str
        )

        self.ds[DATE] = self.ds[TIME].dt.floor('D')

        # Wind components
        self.ds[WIND_U] = -self.ds['wind_speed'] * np.sin(np.radians(self.ds['wind_direction']))
        self.ds[WIND_V] = -self.ds['wind_speed'] * np.cos(np.radians(self.ds['wind_direction']))

        # Normalize longitude
        self.ds[LONGITUDE] = self.ds[LONGITUDE] % 360

        # cast lat and lon to int for partitioning. These cols are for partitioning only
        self.ds[f"{LATITUDE}_int"] = np.floor(self.ds[LATITUDE]).astype('int64')
        self.ds[f"{LONGITUDE}_int"] = np.floor(self.ds[LONGITUDE]).astype('int64')

        #if self.reduce_time:
        #    min_date = self.ds[DATE].min()
        #    self.ds = self.ds[self.ds[DATE] == min_date]

        target_cols = [TIME, DATE, 'year', 'month', 'day', LATITUDE, LONGITUDE, f"{LATITUDE}_int", f"{LONGITUDE}_int", ALTITUDE, WIND_U, WIND_V]

        self.ds = self.ds[target_cols]
        #self.ds = self.ds.dropna(subset=target_cols)

        if self.ds.map_partitions(len).sum().compute() == 0:
            raise ValueError("No data left after preprocessing — check filters.")
    
    def initialize(self, save_path=None):
        self.load()
        self.preprocess()

        self.save_path = save_path

        if save_path is None:
            raise ValueError("You must provide a save_path")

        # Ensure execution happens
        self.ds.to_parquet(
            save_path,
            partition_on=self.partition_cols if self.partition_cols else None,
            schema={
                TIME: 'timestamp[s]',
                DATE: 'timestamp[s]',
                'year': 'int64',
                'month': 'int64',
                'day': 'int64',
                LATITUDE: 'double',
                LONGITUDE: 'double',
                f"{LATITUDE}_int": 'int64',
                f"{LONGITUDE}_int": 'int64',
                ALTITUDE: 'double',
                WIND_U: 'double',
                WIND_V: 'double'
            },
            write_index=False,
            compute=True
        )

    def output_dataset_path(self):
        return self.save_path