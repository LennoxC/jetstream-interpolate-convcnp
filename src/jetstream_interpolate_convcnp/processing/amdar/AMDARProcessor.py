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
        for col in required_cols:
            self.ds[col] = dd.to_numeric(self.ds[col], errors='coerce')

        # Replace sentinel safely
        for col in required_cols:
            self.ds[col] = self.ds[col].mask(self.ds[col] == -9999999, np.nan)

        # Drop invalid rows BEFORE datetime
        self.ds = self.ds.dropna(subset=required_cols)

        # Fast numeric datetime (no warnings)
        self.ds['minute'] = self.ds['minute'].fillna(0)
        self.ds['second'] = self.ds['second'].fillna(0)

        time_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']

        # Convert safely
        for col in time_cols:
            self.ds[col] = dd.to_numeric(self.ds[col], errors='coerce')

        
        self.ds['minute'] = self.ds['minute'].fillna(0)
        self.ds['second'] = self.ds['second'].fillna(0)

        self.ds = self.ds.dropna(subset=['year', 'month', 'day', 'hour'])

        for col in time_cols:
            self.ds[col] = self.ds[col].astype('int64')

        # Build datetime
        self.ds[TIME] = (
            dd.to_datetime(
                self.ds['year'] * 10000 + self.ds['month'] * 100 + self.ds['day'],
                format='%Y%m%d',
                errors='coerce'
            )
            + dd.to_timedelta(self.ds['hour'], unit='h')
            + dd.to_timedelta(self.ds['minute'], unit='m')
            + dd.to_timedelta(self.ds['second'], unit='s')
        )

        self.ds = self.ds.dropna(subset=[TIME])
        self.ds[DATE] = self.ds[TIME].dt.floor('D')

        # Wind components
        self.ds[WIND_U] = -self.ds['wind_speed'] * np.sin(np.radians(self.ds['wind_direction']))
        self.ds[WIND_V] = -self.ds['wind_speed'] * np.cos(np.radians(self.ds['wind_direction']))

        # Normalize longitude
        self.ds[LONGITUDE] = self.ds[LONGITUDE] % 360

        # cast lat and lon to int for partitioning. These cols are for partitioning only
        self.ds[f"{LATITUDE}_int"] = np.floor(self.ds[LATITUDE]).astype('int64')
        self.ds[f"{LONGITUDE}_int"] = np.floor(self.ds[LONGITUDE]).astype('int64')

        if self.reduce_time:
            min_date = self.ds[DATE].min()
            self.ds = self.ds[self.ds[DATE] == min_date]

        target_cols = [TIME, DATE, LATITUDE, LONGITUDE, f"{LATITUDE}_int", f"{LONGITUDE}_int", ALTITUDE, WIND_U, WIND_V]

        self.ds = self.ds[target_cols]
        self.ds = self.ds.dropna(subset=target_cols)

        if self.ds.map_partitions(len).sum().compute() == 0:
            raise ValueError("No data left after preprocessing — check filters.")
    
    def initialize(self, save_path=None):
        self.load()
        self.preprocess()

        if save_path is None:
            raise ValueError("You must provide a save_path")
        
        """
        ValueError: Failed to convert partition to expected pyarrow schema:
            `ArrowInvalid('Float value -33.883000 was truncated converting to int64', 'Conversion failed for column lat with type Float64')`

        Expected partition schema:
            time: timestamp[us]
            date: timestamp[us]
            lat: int64
            lon: int64
            altitude: int64
            u: double
            v: double

        Received partition schema:
            time: timestamp[us]
            date: timestamp[us]
            lat: double
            lon: double
            altitude: double
            u: double
            v: double

        This error *may* be resolved by passing in schema information for
        the mismatched column(s) using the `schema` keyword in `to_parquet`.
        """

        # Ensure execution happens
        self.ds.to_parquet(
            save_path,
            partition_on=self.partition_cols if self.partition_cols else None,
            write_index=False,
            compute=True
        )