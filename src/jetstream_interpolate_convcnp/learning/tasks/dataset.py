from jetstream_interpolate_convcnp.processing.amdar.AMDARProcessor import AMDARProcessor
from jetstream_interpolate_convcnp.utils.constants import DATE, TIME, LATITUDE, LONGITUDE, WIND_U, WIND_V
import dask.dataframe as dd
import numpy as np

class SampleSettings:
    def __init__(self, settings):
        self.settings = settings

        # set later
        self.date_nsamples = None
        self.train_dates = None
        self.test_dates = None
        self.train_size = None
        self.val_size = None

        self.setup_date_distributions()

    def setup_date_distributions(self):
        pct = self.settings['training']['train_dates_pct']
        amdar_dataset_path = self.settings['paths']['process_amdar_path_base']

        # amdar dataset is parquet files partitioned by date then lat/lon
        amdar_ds = dd.read_parquet(amdar_dataset_path)

        required_cols = ['year', 'month', 'day']
        missing_cols = [col for col in required_cols if col not in amdar_ds.columns]
        if missing_cols:
            raise ValueError(f"Dataset must include date-part columns {required_cols}; missing {missing_cols}")

        # Count readings per (year, month, day) and keep chronological order.
        self.date_nsamples = amdar_ds.groupby(required_cols).size().compute().sort_index()
        self.total_nsamples = int(self.date_nsamples.sum())

        if self.total_nsamples == 0:
            raise ValueError("Dataset has no samples; cannot build train/test date distributions")

        self.train_size = int(self.total_nsamples * pct)
        self.val_size = self.total_nsamples - self.train_size

        train_dates = {}
        test_dates = {}
        remaining_train = self.train_size

        # Split each day bucket between train and test so threshold days are handled correctly.
        for date, nsamples in self.date_nsamples.items():
            nsamples = int(nsamples)

            train_nsamples = min(nsamples, remaining_train)
            test_nsamples = nsamples - train_nsamples

            if train_nsamples > 0:
                train_dates[date] = train_nsamples
            if test_nsamples > 0:
                test_dates[date] = test_nsamples

            remaining_train -= train_nsamples

        self.train_dates = train_dates
        self.test_dates = test_dates


