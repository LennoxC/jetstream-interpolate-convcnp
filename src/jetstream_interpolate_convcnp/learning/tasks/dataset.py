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

        # get a dictionary of date to number of samples for that date
        self.date_nsamples = amdar_ds.groupby(DATE).size().compute()
        self.total_nsamples = self.date_nsamples.sum()
        
        self.train_size = int(self.total_nsamples * pct)
        self.val_size = self.total_nsamples - self.train_size

        # find the date threshold for the training set
        cumulative_samples = self.date_nsamples.cumsum()
        train_date_threshold = cumulative_samples[cumulative_samples >= self.train_size].index[0]

        # create a list of dates for the training set including the reduced number of samples for the threshold date
        train_dates = {}
        for date, nsamples in self.date_nsamples.items():
            if date < train_date_threshold:
                train_dates[date] = nsamples
            elif date == train_date_threshold:
                remaining_samples = self.train_size - cumulative_samples[cumulative_samples < self.train_size].sum()
                train_dates[date] = remaining_samples
                break
            else:
                break
        
        # do the same for the test set
        test_dates = {}
        for date, nsamples in self.date_nsamples.items():
            if date < train_date_threshold:
                continue
            elif date == train_date_threshold:
                remaining_samples = self.total_nsamples - self.train_size
                test_dates[date] = remaining_samples
            else:
                test_dates[date] = nsamples

        self.train_dates = train_dates
        self.test_dates = test_dates


