import xarray as xr
from jetstream_interpolate_convcnp.utils.constants import ALTITUDE, GEOPOTENTIAL_HEIGHT, LATITUDE, LONGITUDE, PRESSURE_LEVEL, TIME, WIND_U, WIND_V, g

class ERA5Processor:
    def __init__(self, dataset_path, chunking_in, chunking_out, normalizer=None, reduce_time=False):
        self.dataset_path = dataset_path
        self.chunking_in = chunking_in
        self.chunking_out = chunking_out
        self.normalizer = normalizer
        self.reduce_time = reduce_time
        self.save_path = None
        

    def load(self):
        self.ds = xr.open_mfdataset(self.dataset_path, chunks=self.chunking_in, combine="by_coords")

        if self.reduce_time:
            self.ds = self.ds.isel(valid_time=slice(0, 24))

        if self.normalizer is not None:
            print("Fitting normalizer on ERA5 dataset...")
            self.normalizer.fit(self.ds)
            self.ds = self.normalizer.normalize(self.ds)
            self.normalizer.save()

    def preprocess(self):
        # this isn't able to generalize to all ERA5 datasets

        # rename coordinates to standard names
        self.ds = self.ds.rename({'valid_time': TIME, "latitude": LATITUDE, "longitude": LONGITUDE, "pressure_level": PRESSURE_LEVEL})

        # rename variables to standard names
        self.ds = self.ds.rename({'u': WIND_U, 'v': WIND_V, 'z': GEOPOTENTIAL_HEIGHT})

        # convert geopotential height to actual height in meters
        self.ds[ALTITUDE] = self.ds[GEOPOTENTIAL_HEIGHT] / g
        self.ds = self.ds.drop_vars(GEOPOTENTIAL_HEIGHT)

    def initialize(self, save_path=None):
        self.load()
        self.preprocess()

        self.save_path = save_path

        if save_path is not None and self.chunking_out is not None:
            from jetstream_interpolate_convcnp.processing.chunk_netcdf import chunk_and_save
            chunk_and_save(self.ds, save_path, chunk_size=self.chunking_out)

        if save_path is not None and self.chunking_out is None:
            self.ds.to_netcdf(save_path)
        
        