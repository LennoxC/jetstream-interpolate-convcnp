from jetstream_interpolate_convcnp.utils.constants import ALTITUDE, GEOPOTENTIAL_HEIGHT, LONGITUDE, LATITUDE, PRESSURE_LEVEL, TIME, WIND_U, WIND_V
import xarray as xr
import re

class ECMWFProcessor:
    def __init__(self, dataset_path, chunking_in, chunking_out, normalizer=None, reduce_time=False):
        self.dataset_path = dataset_path
        self.chunking_in = chunking_in
        self.chunking_out = chunking_out
        self.normalizer = normalizer
        self.reduce_time = reduce_time

    def load(self):
        if self.chunking_in is not None:
            self.ds = xr.open_mfdataset(self.dataset_path, chunks=self.chunking_in, combine="by_coords")
        else:
            self.ds = xr.open_mfdataset(self.dataset_path, combine="by_coords")

        if self.reduce_time:
            self.ds = self.ds.isel(time=slice(0, 24))

    def preprocess(self, normalize=True):
        # ECMWF files often store one variable per pressure level (e.g., UGRD_250mb).
        # This stacks those variables into unified arrays with a pressure_level coordinate.
        level_var_pattern = re.compile(r"^(UGRD|VGRD|HGT)_(\d+)mb$")
        target_name_by_source = {
            "UGRD": WIND_U,
            "VGRD": WIND_V,
            "HGT": ALTITUDE,
        }

        grouped_vars = {WIND_U: [], WIND_V: [], ALTITUDE: []}

        for var_name in list(self.ds.data_vars):
            match = level_var_pattern.match(var_name)
            if not match:
                continue

            source_name, level = match.groups()
            grouped_vars[target_name_by_source[source_name]].append((int(level), var_name))

        vars_to_drop = []
        for target_name, level_vars in grouped_vars.items():
            if not level_vars:
                continue

            level_vars = sorted(level_vars, key=lambda item: item[0])
            arrays = [
                self.ds[var_name].expand_dims({PRESSURE_LEVEL: [level]})
                for level, var_name in level_vars
            ]
            self.ds[target_name] = xr.concat(arrays, dim=PRESSURE_LEVEL)
            vars_to_drop.extend([var_name for _, var_name in level_vars])

        if vars_to_drop:
            self.ds = self.ds.drop_vars(vars_to_drop)

        rename_map = {}
        if "latitude" in self.ds.coords and LATITUDE not in self.ds.coords:
            rename_map["latitude"] = LATITUDE
        if "longitude" in self.ds.coords and LONGITUDE not in self.ds.coords:
            rename_map["longitude"] = LONGITUDE
        if "time" in self.ds.coords and TIME != "time" and TIME not in self.ds.coords:
            rename_map["time"] = TIME

        if rename_map:
            self.ds = self.ds.rename(rename_map)

        if normalize and self.normalizer is not None:
            # use the normalizer to normalize the dataset
            print("Fitting normalizer on ECMWF dataset...")
            self.normalizer.fit(self.ds)
            self.ds = self.normalizer.normalize(self.ds)
            self.normalizer.save()
            

    def initialize(self, save_path=None):
        self.load()
        self.preprocess(normalize=(self.normalizer is not None)) # normalize if a normalizer is provided

        if save_path is not None and self.chunking_out is not None:
            from jetstream_interpolate_convcnp.processing.chunk_netcdf import chunk_and_save
            chunk_and_save(self.ds, save_path, chunk_size=self.chunking_out)

        if save_path is not None and self.chunking_out is None:
            self.ds.to_netcdf(save_path)