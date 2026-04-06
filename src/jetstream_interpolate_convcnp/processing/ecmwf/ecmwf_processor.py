from jetstream_interpolate_convcnp.utils.constants import ALTITUDE, GEOPOTENTIAL_HEIGHT, LONGITUDE, LATITUDE, PRESSURE_LEVEL, TIME, WIND_U, WIND_V
import xarray as xr
import numpy as np
import dask.dataframe as dd
import dask
import re
from jetstream_interpolate_convcnp.utils.settings import settings

class ECMWFProcessor:
    def __init__(self, chunking_in=None, do_normalize=True, reduce_time=False):
        self.dataset_path = settings['paths']['ecmwf_load_path']
        self.norm_path = settings['paths']['ecmwf_norm_params_path']
        self.save_path = settings['paths']['process_ecmwf_path_base']
        
        self.chunking_in = chunking_in
        self.do_normalize = do_normalize
        self.reduce_time = reduce_time
        

    def load(self):
        if self.chunking_in is not None:
            self.ds = xr.open_mfdataset(self.dataset_path, chunks=self.chunking_in, combine="by_coords")
        else:
            self.ds = xr.open_mfdataset(self.dataset_path, combine="by_coords")

        if self.reduce_time:
            self.ds = self.ds.isel(time=slice(0, 24))

    def preprocess(self):
        # ECMWF files often store one variable per pressure level (e.g., UGRD_250mb).
        # This stacks those variables into unified arrays with a pressure_level coordinate.

        # ===== netcdf processing ===== #

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

        # ===== dask dataframe processing ===== #
        # now convert to offgrid dataframe and process this + norm
        df = self.ds.to_dask_dataframe().reset_index()

        altitude_band_width_m = 2000
        coarsening_factor_deg = 2

        df['altitude_band'] = np.floor(df['altitude'] / altitude_band_width_m).astype('int32')
        df['log_altitude'] = np.log(df['altitude']).astype('float32')
        
        df['coarse_lat'] = (np.floor(df['lat'] / coarsening_factor_deg) * coarsening_factor_deg).astype('int32')
        df['coarse_lon'] = (np.floor(df['lon'] / coarsening_factor_deg) * coarsening_factor_deg).astype('int32')
        
        df = self.normalize(df)

        df.to_parquet(
            self.save_path,
            partition_on=['altitude_band', 'coarse_lat', 'coarse_lon'],
            write_index=False
        )

    def normalize(self, df):
        group_cols = ['altitude_band', 'coarse_lat', 'coarse_lon']

        # --- Step 1: compute global stats (small table) ---
        # group by and keep altitude_band, coarse_lat, coarse_lon as keys for normalization
        df_norm = df.groupby(group_cols).agg(
            u_mean=('u', 'mean'),
            u_std=('u', 'std'),
            v_mean=('v', 'mean'),
            v_std=('v', 'std'),
        ).reset_index().persist()

        # materialize small lookup table
        df_norm_pd = df_norm.compute()

        # optional: write once here (cheaper than later)
        df_norm.to_parquet(
            self.norm_path,
            partition_on=group_cols,
            write_index=False
        )

        # --- Step 2: partition-local normalization ---
        def normalize_partition(pdf, norm_df):
            pdf = pdf.merge(norm_df, on=group_cols, how='left')

            pdf['u_normed'] = (pdf['u'] - pdf['u_mean']) / pdf['u_std']
            pdf['v_normed'] = (pdf['v'] - pdf['v_mean']) / pdf['v_std']

            return pdf

        # --- Step 3: EXPLICIT META (critical) ---
        meta = df._meta.copy()
        meta['u_mean'] = np.float64()
        meta['u_std'] = np.float64()
        meta['v_mean'] = np.float64()
        meta['v_std'] = np.float64()
        meta['u_normed'] = np.float32()
        meta['v_normed'] = np.float32()

        return df.map_partitions(normalize_partition, df_norm_pd, meta=meta)

    def unnormalize(self, df):
        group_cols = ['altitude_band', 'coarse_lat', 'coarse_lon']

        # load small norm table
        df_norm = self.load_norm_params().compute()

        def unnormalize_partition(pdf, norm_df):
            pdf = pdf.merge(norm_df, on=group_cols, how='left')

            pdf['u'] = pdf['u_normed'] * pdf['u_std'] + pdf['u_mean']
            pdf['v'] = pdf['v_normed'] * pdf['v_std'] + pdf['v_mean']

            return pdf

        meta = df._meta.copy()
        meta['u'] = np.float32()
        meta['v'] = np.float32()

        return df.map_partitions(unnormalize_partition, df_norm, meta=meta)

    def load_norm_params(self):
        if self.norm_path is None:
            raise ValueError("norm_path must be set to load normalization parameters")

        df_norm = dd.read_parquet(self.norm_path)
        return df_norm

    def run(self):
        self.load()
        self.preprocess() # normalize if a normalizer is provided