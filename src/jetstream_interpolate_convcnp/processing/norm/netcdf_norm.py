import xarray as xr
import os
import numpy as np
from jetstream_interpolate_convcnp.utils.constants import LATITUDE, LONGITUDE

class NetCDFNormalizer:
    def __init__(
        self,
        norm_path,
        average_over=None,
        average_per=None,
        vars_to_normalize=None,
        norm_params=None,
        min_std=1e-6,
    ):
        self.norm_path = norm_path

        self.norm_params = norm_params
        self.average_over = average_over
        self.average_per = average_per
        self.vars_to_normalize = vars_to_normalize
        self.min_std = float(min_std)

        self.file_name = "params.nc"

        if self.norm_path is not None:
            os.makedirs(self.norm_path, exist_ok=True)

        # if you can load existing normalization parameters, do so.
        if self.norm_params is not None and os.path.exists(os.path.join(self.norm_path, self.file_name)):
            self.params = xr.open_dataset(os.path.join(self.norm_path, self.file_name))

    def fit(self, ds):
        # create an empty xarray dataset using the coordinates of the input dataset
        # This is a multidimensional mean and std, so we need to average over the specified dimensions
        if not self.vars_to_normalize:
            raise ValueError("vars_to_normalize must be specified when fitting the normalizer.")

        param_data = {}

        for var in self.vars_to_normalize:
            if var not in ds:
                raise ValueError(f"Variable '{var}' not found in the dataset for normalization.")
            param_data[f"{var}_mean"] = ds[var].mean(dim=self.average_over)

            std = ds[var].std(dim=self.average_over)
            # Guard against exact/near-zero standard deviations that cause exploding values.
            std = xr.where(np.isfinite(std) & (std >= self.min_std), std, self.min_std)
            param_data[f"{var}_std"] = std

        self.params = xr.Dataset(param_data)

    def _ensure_params_loaded(self):
        if not hasattr(self, "params"):
            raise ValueError("Normalization parameters are not loaded. Call 'fit' or 'load_from_path' first.")

    def _rename_param_dims_to_match(self, da, ds):
        rename_map = {}
        if LATITUDE in da.dims and "latitude" in ds.coords:
            rename_map[LATITUDE] = "latitude"
        if LONGITUDE in da.dims and "longitude" in ds.coords:
            rename_map[LONGITUDE] = "longitude"
        if "latitude" in da.dims and LATITUDE in ds.coords:
            rename_map["latitude"] = LATITUDE
        if "longitude" in da.dims and LONGITUDE in ds.coords:
            rename_map["longitude"] = LONGITUDE
        if rename_map:
            da = da.rename(rename_map)
        return da

    def _align_param_to_ds(self, da, ds, method="nearest"):
        da = self._rename_param_dims_to_match(da, ds)

        interp_coords = {}
        for dim in da.dims:
            if dim in ds.coords and dim in da.coords:
                if da.sizes[dim] != ds.sizes[dim] or not da[dim].equals(ds[dim]):
                    interp_coords[dim] = ds[dim]

        if interp_coords:
            da = da.interp(interp_coords, method=method)

        return da

    def adapt_params_to_dataset(self, ds, method="nearest"):
        self._ensure_params_loaded()
        adapted = {}
        for var in self.vars_to_normalize:
            adapted[f"{var}_mean"] = self._align_param_to_ds(self.params[f"{var}_mean"], ds, method=method)
            adapted[f"{var}_std"] = self._align_param_to_ds(self.params[f"{var}_std"], ds, method=method)
            adapted[f"{var}_std"] = xr.where(
                np.isfinite(adapted[f"{var}_std"]) & (adapted[f"{var}_std"] >= self.min_std),
                adapted[f"{var}_std"],
                self.min_std,
            )
        self.params = xr.Dataset(adapted)

    def normalize(self, ds):
        self._ensure_params_loaded()

        for var in self.vars_to_normalize:
            if var not in ds:
                raise ValueError(f"Variable '{var}' not found in the dataset for normalization.")
            if f"{var}_mean" not in self.params or f"{var}_std" not in self.params:
                raise ValueError(f"Normalization parameters for variable '{var}' not found. Ensure that 'fit' has been called with the appropriate dataset.")

            mean = self._align_param_to_ds(self.params[f"{var}_mean"], ds, method="nearest")
            std = self._align_param_to_ds(self.params[f"{var}_std"], ds, method="nearest")
            std = xr.where(np.isfinite(std) & (std >= self.min_std), std, self.min_std)

            ds[var] = (ds[var] - mean) / std
            
        return ds

    def unnormalize(self, data):
        self._ensure_params_loaded()

        for var in self.vars_to_normalize:
            if f"{var}_mean" not in self.params or f"{var}_std" not in self.params:
                raise ValueError(f"Normalization parameters for variable '{var}' not found. Ensure that 'fit' has been called with the appropriate dataset.")

            mean = self._align_param_to_ds(self.params[f"{var}_mean"], data, method="nearest")
            std = self._align_param_to_ds(self.params[f"{var}_std"], data, method="nearest")
            std = xr.where(np.isfinite(std) & (std >= self.min_std), std, self.min_std)

            data[var] = data[var] * std + mean
        
        return data
    
    def save(self):
        # Save the mean and std as a NetCDF file
        self.params.to_netcdf(os.path.join(self.norm_path, self.file_name))

    def save_as_csv(self, path):
        # Save the mean and std as a CSV file
        norm_df = self.params.to_dataframe()
        norm_df.to_csv(path, index=True)

    def load_from_path(self):
        norm_ds = xr.open_dataset(os.path.join(self.norm_path, self.file_name))
        self.params = norm_ds

