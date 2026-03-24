import xarray as xr

def chunk_and_save(input_ds, output_file, chunk_size):
    """
    Chunk a NetCDF file using xarray and save the chunked version to a new file.

    Parameters:
    - input_ds: xarray Dataset to be chunked.
    - output_file: Path to save the chunked NetCDF file.
    - chunk_size: Dictionary specifying the chunk size for each dimension, e.g., {'time': 24, 'lat': 10, 'lon': 10}.
    """
    # Open the input NetCDF file with xarray
    ds = input_ds

    # Chunk the dataset according to the specified chunk sizes
    ds_chunked = ds.chunk(chunk_size)

    # Save the chunked dataset to a new NetCDF file
    ds_chunked.to_netcdf(output_file)