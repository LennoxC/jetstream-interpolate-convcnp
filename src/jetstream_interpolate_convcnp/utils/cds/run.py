# download_file = "/mnt/hdd/jetstream/data/development/era5/july2019/reanalysis_2019_07.nc"

# This script is intended to download ERA5 data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
# Currently only implemented for downloads of hourly data, grouped by day.

# After signing up for an API key, you will need to accept the license agreement. 
# See https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences
# You will be prompted to do this the first time you attempt to use the API.

from jetstream_interpolate_convcnp.utils.cds.cds_client import download_era5_data
from argparse import ArgumentParser
import os

def get_commandline_args():
    """
    Parse command line arguments, return parser object.
    """
    parser = ArgumentParser(description="Download ERA5 data from the Copernicus Climate Data Store (CDS).")
    parser.add_argument(
        "-c", "--config", type=str, dest="config", default=None,
        help="Path to the ERA5 configuration file.",
    )
    parser.add_argument(
        "-s", "--start_date", type=str, dest="start_date", default=None,
        help="Start date for the data download in YYYYMMDD format.",
    )
    parser.add_argument(
        "-e", "--end_date", type=str, dest="end_date", default=None,
        help="End date for the data download in YYYYMMDD format.",
    )
    parser.add_argument(
        "-o", "--output", type=str, dest="output", default=None, # default output path is DATA_HOME/ERA5_SUFFIX
        help="Output directory for the downloaded data. Default is DATA_HOME/ERA5_SUFFIX."
    )
    parser.add_argument(
        "-p", "--parallel", action='store_true', dest="parallel", default=True,
        help="Use parallel downloading.",
    )
    return parser.parse_args()

def main():
    """
    Main function to download ERA5 data based on command line arguments.
    """
    args = get_commandline_args()
    download_era5_data(args)
    
if __name__ == "__main__":
    main()