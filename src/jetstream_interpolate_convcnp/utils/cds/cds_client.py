import logging
import cdsapi
import yaml
import datetime
from pathlib import Path

"""
CDS API client and ERA5 data download functions.
"""

logger = logging.getLogger(__name__)

def get_cds_client():
    """
    Create and return a CDS API client using the cdsapi library.
    The API URL and key are retrieved from environment variables.
    """
    
    url = "https://cds.climate.copernicus.eu/api"

    client = cdsapi.Client(
        url = url
        # key must be stored in $HOME/.cdsapirc file with the following content:
        # url: https://cds.climate.copernicus.eu/api
        # key: <your-key>
    )

    return client

def get_pressure_files(_times_dt: list, _cfg: dict):
    """
    Downloads pressure-level files from the ECMWF Climate Data Store (CDS) using the cdsapi.
    :param _times_dt: the list of dates and times to be downloaded.
    :param cfg: the dictionary of configuration settings.
    :return: None
    """
    dates_str = f'{_times_dt.strftime("%Y%m%d")}'
    # hourly frequency is the amount of hours between each time entry. For example, if hourly_freq=3, times will be ['00:00', '03:00', '06:00', ..., '21:00']
    hourly_freq = _cfg["hourly_freq"]
    times = [f'{i:02d}:00' for i in range(0, 24, hourly_freq)]
    download_file = Path(_cfg['download_dir']) / 'pressure' / str(_times_dt.year) / str(_times_dt.month).zfill(2) / f'ERA5_{dates_str}_pressure.nc'
    # Create the directory if it doesn't exist
    download_file.parent.mkdir(parents=True, exist_ok=True)

    # Download the data
    c = get_cds_client()
    c.retrieve('reanalysis-era5-pressure-levels',
               {
                   'product_type':'reanalysis',
                   'format':'netcdf',
                   'pressure_level': _cfg['pressure_levels'],
                   'date': dates_str.replace('-', '/'),
                   'area':[_cfg['Nort'], _cfg['West'], _cfg['Sout'], _cfg['East']],
                   'time':times,
                   'variable':_cfg['pressure_var'],
               },
               download_file
               )
    
    logger.info(f"Downloaded pressure-level data for {dates_str} to {download_file}")

def get_surface_files(_times_dt, _cfg):
    """
    Downloads surface-level files from the ECMWF Climate Data Store(CDS) using the cdsapi.
    : param _times_dt: the list of dates and times to be downloaded.
    : param cfg: the dictionary of configuration settings.
    : return: None
    """
    dates_str = f'{_times_dt.strftime("%Y%m%d")}'    
    # hourly frequency is the amount of hours between each time entry. For example, if hourly_freq=3, times will be ['00:00', '03:00', '06:00', ..., '21:00']
    hourly_freq = _cfg["hourly_freq"]
    times = [f'{i:02d}:00' for i in range(0, 24, hourly_freq)]
    
    download_file = Path(_cfg['download_dir']) / 'surface' / str(_times_dt.year) / str(_times_dt.month).zfill(2) / f'ERA5_{dates_str}_surface.nc'
    # Create the directory if it doesn't exist
    download_file.parent.mkdir(parents=True, exist_ok=True)

    # Download the data
    c = get_cds_client()
    c.retrieve('reanalysis-era5-single-levels',
               {
                   'product_type': 'reanalysis',
                   'format': 'netcdf',
                   'variable': _cfg['surface_var'],
                   'date': dates_str.replace('-', '/'),
                   'area': [_cfg['Nort'], _cfg['West'], _cfg['Sout'], _cfg['East']],
                   'time': times
               },
               download_file)
    logger.info(f"Downloaded surface-level data for {dates_str} to {download_file}")

def download_era5_data(args):

    config_path = Path(args.config)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the configuration settings
    cfg['download_dir'] = output_dir
    cfg['start_date'] = args.start_date
    cfg['end_date'] = args.end_date

    # default to downloading both surface and pressure data unless specified otherwise
    if 'download_surface' not in cfg:
        cfg['download_surface'] = True
    if 'download_pressure' not in cfg:
        cfg['download_pressure'] = True
    

    # Sort out the start and end times.
    times_dt = []
    start_dt = datetime.datetime.strptime(args.start_date, '%Y%m%d')
    end_dt = datetime.datetime.strptime(args.end_date, '%Y%m%d')
    assert end_dt >= start_dt, 'The end time needs to be after start time'

    new_dt = start_dt
    while new_dt <= end_dt:
        times_dt.append(new_dt)
        new_dt += datetime.timedelta(days=1)

    if not args.parallel:
        for day in times_dt:        
            logger.info(f"Downloading data for {day.strftime('%Y-%m-%d')}")

            # Download pressure-level files
            if cfg['download_pressure']:
                logger.info("Downloading pressure-level files.")
                get_pressure_files(day, cfg)

            # Download surface-level files
            if cfg['download_surface']:
                logger.info("Downloading surface-level files.")
                get_surface_files(day, cfg)

    else:
        # Parallel downloads
        import concurrent.futures
        from tqdm import tqdm
        
        # Create a list of all tasks (each day produces 2 tasks)
        all_tasks = []
        for day in times_dt:
            if cfg['download_pressure']:
                all_tasks.append((get_pressure_files, day, cfg))
            if cfg['download_surface']:
                all_tasks.append((get_surface_files, day, cfg))

        # Define your batch size (number of concurrent requests)
        batch_size = 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Process tasks in batches
            for i in range(0, len(all_tasks), batch_size):
                batch = all_tasks[i : i + batch_size]
                futures = [
                    executor.submit(task, day, cfg)
                    for task, day, cfg in batch
                ]
                # Use tqdm to monitor progress of the current batch
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error downloading data: {e}")
    logger.info("All downloads completed.")