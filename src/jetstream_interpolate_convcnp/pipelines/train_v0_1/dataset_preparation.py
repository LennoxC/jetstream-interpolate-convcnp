from jetstream_interpolate_convcnp.utils.constants import TIME, DATE, LATITUDE, LONGITUDE
import os

def convert_era5(input_path, output_path, chunking_in, chunking_out, reduce_time=False):
    from jetstream_interpolate_convcnp.processing.era5.era5_processor import ERA5Processor
    era5_processor = ERA5Processor(input_path, chunking_in, chunking_out, reduce_time=reduce_time)
    era5_processor.initialize(save_path=output_path)

def convert_ecmwf(input_path, output_path, chunking_in, chunking_out, reduce_time=False):
    from jetstream_interpolate_convcnp.processing.ecmwf.ecmwf_processor import ECMWFProcessor
    ecmwf_processor = ECMWFProcessor(input_path, chunking_in, chunking_out, reduce_time=reduce_time)
    ecmwf_processor.initialize(save_path=output_path)


def dataset_conversions(settings):    
    """
    Convert datasets to appropriate formats. Paths is the paths config from the yaml file.
    """
    
    paths = settings['paths']

    # first process era5. chunk by day, and one chunk per latitude/longitude slice.

    if settings['execute']['preprocessing']['era5']:
        from jetstream_interpolate_convcnp.processing.era5.era5_processor import ERA5Processor
        print("Processing era5 dataset...")

        # ensure that the output directory exists
        output_dir = os.path.dirname(paths['process_era5_path_base'])
        os.makedirs(output_dir, exist_ok=True)

        convert_era5(paths['era5_load_path'], 
                     paths['process_era5_path_base'], 
                     chunking_in=None,
                     chunking_out={TIME: 24, LATITUDE: 360, LONGITUDE: 360}, 
                     reduce_time=settings['environment']['small_ds'])
        
        print("Finished processing era5 dataset.")

    if settings['execute']['preprocessing']['ecmwf']:
        from jetstream_interpolate_convcnp.processing.ecmwf.ecmwf_processor import ECMWFProcessor
        print("Processing ecmwf dataset...")

        output_dir = os.path.dirname(paths['process_ecmwf_path_base'])
        os.makedirs(output_dir, exist_ok=True)

        convert_ecmwf(paths['ecmwf_load_path'], 
                      paths['process_ecmwf_path_base'], 
                      chunking_in={"time": 24, "latitude": 360, "longitude": 360},
                      chunking_out={TIME: 24, LATITUDE: 360, LONGITUDE: 360}, 
                      reduce_time=settings['environment']['small_ds'])

        print("Finished processing ecmwf dataset.")

    if settings['execute']['preprocessing']['amdar']:
        from jetstream_interpolate_convcnp.processing.amdar.AMDARProcessor import AMDARProcessor
        print("Processing AMDAR dataset...")

        output_dir = os.path.dirname(paths['process_amdar_path_base'])
        os.makedirs(output_dir, exist_ok=True)

        amdar_processor = AMDARProcessor(paths['amdar_load_path'], 
                                         partition_cols=[DATE, LATITUDE, LONGITUDE], 
                                         reduce_time=settings['environment']['small_ds'],
                                         skiprows=184,
                                         encoding_errors='ignore')
        amdar_processor.initialize(save_path=paths['process_amdar_path_base'])

        print("Finished processing AMDAR dataset.")

def dataset_preparation(settings):
    """
    Clean up datasets, normalize datasets, and define a sampling strategy for training.
    """
    if settings['settings']['clear_dataset_save_dir_on_startup']:
        save_dir = settings['paths']['process_base']
        print(f"Clearing dataset save directory: {save_dir}")
        
        # delete the directory and all contents if it exists, then recreate the directory
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    dataset_conversions(settings)