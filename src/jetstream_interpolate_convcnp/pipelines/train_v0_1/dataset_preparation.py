from jetstream_interpolate_convcnp.utils.constants import TIME, DATE, LATITUDE, LONGITUDE, WIND_U, WIND_V
from jetstream_interpolate_convcnp.processing.norm.netcdf_norm import NetCDFNormalizer
from jetstream_interpolate_convcnp.processing.ecmwf.ecmwf_processor import ECMWFProcessor
from jetstream_interpolate_convcnp.processing.amdar.amdar_processor import AMDARProcessor
import os
import shutil

# def convert_era5(input_path, output_path, chunking_in, chunking_out, normalizer=None, reduce_time=False):
#     from jetstream_interpolate_convcnp.processing.era5.era5_processor import ERA5Processor
#     era5_processor = ERA5Processor(input_path, chunking_in, chunking_out, normalizer=normalizer, reduce_time=reduce_time)
#     era5_processor.initialize(save_path=output_path)


def dataset_conversions(settings):    
    """
    Convert datasets to appropriate formats. Paths is the paths config from the yaml file.
    """
    
    paths = settings['paths']

    # first process era5. chunk by day, and one chunk per latitude/longitude slice.

    # if settings['execute']['preprocessing']['era5']:
    #     from jetstream_interpolate_convcnp.processing.era5.era5_processor import ERA5Processor
    #     print("Processing era5 dataset...")
# 
    #     # ensure that the output directory exists
    #     output_dir = os.path.dirname(paths['process_era5_path_base'])
    #     os.makedirs(output_dir, exist_ok=True)
# 
    #     avg_over = ['valid_time']
    #     average_per = ['latitude', 'longitude']
    #     vars_to_normalize = ['u', 'v']
    #     norm_path = settings['paths']['era5_norm_params_path']
# 
    #     if settings['settings']['clear_norm_params_on_startup']:
    #         print(f"Clearing normalization parameters directory: {norm_path}")
    #         if os.path.exists(norm_path):
    #             shutil.rmtree(norm_path)
    #         os.makedirs(norm_path, exist_ok=True)
# 
    #     convert_era5(paths['era5_load_path'], 
    #                     paths['process_era5_path_base'], 
    #                     chunking_in=None,
    #                     chunking_out={TIME: 24, LATITUDE: 360, LONGITUDE: 360}, 
    #                     normalizer=NetCDFNormalizer(norm_path, average_over=avg_over, average_per=average_per, vars_to_normalize=vars_to_normalize),
    #                     reduce_time=settings['environment']['small_ds'])
    #     
    #     print("Finished processing era5 dataset.")


    # if recomputing norm params, ecmwf must be processed first. AMDAR uses the ECMWF norm params for normalization.
    if settings['execute']['preprocessing']['ecmwf']:
        print("Processing ecmwf dataset...")

        output_dir = os.path.dirname(paths['process_ecmwf_path_base'])
        os.makedirs(output_dir, exist_ok=True)

        norm_path = settings['paths']['ecmwf_norm_params_path']

        if settings['settings']['clear_norm_params_on_startup']:
            print(f"Clearing normalization parameters directory: {norm_path}")
            if os.path.exists(norm_path):
                shutil.rmtree(norm_path)
            os.makedirs(norm_path, exist_ok=True)

        ecmwf_processor = ECMWFProcessor(chunking_in={"time": 24, "latitude": 360, "longitude": 360}, do_normalize=True, reduce_time=settings['environment']['small_ds'])
        ecmwf_processor.run()

        print("Finished processing ecmwf dataset.")

    if settings['execute']['preprocessing']['amdar']:
        print("Processing AMDAR dataset...")

        output_dir = os.path.dirname(paths['process_amdar_path_base'])
        os.makedirs(output_dir, exist_ok=True)

        amdar_processor = AMDARProcessor(partition_cols=['year', 'month', 'day', f"coarse_{LATITUDE}", f"coarse_{LONGITUDE}"], 
                                         reduce_time=settings['environment']['small_ds'],
                                         skiprows=184,
                                         encoding_errors='ignore')
        
        amdar_processor.run()

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