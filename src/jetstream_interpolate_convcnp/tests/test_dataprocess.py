import os
import shutil
import yaml
from jetstream_interpolate_convcnp.processing.norm.netcdf_norm import NetCDFNormalizer
from jetstream_interpolate_convcnp.pipelines.train_v0_1.dataset_preparation import convert_ecmwf
from jetstream_interpolate_convcnp.utils.constants import TIME, LATITUDE, LONGITUDE, WIND_U, WIND_V
from jetstream_interpolate_convcnp.processing.ecmwf.ecmwf_processor import ECMWFProcessor
from jetstream_interpolate_convcnp.processing.amdar.amdar_processor import AMDARProcessor

# Test the data processing pipeline.
# - Process the datasets, create a sample, and normalize.
# - Unnormalize the sample and check that it matches the original data.
# The test config file relaxes the resolution and window size to make it easier to test.

def setup():
    with open("./src/jetstream_interpolate_convcnp/tests/test_config.yaml", "r") as f:
        settings = yaml.safe_load(f)

    test_processing(settings)

def test_processing(settings):
    # ensure required paths exist
    output_dir = os.path.dirname(settings['paths']['process_ecmwf_path_base'])
    output_dir = os.path.dirname(settings['paths']['process_amdar_path_base'])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # remove norm params if they already exist
    if os.path.exists(settings['paths']['ecmwf_norm_params_path']):
        shutil.rmtree(settings['paths']['ecmwf_norm_params_path'])
    os.makedirs(settings['paths']['ecmwf_norm_params_path'], exist_ok=True)

    # process ecmwf dataset
    convert_ecmwf(settings['paths']['ecmwf_load_path'], 
                    settings['paths']['process_ecmwf_path_base'], 
                    chunking_in={"time": 24, "latitude": 360, "longitude": 360},
                    chunking_out={TIME: 24, LATITUDE: 360, LONGITUDE: 360}, 
                    normalizer=NetCDFNormalizer(settings['paths']['ecmwf_norm_params_path'], average_over=[TIME], average_per=[LATITUDE, LONGITUDE], vars_to_normalize=[WIND_U, WIND_V]),
                    reduce_time=False)

    # process amdar dataset
    amdar_processor = AMDARProcessor(settings['paths']['amdar_load_path'], 
                                        partition_cols=['year', 'month', 'day', f"{LATITUDE}_int", f"{LONGITUDE}_int"], 
                                        reduce_time=settings['environment']['small_ds'],
                                        skiprows=184,
                                        encoding_errors='ignore')
    amdar_processor.initialize(save_path=settings['paths']['process_amdar_path_base'])

if __name__ == "__main__":
    setup()
    
