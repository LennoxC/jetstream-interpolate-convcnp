import tqdm
import os
import yaml
from jetstream_interpolate_convcnp.pipelines.train_v0_1.dataset_preparation import dataset_preparation

"""
Training pipeline v0.1

Tasks:
- convert datasets to appropriate formats
- clean up datasets
- normalize datasets
- define a sampling strategy for training
- build convcnp model
- train model

"""

def program():
    """
    Execute the actions in order
    """
    # load settings from the config file at ./configs/train_v0.1.yaml

    with open("./src/jetstream_interpolate_convcnp/pipelines/configs/train_v0.1.yaml", "r") as f:
        settings = yaml.safe_load(f)

    if settings['execute']['dataset_preprocessing']:
        dataset_preparation(settings)

def main():
    # skinny main method to run program, which is where the actual logic of the training pipeline lives.
    program()

if __name__ == "__main__":
    main()