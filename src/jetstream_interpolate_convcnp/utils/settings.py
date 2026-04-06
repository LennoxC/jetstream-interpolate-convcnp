import yaml

experiment_name = "train_v0.1"

with open(f"./src/jetstream_interpolate_convcnp/pipelines/configs/{experiment_name}.yaml", "r") as f:
    settings = yaml.safe_load(f)

