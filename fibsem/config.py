METADATA_VERSION = "v1"

# sputtering rates, from microscope application files
MILLING_SPUTTER_RATE = {
    20e-12: 6.85e-3,  # 30kv
    0.2e-9: 6.578e-2,  # 30kv
    0.74e-9: 3.349e-1,  # 30kv
    0.89e-9: 3.920e-1,  # 20kv
    2.0e-9: 9.549e-1,  # 30kv
    2.4e-9: 1.309,  # 20kv
    6.2e-9: 2.907,  # 20kv
    7.6e-9: 3.041,  # 30kv
    28.0e-9: 1.18e1,  # 30 kv
}

import os
import fibsem

BASE_PATH = os.path.dirname(fibsem.__path__[0])
CONFIG_PATH = os.path.join(BASE_PATH, "fibsem", "config")

import yaml


def load_yaml(fname):
    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_microscope_manufacturer(config_path=None) -> str:
    if config_path is None:
        config_path = CONFIG_PATH

    # system settings
    settings = load_yaml(os.path.join(config_path, "system.yaml"))
    manufacturer = settings["system"]["manufacturer"]

    return manufacturer
