METADATA_VERSION = "v3"

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

SUPPORTED_COORDINATE_SYSTEMS = [
    "RAW",
    "SPECIMEN",
    "STAGE",
    "Raw",
    "raw",
    "specimen",
    "Specimen",
    "Stage",
    "stage",
]


REFERENCE_HFW_WIDE = 2750e-6
REFERENCE_HFW_LOW = 900e-6
REFERENCE_HFW_MEDIUM = 400e-6
REFERENCE_HFW_HIGH = 150e-6
REFERENCE_HFW_SUPER = 80e-6
REFERENCE_HFW_ULTRA = 50e-6

REFERENCE_RES_SQUARE = [1024, 1024]
REFERENCE_RES_LOW = [768, 512]
REFERENCE_RES_MEDIUM = [1536, 1024]
REFERENCE_RES_HIGH = [3072, 2048]
REFERENCE_RES_SUPER = [6144, 4096]

MILL_HFW_THRESHOLD = 0.005  # 0.5% of the image


import os
import fibsem

BASE_PATH = os.path.dirname(
    fibsem.__path__[0]
)  # TODO: figure out a more stable way to do this
CONFIG_PATH = os.path.join(BASE_PATH, "fibsem", "config")
SYSTEM_PATH = os.path.join(CONFIG_PATH, "system.yaml")
PROTOCOL_PATH = os.path.join(CONFIG_PATH, "protocol.yaml")
LOG_PATH = os.path.join(BASE_PATH, "fibsem", "log")
DATA_PATH = os.path.join(BASE_PATH, "fibsem", "log", "data")
DATA_ML_PATH: str = os.path.join(BASE_PATH, "fibsem", "log", "data", "ml")
DATA_CC_PATH: str = os.path.join(BASE_PATH, "fibsem", "log", "data", "crosscorrelation")
DATA_TILE_PATH: str = os.path.join(DATA_PATH, "tile")
POSITION_PATH = os.path.join(CONFIG_PATH, "positions.yaml")
MODELS_PATH = os.path.join(BASE_PATH, "fibsem", "segmentation", "models")
MICROSCOPE_CONFIGURATION_PATH = os.path.join(
    CONFIG_PATH, "microscope-configuration.yaml"
)



os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DATA_ML_PATH, exist_ok=True)
os.makedirs(DATA_CC_PATH, exist_ok=True)
os.makedirs(DATA_TILE_PATH, exist_ok=True)

DATABASE_PATH = os.path.join(BASE_PATH, "fibsem", "db", "fibsem.db")
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
import yaml


def load_yaml(fname):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        fname (str): The path to the YAML file to be loaded.

    Returns:
        dict: A dictionary containing the contents of the YAML file.

    Raises:
        IOError: If the file cannot be opened or read.
        yaml.YAMLError: If the file is not valid YAML.
    """
    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config


__SUPPORTED_MANUFACTURERS__ = ["Thermo", "Tescan", "Demo"]
__DEFAULT_MANUFACTURER__ = "Thermo"
__DEFAULT_IP_ADDRESS__ = "192.168.0.1"
__SUPPORTED_PLASMA_GASES__ = ["Argon", "Oxygen", "Nitrogen", "Xenon"]


def get_default_user_config() -> dict:
    """Return the default configuration."""
    return {
        "name":                           "default-configuration",       # a descriptive name for your configuration 
        "ip_address":                     __DEFAULT_IP_ADDRESS__,        # the ip address of the microscope PC
        "manufacturer":                   __DEFAULT_MANUFACTURER__,      # the microscope manufactuer, Thermo, Tescan or Demo                       
        "rotation-reference":             0,                             # the reference rotation value (rotation when loading)  [degrees]
        "shuttle-pre-tilt":               35,                            # the pre-tilt of the shuttle                           [degrees]
        "electron-beam-eucentric-height": 7.0e-3,                        # the eucentric height of the electron beam             [metres]
        "ion-beam-eucentric-height":      16.5e-3,                       # the eucentric height of the ion beam                  [metres]
    }


# user configurations -> move to fibsem.db eventually
USER_CONFIGURATIONS_PATH = os.path.join(CONFIG_PATH, "user-configurations.yaml")
USER_CONFIGURATIONS_YAML = load_yaml(USER_CONFIGURATIONS_PATH)
USER_CONFIGURATIONS = USER_CONFIGURATIONS_YAML["configurations"]
DEFAULT_CONFIGURATION_NAME = USER_CONFIGURATIONS_YAML["default"]
DEFAULT_CONFIGURATION_PATH = USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME]["path"]


if DEFAULT_CONFIGURATION_PATH is None:
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME][
        "path"
    ] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH

if not os.path.exists(DEFAULT_CONFIGURATION_PATH):
    DEFAULT_CONFIGURATION_NAME = "default-configuration"
    USER_CONFIGURATIONS[DEFAULT_CONFIGURATION_NAME][
        "path"
    ] = MICROSCOPE_CONFIGURATION_PATH
    DEFAULT_CONFIGURATION_PATH = MICROSCOPE_CONFIGURATION_PATH
        
print(f"Default configuration: {DEFAULT_CONFIGURATION_NAME}")
print(f"Default configuration path: {DEFAULT_CONFIGURATION_PATH}")

def add_configuration(configuration_name: str, path: str):
    """Add a new configuration to the user configurations file."""
    if configuration_name in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' already exists.")

    USER_CONFIGURATIONS[configuration_name] = {"path": path}
    USER_CONFIGURATIONS_YAML["configurations"] = USER_CONFIGURATIONS
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


def remove_configuration(configuration_name: str):
    """Remove a configuration from the user configurations file."""
    if configuration_name not in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' does not exist.")

    del USER_CONFIGURATIONS[configuration_name]
    USER_CONFIGURATIONS_YAML["configurations"] = USER_CONFIGURATIONS
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


def set_default_configuration(configuration_name: str):
    """Set the default configuration in the user configurations file."""
    if configuration_name not in USER_CONFIGURATIONS:
        raise ValueError(f"Configuration name '{configuration_name}' does not exist.")

    USER_CONFIGURATIONS_YAML["default"] = configuration_name
    with open(USER_CONFIGURATIONS_PATH, "w") as f:
        yaml.dump(USER_CONFIGURATIONS_YAML, f)


# default configuration values
DEFAULT_CONFIGURATION_VALUES = {
    "Thermo": {
        "ion-column-tilt": 52,
        "electron-column-tilt": 0,
    },
    "Tescan": {
        "ion-column-tilt": 55,
        "electron-column-tilt": 0,
    },
    "Demo": {
        "ion-column-tilt": 52,
        "electron-column-tilt": 0,
    },
}


# machine learning
HUGGINFACE_REPO = "patrickcleeve/autolamella"
__DEFAULT_CHECKPOINT__ = "autolamella-mega-20240107.pt"




# feature flags

_LIVE_IMAGING_ENABLED = False
_MINIMAP_VISUALISATION = False
_MINIMAP_MOVE_WITH_TRANSLATION = False
_MINIMAP_ACQUIRE_AFTER_MOVEMENT = False
_APPLY_CONFIGURATION_ENABLED = True



# tescan manipulator

TESCAN_MANIPULATOR_CALIBRATION_PATH = os.path.join(CONFIG_PATH, "tescan_manipulator.yaml")

def load_tescan_manipulator_calibration() -> dict:
    """Load the tescan manipulator calibration"""
    config = utils.load_yaml(cfg.TESCAN_MANIPULATOR_CALIBRATION_PATH)
    return config

def save_tescan_manipulator_calibration(config: dict) -> None:
    """Save the tescan manipulator calibration"""
    utils.save_yaml(cfg.TESCAN_MANIPULATOR_CALIBRATION_PATH, config)
    return None