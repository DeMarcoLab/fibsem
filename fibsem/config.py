METADATA_VERSION = "v2"

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
    "stage"

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

MILL_HFW_THRESHOLD = 0.005 # 0.5% of the image 


import os
import fibsem

BASE_PATH = os.path.dirname(fibsem.__path__[0]) # TODO: figure out a more stable way to do this
CONFIG_PATH = os.path.join(BASE_PATH, "fibsem", "config")
SYSTEM_PATH = os.path.join(CONFIG_PATH, "system.yaml")
PROTOCOL_PATH = os.path.join(CONFIG_PATH, "protocol.yaml")
LOG_PATH = os.path.join(BASE_PATH, "fibsem", "log")
DATA_PATH = os.path.join(BASE_PATH, "fibsem", "log", "data")
DATA_ML_PATH:str = os.path.join(BASE_PATH, "fibsem", "log", "data", "ml")
DATA_CC_PATH:str = os.path.join(BASE_PATH, "fibsem", "log", "data", "crosscorrelation")
DATA_TILE_PATH:str = os.path.join(DATA_PATH, "tile")
POSITION_PATH = os.path.join(CONFIG_PATH, "positions.yaml")   
MODELS_PATH = os.path.join(BASE_PATH, "fibsem", "segmentation", "models")

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
__DEFAULT_IP_ADDRESS__ = "10.0.0.1"


_LIVE_IMAGING_ENABLED = False
_MINIMAP_VISUALISATION = False
_MINIMAP_MOVE_WITH_TRANSLATION = True