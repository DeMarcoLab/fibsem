import datetime
import glob
import json
import logging
import os
import numpy as np
import time
from pathlib import Path
import sys

import yaml

from PIL import Image
from fibsem.structures import (
    BeamType,
    MicroscopeSettings,
    ImageSettings,
    SystemSettings,
    FibsemImage,
    FibsemMillingSettings,
    FibsemHardware,
    FibsemPatternSettings,
)
from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope


def current_timestamp():
    """Returns current time in a specific string format

    Returns:
        String: Current time
    """
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%I-%M-%S%p") #PM/AM doesnt work?


def current_timestamp_v2():
    """Returns current time in a specific string format

    Returns:
        String: Current time
    """
    return str(time.time()).replace(".", "_")

def _format_time_seconds(seconds: float) -> str:
    """Format a time delta in seconds to proper string format."""
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


def make_logging_directory(path: Path = None, name="run"):
    """
    Create a logging directory with the specified name at the specified file path. 
    If no path is given, it creates the directory at the default base path.

    Args:
        path (Path, optional): The file path to create the logging directory at. If None, default base path is used. 
        name (str, optional): The name of the logging directory to create. Default is "run".

    Returns:
        str: The file path to the created logging directory.
        """
    
    if path is None:
        path = os.path.join(cfg.BASE_PATH, "log")
    directory = os.path.join(path, name)
    os.makedirs(directory, exist_ok=True)
    return directory

# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
# https://stackoverflow.com/questions/61483056/save-logging-debug-and-show-only-logging-info-python
def configure_logging(path: Path = "", log_filename="logfile", log_level=logging.DEBUG, _DEBUG: bool = False):
    """Log to the terminal and to file simultaneously."""
    logfile = os.path.join(path, f"{log_filename}.log")

    file_handler = logging.FileHandler(logfile)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO if _DEBUG is False else logging.DEBUG)

    logging.basicConfig(
        format="%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s",
        level=log_level,
        # Multiple handlers can be added to your logging configuration.
        # By default log messages are appended to the file if it exists already
        handlers=[file_handler, stream_handler],
        force=True,
    )

    return logfile


def load_yaml(fname: Path) -> dict:
    """load yaml file

    Args:
        fname (Path): yaml file path

    Returns:
        dict: Items in yaml
    """
    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_yaml(path: Path, data: dict) -> None:
    """Saves a python dictionary object to a yaml file

    Args:
        path (Path): path location to save yaml file
        data (dict): dictionary object
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    path = Path(path).with_suffix(".yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, indent=4)


def create_gif(path: Path, search: str, gif_fname: str, loop: int = 0) -> None:
    """Creates a GIF from a set of images. Images must be in same folder

    Args:
        path (Path): Path to images folder
        search (str): search name
        gif_fname (str): name to save gif file
        loop (int, optional): _description_. Defaults to 0.
    """
    filenames = glob.glob(os.path.join(path, search))

    imgs = [Image.fromarray(FibsemImage.load(fname).data) for fname in filenames]

    print(f"{len(filenames)} images added to gif.")
    imgs[0].save(
        os.path.join(path, f"{gif_fname}.gif"),
        save_all=True,
        append_images=imgs[1:],
        loop=loop,
    )


def setup_session(
    session_path: Path = None,
    config_path: Path = None,
    protocol_path: Path = None,
    setup_logging: bool = True,
    ip_address: str = None,
    manufacturer: str = None
) -> tuple[FibsemMicroscope, MicroscopeSettings]:
    """Setup microscope session

    Args:
        session_path (Path): path to logging directory
        config_path (Path): path to config directory (system.yaml)
        protocol_path (Path): path to protocol file

    Returns:
        tuple: microscope, settings
    """

    # load settings
    settings = load_settings_from_config(config_path, protocol_path)

    # create session directories
    session = f'{settings.protocol["name"]}_{current_timestamp()}'
    if protocol_path is None:
        protocol_path = os.getcwd()

    # configure paths
    if session_path is None:
        session_path = cfg.LOG_PATH
    os.makedirs(session_path, exist_ok=True)

    # configure logging
    if setup_logging:
        configure_logging(session_path)

    # connect to microscope
    import fibsem.microscope as FibSem

    # cheap overloading
    if ip_address:
        settings.system.ip_address = ip_address
    
    if manufacturer:
        settings.system.manufacturer = manufacturer

    if settings.system.manufacturer == "Thermo":
        microscope = FibSem.ThermoMicroscope(settings.hardware, settings.system.stage)
        microscope.connect_to_microscope(
            ip_address=settings.system.ip_address, port=7520
        )

    elif settings.system.manufacturer == "Tescan":
        microscope = FibSem.TescanMicroscope(ip_address=settings.system.ip_address, hardware_settings=settings.hardware, stage_settings=settings.system.stage)
        microscope.connect_to_microscope(
            ip_address=settings.system.ip_address, port=8300
        )
    
    elif settings.system.manufacturer == "Demo":
        microscope = FibSem.DemoMicroscope(settings.hardware, settings.system.stage)
        microscope.connect_to_microscope()

    # image_settings
    settings.image.save_path = session_path

    logging.info(f"Finished setup for session: {session}")

    return microscope, settings


def load_settings_from_config(
    config_path: Path = None, protocol_path: Path = None
) -> MicroscopeSettings:
    """Load microscope settings from configuration files

    Args:
        config_path (Path, optional): path to config directory. Defaults to None.
        protocol_path (Path, optional): path to protocol file. Defaults to None.

    Returns:
        MicroscopeSettings: microscope settings
    """
    # TODO: this should just be system.yaml path, not directory
    if config_path is None:
        from fibsem.config import CONFIG_PATH

        config_path = os.path.join(CONFIG_PATH, "system.yaml")
    
    # system settings
    settings = load_yaml(os.path.join(config_path))
    system_settings = SystemSettings.__from_dict__(settings["system"])

    # user settings
    image_settings = ImageSettings.__from_dict__(settings["user"]["imaging"])

    milling_settings = FibsemMillingSettings.__from_dict__(settings["user"]["milling"])

    # protocol settings
    protocol = load_protocol(protocol_path)

    # hardware settings
    hardware_settings = FibsemHardware.__from_dict__(settings["model"])

    settings = MicroscopeSettings(
        system=system_settings,
        image=image_settings,
        protocol=protocol,
        milling=milling_settings,
        hardware=hardware_settings,
    )

    return settings


def load_protocol(protocol_path: Path = None) -> dict:
    """Load the protocol file from yaml

    Args:
        protocol_path (Path, optional): path to protocol file. Defaults to None.

    Returns:
        dict: protocol dictionary
    """
    if protocol_path is not None:
        protocol = load_yaml(protocol_path)
    else:
        protocol = {"name": "demo"}

    #protocol = _format_dictionary(protocol)

    return protocol


def _format_dictionary(dictionary: dict) -> dict:
    """Recursively traverse dictionary and covert all numeric values to flaot.

    Parameters
    ----------
    dictionary : dict
        Any arbitrarily structured python dictionary.

    Returns
    -------
    dictionary
        The input dictionary, with all numeric values converted to float type.
    """
    for key, item in dictionary.items():
        if isinstance(item, dict):
            _format_dictionary(item)
        elif isinstance(item, list):
            dictionary[key] = [
                _format_dictionary(i)
                for i in item
                if isinstance(i, list) or isinstance(i, dict)
            ]
        else:
            if item is not None:
                try:
                    dictionary[key] = float(dictionary[key])
                except ValueError:
                    pass
    return dictionary

def get_params(main_str: str) -> list:
    """Helper function to access relevant metadata parameters from sub field

    Args:
        main_str (str): Sub string of relevant metadata

    Returns:
        list: Parameters covered by metadata
    """
    cats = []
    cat_str = ""

    i = main_str.find("\n")
    i += 1
    while i < len(main_str):

        if main_str[i] == "=":
            cats.append(cat_str)
            cat_str = ""
            i += main_str[i:].find("\n")
        else:
            cat_str += main_str[i]

        i += 1
    return cats


def _get_position(name: str):
    
    from fibsem import config as cfg
    from fibsem.structures import FibsemStagePosition
    import os

    ddict = load_yaml(fname=os.path.join(cfg.CONFIG_PATH, "positions.yaml"))
    # get position from save positions?
    for d in ddict:
        if d["name"] == name:
            return FibsemStagePosition.__from_dict__(d)
    return None