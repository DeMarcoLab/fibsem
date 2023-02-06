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
)
from fibsem.microscope import FibsemMicroscope


def save_image(image: FibsemImage, save_path: Path, label: str = "image"):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{label}.tif")
    image.save(path)


def current_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%I-%M-%S%p")


def _format_time_seconds(seconds: float) -> str:
    """Format a time delta in seconds to proper string format."""
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


# TODO: better logs: https://www.toptal.com/python/in-depth-python-logging
# https://stackoverflow.com/questions/61483056/save-logging-debug-and-show-only-logging-info-python
def configure_logging(path: Path = "", log_filename="logfile", log_level=logging.DEBUG):
    """Log to the terminal and to file simultaneously."""
    logfile = os.path.join(path, f"{log_filename}.log")

    file_handler = logging.FileHandler(logfile)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

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

    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_yaml(path: Path, data: dict) -> None:

    with open(path, "w") as f:
        yaml.dump(data, f, indent=4)


from fibsem.structures import MicroscopeState


def save_state_yaml(path: Path, state: MicroscopeState) -> None:

    state_dict = state.__to_dict__()

    save_yaml(path, state_dict)


def save_metadata(settings: MicroscopeSettings, path: Path):
    # TODO: finish this
    pass
    # settings_dict = settings.__to_dict__()

    # fname = os.path.join(path, "metadata.json")
    # with open(fname, "w") as fp:
    #     json.dump(settings_dict, fp, sort_keys=True, indent=4)


def create_gif(path: Path, search: str, gif_fname: str, loop: int = 0) -> None:
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
        session_path = os.path.join(os.path.dirname(protocol_path), session)
    os.makedirs(session_path, exist_ok=True)

    # configure logging
    if setup_logging:
        configure_logging(session_path)

    # connect to microscope
    import fibsem.microscope as FibSem

    if settings.system.manufacturer == "Thermo":
        microscope = FibSem.ThermoMicroscope()
        microscope.connect_to_microscope(
            ip_address=settings.system.ip_address, port=7520
        )

    elif settings.system.manufacturer == "Tescan":
        microscope = FibSem.TescanMicroscope(ip_address=settings.system.ip_address)
        microscope.connect_to_microscope(
            ip_address=settings.system.ip_address, port=8300
        )

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
    # print("HELLO")
    # TODO: this should just be system.yaml path, not directory
    if config_path is None:
        from fibsem.config import CONFIG_PATH

        config_path = CONFIG_PATH

    # system settings
    settings = load_yaml(os.path.join(config_path, "system.yaml"))
    system_settings = SystemSettings.__from_dict__(settings["system"])

    # user settings
    # default_settings = DefaultSettings.__from_dict__(settings["user"])
    image_settings = ImageSettings.__from_dict__(settings["user"]["imaging"])

    milling_settings = FibsemMillingSettings.__from_dict__(settings["user"]["milling"])

    # protocol settings
    protocol = load_protocol(protocol_path)

    settings = MicroscopeSettings(
        system=system_settings,
        # default=default_settings,
        image=image_settings,
        protocol=protocol,
        milling=milling_settings,
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

    protocol = _format_dictionary(protocol)

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


def match_image_settings(
    ref_image: FibsemImage,
    image_settings: ImageSettings,
    beam_type: BeamType = BeamType.ELECTRON,
) -> ImageSettings:
    """Generate matching image settings from an image."""
    image_settings.resolution = (ref_image.data.shape[1], ref_image.data.shape[0])
    # image_settings.dwell_time = ref_image.metadata.scan_settings.dwell_time
    image_settings.hfw = ref_image.data.shape[1] * ref_image.metadata.pixel_size.x
    image_settings.beam_type = beam_type
    image_settings.save = True
    image_settings.label = current_timestamp()

    return image_settings


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
