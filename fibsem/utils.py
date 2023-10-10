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

def _get_positions(fname: str = None) -> list[str]:    
    
    from fibsem import config as cfg
    from fibsem.structures import FibsemStagePosition
    import os

    if fname is None:
        fname = os.path.join(cfg.CONFIG_PATH, "positions.yaml")

    ddict = load_yaml(fname=fname)

    return [d["name"] for d in ddict]

def _display_metadata(img: FibsemImage, timezone: str = 'Australia/Sydney', show: bool = True):
    import pytz
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.pyplot as plt
    import fibsem.constants as constants
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img.data, cmap='gray')
    image_height, image_width = img.data.shape
    # Hide axis
    ax.axis('off')
    ax.set_xlim(0, image_width)  # Set the width of the image
    ax.set_ylim(0, image_height)  # Set the height of the image
    # Create a list to store metadata lines
    if img.metadata.image_settings.beam_type == BeamType.ELECTRON:
        metadata_lines = "Electron Beam \n"
    else: 
        metadata_lines = "Ion Beam \n"

    # add metadata lines
    if img.metadata.image_settings.beam_type == BeamType.ELECTRON and img.metadata.microscope_state.eb_settings.voltage is not None:
        metadata_lines += f'{img.metadata.microscope_state.eb_settings.voltage * constants.SI_TO_KILO} kV  |  '
    elif img.metadata.image_settings.beam_type == BeamType.ION and img.metadata.microscope_state.ib_settings.voltage is not None:
        metadata_lines += f'{img.metadata.microscope_state.ib_settings.voltage * constants.SI_TO_KILO} kV  |  '
    else:
        metadata_lines += 'Voltage: Unknown  |  '
    metadata_lines += (f'HFW: {img.metadata.image_settings.hfw * constants.SI_TO_MICRO} μm  | ')
    metadata_lines += (f'{img.metadata.image_settings.resolution[0]} x {img.metadata.image_settings.resolution[1]}  |  ')

    desired_timezone = pytz.timezone(timezone)  
    timestamp = img.metadata.microscope_state.timestamp

    if isinstance(timestamp, str):
        timestamp_format = "%m/%d/%Y %H:%M:%S"
        timestamp = datetime.datetime.strptime(img.metadata.microscope_state.timestamp, timestamp_format)

    if isinstance(timestamp, (int,float)):
        timestamp_str = datetime.datetime.fromtimestamp(timestamp, tz=desired_timezone).strftime('%Y-%m-%d %I:%M %p')    
    
    if isinstance(timestamp, datetime.datetime):
        timestamp_str = timestamp.astimezone(desired_timezone).strftime('%Y-%m-%d %I:%M %p')         

    metadata_lines += (f"{timestamp_str}")

    # add empty char to second line to fill up space
    _line2 = metadata_lines.split("\n")[1]
    metadata_lines += " " * (70 - len(_line2))

    metadata_rect = plt.text(
            0.01, 0.03, metadata_lines,
            transform=ax.transAxes,
            fontsize=10,
            color='white',
            bbox=dict(facecolor='black', alpha=0.7),
            ha='left',
        )

    metadata_rect.set_clip_box(dict(width=1.0))
    scale = (img.metadata.image_settings.hfw * constants.SI_TO_MICRO) / img.data.shape[1]
    
    # transparent background
    scalebar = ScaleBar(scale, "um", 
        color="black", box_color="white", box_alpha=0.3) 

    plt.gca().add_artist(scalebar)

    if show:
        plt.show()

    return fig

# TODO: re-think this, dont like the pop ups
def _register_metadata(microscope: FibsemMicroscope, application_software: str, application_software_version: str, experiment_name: str, experiment_method: str) -> None:
    from fibsem.structures import FibsemUser, FibsemExperiment
    import fibsem

    import platform
    username = os.environ.get("USERNAME", "username")

    if platform.system() == "Windows":
        hostname = os.environ.get("COMPUTERNAME", "hostname")
    elif platform.system() == "Linux":
        hostname = os.environ.get("HOSTNAME", "hostname")
    elif platform.system() == "Darwin":
        hostname = os.environ.get("HOSTNAME", "hostname")
    else:
        hostname = "hostname"
        
    user = FibsemUser(name=username, email="null", organization="null", computer=hostname)
        
    experiment = FibsemExperiment(
        id = experiment_name,
        method=experiment_method,
        application=application_software, 
        fibsem_version=fibsem.__version__,
        application_version=application_software_version,
    )
    microscope.user = user
    microscope.experiment = experiment










################## MIGRATE FROM OLD AUTOLIFTOUT
# import os
# from pathlib import Path

# import numpy as np
# from fibsem.structures import ImageSettings

# def plot_two_images(img1, img2) -> None:
#     import matplotlib.pyplot as plt
#     from fibsem.structures import Point

#     c = Point(img1.data.shape[1] // 2, img1.data.shape[0] // 2)

#     fig, ax = plt.subplots(1, 2, figsize=(30, 30))
#     ax[0].imshow(img1.data, cmap="gray")
#     ax[0].plot(c.x, c.y, "y+", ms=50, markeredgewidth=2)
#     ax[1].imshow(img2.data, cmap="gray")
#     ax[1].plot(c.x, c.y, "y+", ms=50, markeredgewidth=2)
#     plt.show()


# def take_reference_images_and_plot(microscope, image_settings: ImageSettings):
#     from pprint import pprint

#     from fibsem import acquire

#     eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
#     plot_two_images(eb_image, ib_image)

#     return eb_image, ib_image


# # cross correlate
# def crosscorrelate_and_plot(
#     ref_image,
#     new_image,
#     rotate: bool = False,
#     lp: int = 128,
#     hp: int = 8,
#     sigma: int = 6,
#     ref_mask: np.ndarray = None,
#     xcorr_limit: int = None
# ):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from fibsem import alignment
#     from fibsem.structures import Point
#     from fibsem.imaging import utils as image_utils

#     # rotate ref
#     if rotate:
#         ref_image = image_utils.rotate_image(ref_image)

#     dx, dy, xcorr = alignment.shift_from_crosscorrelation(
#         ref_image,
#         new_image,
#         lowpass=lp,
#         highpass=hp,
#         sigma=sigma,
#         use_rect_mask=True,
#         ref_mask=ref_mask,
#         xcorr_limit=xcorr_limit
#     )

#     pixelsize = ref_image.metadata.binary_result.pixel_size.x
#     dx_p, dy_p = int(dx / pixelsize), int(dy / pixelsize)

#     print(f"shift_m: {dx}, {dy}")
#     print(f"shift_px: {dx_p}, {dy_p}")

#     shift = np.roll(new_image.data, (-dy_p, -dx_p), axis=(0, 1))

#     mid = Point(shift.shape[1] // 2, shift.shape[0] // 2)

#     if ref_mask is None:
#         ref_mask = np.ones_like(ref_image.data)

#     fig, ax = plt.subplots(1, 4, figsize=(30, 30))
#     ax[0].imshow(ref_image.data * ref_mask , cmap="gray")
#     ax[0].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
#     ax[0].set_title(f"Reference (rotate={rotate})")
#     ax[1].imshow(new_image.data, cmap="gray")
#     ax[1].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
#     ax[1].set_title(f"New Image")
#     ax[2].imshow(xcorr, cmap="turbo")
#     ax[2].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
#     ax[2].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2)
#     ax[2].set_title("XCORR")
#     ax[3].imshow(shift, cmap="gray")
#     ax[3].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2, label="new_position")
#     ax[3].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2, label="old_position")
#     ax[3].set_title("New Image Shifted")
#     ax[3].legend()
#     plt.show()

#     return dx, dy, xcorr


# def plot_crosscorrelation(ref_image, new_image, dx, dy, xcorr):
#     import matplotlib.pyplot as plt
#     from fibsem.structures import Point

#     pixelsize = ref_image.metadata.binary_result.pixel_size.x
#     dx_p, dy_p = int(dx / pixelsize), int(dy / pixelsize)

#     print(f"shift_m: {dx}, {dy}")
#     print(f"shift_px: {dx_p}, {dy_p}")

#     shift = np.roll(new_image.data, (-dy_p, -dx_p), axis=(0, 1))

#     mid = Point(shift.shape[1] // 2, shift.shape[0] // 2)

#     fig, ax = plt.subplots(1, 4, figsize=(30, 30))
#     ax[0].imshow(ref_image.data, cmap="gray")
#     ax[0].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
#     ax[0].set_title(f"Reference)")
#     ax[1].imshow(new_image.data, cmap="gray")
#     ax[1].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
#     ax[1].set_title(f"New Image")
#     ax[2].imshow(xcorr, cmap="turbo")
#     ax[2].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
#     ax[2].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2)
#     ax[2].set_title("XCORR")
#     ax[3].imshow(shift, cmap="gray")
#     ax[3].plot(mid.x, mid.y, color="lime", marker="+", ms=50, markeredgewidth=2)
#     ax[3].plot(mid.x - dx_p, mid.y - dy_p, "m+", ms=50, markeredgewidth=2)
#     ax[3].set_title("New Image Shifted")
#     plt.show()