import datetime
import glob
import json
import logging
import os
import time
from pathlib import Path
import sys 

import yaml
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    ManipulatorPosition,
)
from PIL import Image
import fibsem
from fibsem.structures import (
    BeamType,
    MicroscopeSettings,
    ImageSettings,
    SystemSettings,
    DefaultSettings,
)


def connect_to_microscope(ip_address="10.0.0.1", port: int = 7520) -> SdbMicroscopeClient:
    """Connect to the FIBSEM microscope."""
    try:
        # TODO: get the port
        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        microscope = SdbMicroscopeClient()
        microscope.connect(host=ip_address, port=port)
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")
    except Exception as e:
        logging.error(f"Unable to connect to the microscope: {e}")
        microscope = None

    return microscope


def sputter_platinum(
    microscope: SdbMicroscopeClient,
    protocol: dict,
    whole_grid: bool = False,
    default_application_file: str = "autolamella",
):
    """Sputter platinum over the sample.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        protocol (dict): platinum protcol dictionary
        whole_grid (bool, optional): sputtering protocol. Defaults to False.

    Raises:
        RuntimeError: Error Sputtering
    """

    # protcol = settings.protocol["platinum"] in old system
    # protocol = settings.protocol["platinum"] in new
    if whole_grid:

        sputter_time = protocol["whole_grid"]["time"]  # 20
        hfw = protocol["whole_grid"]["hfw"]  # 30e-6
        line_pattern_length = protocol["whole_grid"]["length"]  # 7e-6
        logging.info("sputtering platinum over the whole grid.")
    else:
        sputter_time = protocol["weld"]["time"]  # 20
        hfw = protocol["weld"]["hfw"]  # 100e-6
        line_pattern_length = protocol["weld"]["length"]  # 15e-6
        logging.info("sputtering platinum to weld.")

    # Setup
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(BeamType.ELECTRON.value)
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(protocol["application_file"])
    microscope.patterning.set_default_beam_type(BeamType.ELECTRON.value)
    multichem = microscope.gas.get_multichem()
    multichem.insert(protocol["position"])
    multichem.turn_heater_on(protocol["gas"])  # "Pt cryo")
    time.sleep(3)

    # Create sputtering pattern
    microscope.beams.electron_beam.horizontal_field_width.value = hfw
    pattern = microscope.patterning.create_line(
        -line_pattern_length / 2,  # x_start
        +line_pattern_length,  # y_start
        +line_pattern_length / 2,  # x_end
        +line_pattern_length,  # y_end
        2e-6,
    )  # milling depth
    pattern.time = sputter_time + 0.1

    # Run sputtering
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        logging.info("Sputtering with platinum for {} seconds...".format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
        time.sleep(sputter_time + 5)
    else:
        raise RuntimeError("Can't sputter platinum, patterning state is not ready.")
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(microscope.patterning.state))
        logging.warning("Consider adjusting the patterning line depth.")

    # Cleanup
    microscope.patterning.clear_patterns()
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(BeamType.ION.value)  # set ion beam
    multichem.retract()
    logging.info("sputtering platinum finished.")


def save_image(image: AdornedImage, save_path: Path, label: str = "image"):
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, f"{label}.tif")
    image.save(path)


def current_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d.%I-%M-%S%p")


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
        force=True
    )

    return logfile


def load_yaml(fname: Path) -> dict:

    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    return config

def save_yaml(path: Path, data: dict) -> None:

    with open(path, "w") as f:
        yaml.dump(data, f, indent=4)


def save_needle_yaml(path: Path, position: ManipulatorPosition) -> None:
    """Save the manipulator position from disk"""
    from fibsem.structures import manipulator_position_to_dict

    with open(os.path.join(path, "needle.yaml"), "w") as f:
        yaml.dump(manipulator_position_to_dict(position), f, indent=4)


def load_needle_yaml(path: Path) -> ManipulatorPosition:
    """Load the manipulator position from disk"""
    from fibsem.structures import manipulator_position_from_dict

    position_dict = load_yaml(os.path.join(path, "needle.yaml"))
    position = manipulator_position_from_dict(position_dict)

    return position

from fibsem.structures import MicroscopeState
def save_state_yaml(path: Path, state: MicroscopeState) -> None:

    state_dict = state.__to_dict__()

    save_yaml(path, state_dict)


def get_updated_needle_insertion_position(path: Path) -> ManipulatorPosition:

    position = None

    # if os.path.exists(os.path.join(path, "needle.yaml")):

    #     position = load_needle_yaml(path)

    return position


def save_metadata(settings: MicroscopeSettings, path: Path):
    # TODO: finish this
    pass
    # settings_dict = settings.__to_dict__()

    # fname = os.path.join(path, "metadata.json")
    # with open(fname, "w") as fp:
    #     json.dump(settings_dict, fp, sort_keys=True, indent=4)


def create_gif(path: Path, search: str, gif_fname: str, loop: int = 0) -> None:
    filenames = glob.glob(os.path.join(path, search))

    imgs = [Image.fromarray(AdornedImage.load(fname).data) for fname in filenames]

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
) -> tuple[SdbMicroscopeClient, MicroscopeSettings]:
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
    microscope = connect_to_microscope(ip_address=settings.system.ip_address)

    # image_setttings
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
        config_path = os.path.join(os.path.dirname(fibsem.__file__), "config")

    # system settings
    settings = load_yaml(os.path.join(config_path, "system.yaml"))
    system_settings = SystemSettings.__from_dict__(settings["system"])

    # user settings
    default_settings = DefaultSettings.__from_dict__(settings["user"])
    image_settings = ImageSettings.__from_dict__(settings["user"])

    # protocol settings
    protocol = load_protocol(protocol_path)

    settings = MicroscopeSettings(
        system=system_settings,
        image=image_settings,
        protocol=protocol,
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
    ref_image: AdornedImage,
    image_settings: ImageSettings,
    beam_type: BeamType = BeamType.ELECTRON,
) -> ImageSettings:
    """Generate matching image settings from an image."""
    image_settings.resolution = f"{ref_image.width}x{ref_image.height}"
    # image_settings.dwell_time = ref_image.metadata.scan_settings.dwell_time
    image_settings.hfw = ref_image.width * ref_image.metadata.binary_result.pixel_size.x
    image_settings.beam_type = beam_type
    image_settings.save = True
    image_settings.label = current_timestamp()

    return image_settings


def log_current_figure(fig, name: str):
    # save with timestamp
    import matplotlib.pyplot as plt
    import datetime
    import os
    from liftout.config import config

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    day = now.strftime("%Y-%m-%d")
    os.makedirs(os.path.join(config.LOG_DATA_PATH, day), exist_ok=True)

    try:
        plt.savefig(os.path.join(config.LOG_DATA_PATH, day, f"{name}_{timestamp}.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error: {e}")