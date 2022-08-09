
from fibsem import constants
import logging
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient

########################### SETUP 

# TODO: remove, unused?
def reset_state(microscope: SdbMicroscopeClient, settings: dict, application_file=None):
    """Reset the microscope state.
    Parameters
    ----------
    microscope : Autoscript microscope object.
    settings :  Dictionary of user input argument settings.
    application_file : str, optional
        Name of the application file for milling, by default None
    """
    microscope.patterning.clear_patterns()
    if application_file:  # optionally specified
        microscope.patterning.set_default_application_file(application_file)
    microscope.beams.ion_beam.scanning.resolution.value = settings["imaging"]["resolution"]
    microscope.beams.ion_beam.scanning.dwell_time.value = settings["imaging"]["dwell_time"]
    microscope.beams.ion_beam.horizontal_field_width.value =  settings["imaging"]["horizontal_field_width"]
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    return microscope


def setup_milling(
    microscope: SdbMicroscopeClient,
    application_file: str = "autolamella",
    patterning_mode: str = "Serial",
    hfw:float = 100e-6,
):
    """Setup for rectangle ion beam milling patterns.

    Parameters
    ----------
    microscope : AutoScript microscope instance.
        The AutoScript microscope object.
    application_file : str, optional
        Application file for ion beam milling, by default "autolamella"
    patterning_mode : str, optional
        Ion beam milling pattern mode, by default "Serial".
        The available options are "Parallel" or "Serial".
    hfw : float, optional
        Width of ion beam field of view in meters, by default 100e-6
    """
    microscope.imaging.set_active_view(2)  # the ion beam view
    microscope.patterning.set_default_beam_type(2)  # ion beam default
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = patterning_mode
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = hfw
    logging.info(f"milling: setup ion beam milling")
    logging.info(f"milling: application file:  {application_file}")
    logging.info(f"milling: patterning mode: {patterning_mode}")
    logging.info(f"milling: ion horizontal field width: {hfw}")

def run_milling(
    microscope: SdbMicroscopeClient,
    settings: dict,
    milling_current: float = None,
    asynch: bool = False,
):
    """Run ion beam milling at specified current.
    
    - Change to milling current
    - Run milling (synchronous) or Start Milling (asynchronous)

    """
    logging.info("milling: running ion beam milling now...")

    # change to milling current
    microscope.imaging.set_active_view(2)  # the ion beam view
    if milling_current is None:
        milling_current = settings["imaging"]["milling_current"]
    if microscope.beams.ion_beam.beam_current.value != milling_current:
        # if milling_current not in microscope.beams.ion_beam.beam_current.available_values:
        #   switch to closest

        microscope.beams.ion_beam.beam_current.value = milling_current

    # run milling (asynchronously)
    if asynch:
        microscope.patterning.start()
    else:
        microscope.patterning.run()
        microscope.patterning.clear_patterns()


def finish_milling(microscope: SdbMicroscopeClient, imaging_current: float = 20e-12) -> None:
    """Finish milling by clearing the patterns and restoring the default imaging current.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client connection
        settings (dict): configuration settings
    """
    # restore imaging current
    logging.info("returning to the ion beam imaging current now.")
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = imaging_current
    microscope.patterning.mode = "Serial"
    logging.info("ion beam milling complete.")


############################# UTILS #############################

def read_protocol_dictionary(settings, stage_name) -> list:

    # multi-stage
    if "protocol_stages" in settings[stage_name]:
        protocol_stages = []
        for stage_settings in settings[stage_name]["protocol_stages"]:
            tmp_settings = settings[stage_name].copy()
            tmp_settings.update(stage_settings)
            protocol_stages.append(tmp_settings)
    # single-stage
    else:
        protocol_stages = [settings[stage_name]]

    return protocol_stages


def calculate_milling_time(patterns: list, milling_current: float) -> float:

    from fibsem import config 

    # volume (width * height * depth) / total_volume_sputter_rate

    # calculate sputter rate
    if milling_current in config.MILLING_SPUTTER_RATE:
        total_volume_sputter_rate = config.MILLING_SPUTTER_RATE[milling_current]
    else:
        total_volume_sputter_rate = 3.920e-1

    # calculate total milling volume
    volume = 0
    for stage in patterns:
        for pattern in stage:
            width = pattern.width * constants.METRE_TO_MICRON
            height = pattern.height * constants.METRE_TO_MICRON
            depth = pattern.depth * constants.METRE_TO_MICRON
            volume += width * height * depth
    
    # estimate time
    milling_time_seconds = volume / total_volume_sputter_rate # um3 * 1/ (um3 / s) = seconds

    logging.info(f"WHDV: {width:.2f}um, {height:.2f}um, {depth:.2f}um, {volume:.2f}um3")
    logging.info(f"Volume: {volume:.2e}, Rate: {total_volume_sputter_rate:.2e} um3/s")
    logging.info(f"Milling Estimated Time: {milling_time_seconds / 60:.2f}m")

    return milling_time_seconds

# TODO: MillingSettings dataclass


### PATTERNING

def _draw_rectangle_pattern(microscope:SdbMicroscopeClient, settings:dict , x: float = 0.0, y: float = 0.0):

    if settings["cleaning_cross_section"]:
        pattern = microscope.patterning.create_cleaning_cross_section(
        center_x=x,
        center_y=y,
        width=settings["width"],
        height=settings["height"],
        depth=settings["depth"],
    )
    else:
        pattern = microscope.patterning.create_rectangle(
            center_x=x,
            center_y=y,
            width=settings["width"],
            height=settings["height"],
            depth=settings["depth"],
        )
    
    # need to make each protocol setting have these....which means validation
    pattern.rotation=np.deg2rad(settings["rotation"])
    pattern.scan_direction = settings["scan_direction"]

    return pattern