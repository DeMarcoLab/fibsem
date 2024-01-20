import logging

import numpy as np
from fibsem.microscope import FibsemMicroscope
from fibsem import calibration
from fibsem.structures import (
    BeamSystemSettings,
    BeamType,
    ImageSettings,
    MicroscopeSettings,
    MicroscopeState,
    FibsemImage,
)


# TODO: change return type to list of warnings rather than reading the log...
def validate_initial_microscope_state(
    microscope: FibsemMicroscope, settings: MicroscopeSettings
) -> None:
    """Set the initial microscope state to default, and validate other settings."""

    # validate chamber state
    _validate_chamber_state(microscope=microscope)

    # validate stage calibration (homed, linked)
    _validate_stage_calibration(microscope=microscope)

    # validate needle calibration (needle calibration, retracted)
    _validate_needle_calibration(microscope=microscope)

    # validate beam settings and calibration
    _validate_beam_system_settings(microscope=microscope, settings=settings)

    # validate scan rotation
    microscope.set("scan_rotation", value=0.0, beam_type=BeamType.ION)



def _validate_stage_calibration(microscope: FibsemMicroscope) -> None:
    """Validate if the required stage calibration has been performed.

    Args:
        microscope (FibsemMicroscope): autoscript microscope instance
    """
    # QUERY: should we home?
    if not microscope.get("stage_homed"):
        logging.warning("Stage is not homed.")

    # QUERY: should we focus and link?
    if not microscope.get("stage_linked"):
        logging.warning("Stage is not linked.")

    logging.info("Stage calibration validation complete.")

    return


def _validate_needle_calibration(microscope: FibsemMicroscope) -> None:
    """Validate if the needle is inserted
    Args:
        microscope (FibsemMicroscope): autoscript microscope instance
    """

    needle_state = microscope.get("manipulator_state")
    logging.info(f"Needle is {needle_state}")

    if needle_state != "Retracted":
        logging.warning(f"Needle is {needle_state}. Please retract before starting.")

    # movement.retract_needle() # TODO: decide whether or not to do this
    # TODO: calibrate needle? check if needle has been calibrated? how

    return


# TODO: use _set_type_mode for setting the detector type and mode not directly setting the values


def _validate_beam_system_state(
    microscope: FibsemMicroscope, settings: BeamSystemSettings
) -> None:

    beam_name = settings.beam_type.name

    logging.info(f"Validating {beam_name} Beam")
    if not microscope.get("on", settings.beam_type):
        logging.warning(f"{beam_name} Beam is not on, switching on now...")
        microscope.set("on", True, settings.beam_type)
        assert microscope.get("on", settings.beam_type), f"Unable to turn on {beam_name} Beam."
        logging.warning(f"{beam_name} Beam turned on.")

    # blanked?
    if microscope.get("blanked", settings.beam_type):
        logging.warning(f"{beam_name} Beam is blanked, unblanking now...")
        microscope.set("blanked", False, settings.beam_type)
        assert not microscope.get("blanked", settings.beam_type), f"Unable to unblank {beam_name} Beam."
        logging.warning(f"{beam_name} Beam unblanked.")

    # set detectors
    microscope.set("detector_type", settings.detector_type, settings.beam_type)
    microscope.set("detector_mode", settings.detector_mode, settings.beam_type)       

    # validate working distances
    if not check_working_distance_is_within_tolerance(
        microscope, settings, atol=0.5e-3
    ):
        logging.warning(
            f"""{beam_name} Beam is not close to eucentric height. It should be {settings.eucentric_height}m
            (Currently is {microscope.get("working_distance", settings.beam_type):.4f}m)"""
        )

    # validate high voltage
    high_voltage_limits = str(microscope.get("voltage_limits", settings.beam_type))
    logging.info(f"{beam_name} Beam High Voltage Limits are: {high_voltage_limits}")

    if microscope.get("voltage", settings.beam_type) != settings.voltage:
        logging.warning(
            f"{beam_name} Beam High Voltage should be {settings.voltage}V (Currently {microscope.get('voltage', settings.beam_type)}V)"
        )

        if bool(microscope.get("voltage_controllable", settings.beam_type)):
            logging.warning(
                f"Changing {beam_name} Beam High Voltage to {settings.voltage}V."
            )
            microscope.set("voltage", settings.voltage, settings.beam_type)
            assert (
                microscope.get("voltage", settings.beam_type) == settings.voltage
            ), f"Unable to change {beam_name} Beam High Voltage"
            logging.warning(f"{beam_name} Beam High Voltage Changed")

    # validate plasma gas (only for ION beam)
    if settings.beam_type is BeamType.ION:

        plasma_gas = settings.plasma_gas.capitalize()
        if plasma_gas not in microscope.get_available(key="plasma_gas", beam_type=settings.beam_type):
            logging.warning(f"{plasma_gas} is not available as a plasma gas.")

        current_plasma_gas = microscope.get("plasma_gas", settings.beam_type)
        if current_plasma_gas != plasma_gas:
            logging.warning(f"Plasma Gas is should be {plasma_gas} (Currently {current_plasma_gas})")

def _validate_beam_system_settings(
    microscope: FibsemMicroscope, settings: MicroscopeSettings
) -> None:
    """Validate Beam Settings"""

    # electron beam
    _validate_beam_system_state(microscope, settings.system.electron)

    # ion beam
    _validate_beam_system_state(microscope, settings.system.ion)


def _validate_chamber_state(microscope: FibsemMicroscope) -> None:
    """Validate the state of the chamber"""

    chamber_state = str(microscope.get('chamber_state'))
    chamber_pressure = microscope.get("chamber_pressure")
    
    logging.info(f"Vacuum Chamber State: {chamber_state}")
    if not chamber_state == "Pumped":
        logging.warning(
            f"Chamber vacuum state should be Pumped (Currently is {chamber_state})"
        )

    logging.info(f"Vacuum Chamber Pressure: {chamber_pressure:.6f} mbar"
    )
    if chamber_pressure >= 1e-4:
        logging.warning(f"Chamber pressure is too high, please pump the system (Currently {chamber_pressure:.6f} mbar)"            )

    logging.info(f"Vacuum Chamber State Validation finished.")

def validate_stage_height_for_needle_insertion(
    microscope: FibsemMicroscope, needle_stage_height_limit: float = 3.7e-3
) -> bool:
    """Check if the needle can be inserted, based on the current stage height.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope limit
        needle_stage_height_limit (float, optional): minimum stage height limit. Defaults to 3.7e-3.

    Returns:
        bool: needle is insertable
    """

    stage_position = microscope.get_stage_position()

    # Unable to insert the needle if the stage height is below this limit (3.7e-3)
    return bool(stage_position.z > needle_stage_height_limit)


def validate_focus(
    microscope: FibsemMicroscope,
    settings: BeamSystemSettings,
    link: bool = True,
) -> bool:

    # check focus distance is within tolerance
    # if link:
    #     calibration.auto_link_stage(microscope)

    return check_working_distance_is_within_tolerance(microscope, settings=settings)


def check_working_distance_is_within_tolerance(
    microscope: FibsemMicroscope, settings: BeamSystemSettings, atol=0.5e-3
) -> bool:

    working_distance = microscope.get("working_distance", settings.beam_type)
    eucentric_height = settings.eucentric_height

    logging.info(
        f"{settings.beam_type.name} Beam working distance is {working_distance:.4f}m"
    )
    return np.isclose(working_distance, eucentric_height, atol=atol)


def check_shift_within_tolerance(
    dx: float, dy: float, ref_image: FibsemImage, limit: float = 0.25
) -> bool:
    """Check if required shift is wihtin safety limit"""
    # check if the cross correlation movement is within the safety limit
    
    pixelsize_x = ref_image.metadata.pixel_size.x
    width, height = ref_image.metadata.image_settings.resolution
    X_THRESHOLD = limit * pixelsize_x * width
    Y_THRESHOLD = limit * pixelsize_x * height

    return abs(dx) < X_THRESHOLD and abs(dy) < Y_THRESHOLD



def _validate_milling_protocol(
    stage_protocol: dict, settings: MicroscopeSettings
) -> dict:

    if "milling_current" not in stage_protocol:
        stage_protocol["milling_current"] = settings.milling.milling_current
    if "cleaning_cross_section" not in stage_protocol:
        stage_protocol["cleaning_cross_section"] = False
    if "rotation" not in stage_protocol:
        stage_protocol["rotation"] = 0.0
    if "scan_direction" not in stage_protocol:
        stage_protocol["scan_direction"] = "TopToBottom"

    # remove list element from settings
    if "protocol_stages" in stage_protocol:
        del stage_protocol["protocol_stages"]

    return stage_protocol


def _validate_configuration_values(microscope: FibsemMicroscope, dictionary: dict):
    """Recursively traverse dictionary and validate all parameters.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        dictionary (dict): settings dictionary

    Returns:
        dict: validated settings dictionary
    """
    # TODO: this function is inefficient, recursive calls can be wasteful
    for key, item in dictionary.items():
        if isinstance(item, dict):
            _validate_configuration_values(microscope, item)
        elif isinstance(item, list):
            dictionary[key] = [
                _validate_configuration_values(microscope, i)
                for i in item
                if isinstance(i, dict)
            ]
        else:
            if isinstance(item, float):
                if "hfw" in key:
                    microscope.check_available_values(key, values=[item])

                if "milling_current" in key:
                    microscope.check_available_values("current", values=[item], beam_type=BeamType.ION)
                
                if "resolution" in key:
                    microscope.check_available_values(key, values=[item])

                if "dwell_time" in key:
                    microscope.check_available_values(key, values=[item])
                    
            if isinstance(item, str):
                if "application_file" in key:
                    microscope.check_available_values("application_file", values=[item])

    return dictionary



# new validation v2

from fibsem import utils
from fibsem.structures import BeamType
from fibsem.microscope import FibsemMicroscope
import logging


def validate_microscope(microscope: FibsemMicroscope):
    # append exceptions to list
    warnings = []

    # check beams are on
    if not microscope.get("on", BeamType.ELECTRON):
        warnings.append("Electron beam is off")

    if not microscope.get("on", BeamType.ION):
        warnings.append("Ion beam is off")

    # check chamber is pumped
    if microscope.get("chamber_state") != "Pumped":
        warnings.append("Chamber is not pumped")


    # ThermoFisher specific validation
    from fibsem.microscope import ThermoMicroscope, DemoMicroscope
    if isinstance(microscope, (ThermoMicroscope, DemoMicroscope)):
        # check stage is homed
        if not microscope.get("stage_homed"):
            warnings.append("Stage is not homed")

        # check stage is linked
        if not microscope.get("stage_linked"):
            warnings.append("Stage is not linked")

        # check needle is retracted
        if microscope.get("manipulator_state") != "Retracted":
            warnings.append("Needle is not retracted")


    logging.warning(f"Microscope Validation Warnings: {warnings}")

    return warnings


if __name__ == "__main__":

    microscope, settings = utils.setup_session()

    print(microscope.get("on", BeamType.ELECTRON))
    print(microscope.get("on", BeamType.ION))

    # microscope.set("on", True, BeamType.ELECTRON)
    # microscope.set("on", True, BeamType.ION)


    # get the system is pumped
    print("Chamber State: ", microscope.get("chamber_state"))
    print("Chamber Pressure: ", microscope.get("chamber_pressure"))

    print("Chamber State: ", microscope.get("chamber_state"))
    print("Chamber Pressure: ", microscope.get("chamber_pressure"))

    warnings = validate_microscope(microscope)

    print(warnings)