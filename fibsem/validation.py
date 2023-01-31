import logging

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import (
    CoordinateSystem,
    ManipulatorState,
)
from autoscript_sdb_microscope_client.structures import AdornedImage
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
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings
) -> None:
    """Set the initial microscope state to default, and validate other settings."""

    # set default microscope state
    set_initial_microscope_state(microscope, None)

    # validate chamber state
    _validate_chamber_state(microscope=microscope)

    # validate stage calibration (homed, linked)
    _validate_stage_calibration(microscope=microscope)

    # validate needle calibration (needle calibration, retracted)
    _validate_needle_calibration(microscope=microscope)

    # validate beam settings and calibration
    _validate_beam_system_settings(microscope=microscope, settings=settings)

    # validate scan rotation
    _validate_scanning_rotation(microscope=microscope)


# TODO: need to get this initial state from the user some how...
def set_initial_microscope_state(
    microscope: SdbMicroscopeClient, microscope_state: MicroscopeState
) -> None:

    # set default microscope state
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    # set state
    # calibration.set_microscope_state(microscope, microscope_state)

    return


def _validate_stage_calibration(microscope: SdbMicroscopeClient) -> None:
    """Validate if the required stage calibration has been performed.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
    """
    # QUERY: should we home?
    if not microscope.specimen.stage.is_homed:
        logging.warning("Stage is not homed.")

    # QUERY: should we focus and link?
    if not microscope.specimen.stage.is_linked:
        logging.warning("Stage is not linked.")

    logging.info("Stage calibration validation complete.")

    return


def _validate_needle_calibration(microscope: SdbMicroscopeClient) -> None:
    """Validate if the needle is inserted
    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
    """

    needle_state = microscope.specimen.manipulator.state
    logging.info(f"Needle is {needle_state}")

    if needle_state != "Retracted":
        logging.warning(f"Needle is {needle_state}. Please retract before starting.")

    # movement.retract_needle() # TODO: decide whether or not to do this
    # TODO: calibrate needle? check if needle has been calibrated? how

    return


# TODO: use _set_type_mode for setting the detector type and mode not directly setting the values


def _validate_beam_system_state(
    microscope: SdbMicroscopeClient, settings: BeamSystemSettings
) -> None:

    if settings.beam_type is BeamType.ELECTRON:
        microscope_beam = microscope.beams.electron_beam
    if settings.beam_type is BeamType.ION:
        microscope_beam = microscope.beams.ion_beam

    beam_name = settings.beam_type.name

    logging.info(f"Validating {beam_name} Beam")
    if not microscope_beam.is_on:
        logging.warning(f"{beam_name} Beam is not on, switching on now...")
        microscope_beam.turn_on()
        assert microscope_beam.is_on, f"Unable to turn on {beam_name} Beam."
        logging.warning(f"{beam_name} Beam turned on.")

    # blanked?
    if microscope_beam.is_blanked:
        logging.warning(f"{beam_name} Beam is blanked, unblanking now...")
        microscope_beam.unblank()
        assert not microscope_beam.is_blanked, f"Unable to unblank {beam_name} Beam."
        logging.warning(f"{beam_name} Beam unblanked.")

    # set beam active view and device
    microscope.imaging.set_active_view(settings.beam_type.value)
    microscope.imaging.set_active_device(settings.beam_type.value)

    # validate detector type
    if str(microscope.detector.type.value) != settings.detector_type:
        logging.warning(
            f"{beam_name} detector type should be {settings.detector_type} (Currently is {str(microscope.detector.type.value)})"
        )
        if settings.detector_type in microscope.detector.type.available_values:
            microscope.detector.type.value = settings.detector_type
            logging.warning(
                f"Changed {beam_name} detector type to {str(microscope.detector.type.value)}"
            )

    # validate detector mode
    if str(microscope.detector.mode.value) != settings.detector_mode:
        logging.warning(
            f"{beam_name} detector mode should be {settings.detector_mode} (Currently is {str(microscope.detector.mode.value)}"
        )
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = settings.detector_mode
            logging.warning(
                f"Changed {beam_name} detector mode to {str(microscope.detector.mode.value)}"
            )

    # validate working distances
    if not check_working_distance_is_within_tolerance(
        microscope, settings, atol=0.5e-3
    ):
        logging.warning(
            f"""{beam_name} Beam is not close to eucentric height. It should be {settings.eucentric_height}m
            (Currently is {microscope_beam.working_distance.value:.4f}m)"""
        )

    # validate high voltage
    high_voltage_limits = str(microscope_beam.high_voltage.limits)
    logging.info(f"{beam_name} Beam High Voltage Limits are: {high_voltage_limits}")

    if microscope_beam.high_voltage.value != settings.voltage:
        logging.warning(
            f"{beam_name} Beam High Voltage should be {settings.voltage}V (Currently {microscope_beam.high_voltage.value}V)"
        )

        if bool(microscope_beam.high_voltage.is_controllable):
            logging.warning(
                f"Changing {beam_name} Beam High Voltage to {settings.voltage}V."
            )
            microscope_beam.high_voltage.value = settings.voltage
            assert (
                microscope_beam.high_voltage.value == settings.voltage
            ), f"Unable to change {beam_name} Beam High Voltage"
            logging.warning(f"{beam_name} Beam High Voltage Changed")

    # validate plasma gas (only for ION beam)
    if settings.beam_type is BeamType.ION:

        plasma_gas = settings.plasma_gas.capitalize()
        if plasma_gas not in microscope_beam.source.plasma_gas.available_values:
            logging.warning(f"{plasma_gas} is not available as a plasma gas.")

        if microscope_beam.source.plasma_gas.value != plasma_gas:
            logging.warning(
                f"Plasma Gas is should be {plasma_gas} (Currently {microscope_beam.source.plasma_gas.value})"
            )


def _validate_beam_system_settings(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings
) -> None:
    """Validate Beam Settings"""

    # electron beam
    _validate_beam_system_state(microscope, settings.system.electron)

    # ion beam
    _validate_beam_system_state(microscope, settings.system.ion)


def _validate_chamber_state(microscope: SdbMicroscopeClient) -> None:
    """Validate the state of the chamber"""

    logging.info(
        f"Validating Vacuum Chamber State: {str(microscope.vacuum.chamber_state)}"
    )
    if not str(microscope.vacuum.chamber_state) == "Pumped":
        logging.warning(
            f"Chamber vacuum state should be Pumped (Currently is {str(microscope.vacuum.chamber_state)})"
        )

    logging.info(
        f"Validating Vacuum Chamber Pressure: {microscope.vacuum.chamber_pressure.value:.6f} mbar"
    )
    if microscope.vacuum.chamber_pressure.value >= 1e-4:
        logging.warning(
            f"Chamber pressure is too high, please pump the system (Currently {microscope.vacuum.chamber_pressure.value:.6f} mbar)"
        )

    logging.info(f"Vacuum Chamber State Validation finished.")


def validate_stage_height_for_needle_insertion(
    microscope: SdbMicroscopeClient, needle_stage_height_limit: float = 3.7e-3
) -> bool:
    """Check if the needle can be inserted, based on the current stage height.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope limit
        needle_stage_height_limit (float, optional): minimum stage height limit. Defaults to 3.7e-3.

    Returns:
        bool: needle is insertable
    """

    stage = microscope.specimen.stage

    # Unable to insert the needle if the stage height is below this limit (3.7e-3)
    return bool(stage.current_position.z > needle_stage_height_limit)


def validate_focus(
    microscope: SdbMicroscopeClient,
    settings: BeamSystemSettings,
    link: bool = True,
) -> bool:

    # check focus distance is within tolerance
    if link:
        calibration.auto_link_stage(microscope)

    return check_working_distance_is_within_tolerance(microscope, settings=settings)


def check_working_distance_is_within_tolerance(
    microscope: SdbMicroscopeClient, settings: BeamSystemSettings, atol=0.5e-3
) -> bool:

    if settings.beam_type is BeamType.ELECTRON:
        microscope_beam = microscope.beams.electron_beam
    if settings.beam_type is BeamType.ION:
        microscope_beam = microscope.beams.ion_beam

    working_distance = microscope_beam.working_distance.value
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
        stage_protocol["milling_current"] = settings.default.milling_current
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


# validate microscope settings
def _validate_configuration_values(microscope: SdbMicroscopeClient, dictionary: dict):
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
                    _validate_hfw(microscope=microscope, horizontal_field_widths=[item])

                if "milling_current" in key:
                    _validate_ion_beam_currents(microscope, [item])

                if "imaging_current" in key:
                    _validate_electron_beam_currents(microscope, [item])

                if "resolution" in key:
                    _validate_scanning_resolutions(microscope, [item])

                if "dwell_time" in key:
                    _validate_dwell_time(microscope, [item])

            if isinstance(item, str):
                if "application_file" in key:
                    _validate_application_files(microscope, [item])

    return dictionary


def _validate_application_files(
    microscope: SdbMicroscopeClient, application_files: list[str]
) -> None:
    """Check that the user supplied application files exist on this system.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    application_files : list
        List of application files, eg: ['Si', 'Si_small']

    Raises
    ------
    ValueError
        Application file name not found in list of available application files.
    """
    available_files = microscope.patterning.list_all_application_files()
    for app_file in application_files:
        if app_file not in available_files:
            raise ValueError(
                "{} not found ".format(app_file)
                + "in list of available application files!\n"
                "Please choose one from the list: \n"
                "{}".format(available_files)
            )


def _validate_dwell_time(
    microscope: SdbMicroscopeClient, dwell_times: list[float]
) -> None:
    """Check that the user specified dwell times are within the limits.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    dwell_times : list
        List of dwell times, eg: [1e-7, 1e-6]

    Raises
    ------
    ValueError
        Dwell time is smaller than the minimum limit.
    ValueError
        Dwell time is larger than the maximum limit.
    """
    dwell_limits = microscope.beams.ion_beam.scanning.dwell_time.limits
    for dwell in dwell_times:
        if not isinstance(dwell, (int, float)):
            raise ValueError(
                "Dwell time {} must be a number!\n".format(dwell)
                + "Please choose a value between the limits: \n"
                "{}".format(dwell_limits)
            )
        if dwell < dwell_limits.min:
            raise ValueError(
                "{} dwell time is too small!\n".format(dwell)
                + "Please choose a value between the limits: \n"
                "{}".format(dwell_limits)
            )
        elif dwell > dwell_limits.max:
            raise ValueError(
                "{} dwell time is too large!\n".format(dwell)
                + "Please choose a value between the limits: \n"
                "{}".format(dwell_limits)
            )
        else:
            if dwell is np.nan:
                raise ValueError(
                    "{} dwell time ".format(dwell) + "is not a number!\n"
                    "Please choose a value between the limits:\n"
                    "{}".format(dwell_limits)
                )


def _validate_electron_beam_currents(
    microscope: SdbMicroscopeClient, electron_beam_currents: list[float]
) -> None:
    """Check that the user supplied electron beam current values are valid.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    electron_beam_currents : list
        List of electron beam currents, eg: [ 3e-10, 1e-09]

    Raises
    ------
    ValueError
        Beam current not within limits of available electron beam currents.
    """
    available_electron_beam_currents = (
        microscope.beams.electron_beam.beam_current.limits
    )
    for beam_current in electron_beam_currents:

        if not available_electron_beam_currents.is_in(beam_current):
            raise ValueError(
                "{} not found ".format(beam_current)
                + "in range of available electron beam currents!\n"
                "Please choose one from within the range: \n"
                "{}".format(available_electron_beam_currents)
            )


def _validate_ion_beam_currents(
    microscope: SdbMicroscopeClient, ion_beam_currents: list[float]
) -> None:
    """Check that the user supplied ion beam current values are valid.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    ion_beam_currents : list
        List of ion beam currents, eg: [ 3e-10, 1e-09]

    Raises
    ------
    ValueError
        Beam current not found in list of available ion beam currents.
    """
    available_ion_beam_currents = (
        microscope.beams.ion_beam.beam_current.available_values
    )
    # TODO: decide how strict we want to be on the available currents (e.g. exact or within range)
    for beam_current in ion_beam_currents:
        if beam_current <= min(available_ion_beam_currents) or beam_current >= max(
            available_ion_beam_currents
        ):
            raise ValueError(
                "{} not found ".format(beam_current)
                + "in list of available ion beam currents!\n"
                "Please choose one from the list: \n"
                "{}".format(available_ion_beam_currents)
            )


def _validate_hfw(
    microscope: SdbMicroscopeClient, horizontal_field_widths: list[float]
) -> None:
    """Check that the ion beam horizontal field width is within the limits.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    horizontal_field_widths : list
        List of ion beam horizontal field widths, eg: [50e-6, 100e-6]

    Raises
    ------
    ValueError
        Ion beam horizontal field width is smaller than the minimum limit.
    ValueError
        Ion beam horizontal field width is larger than the maximum limit.
    """
    hfw_limits = microscope.beams.ion_beam.horizontal_field_width.limits
    for hfw in horizontal_field_widths:
        if not isinstance(hfw, (int, float)):
            raise ValueError(
                "Horizontal field width must be a number!\n"
                "Please choose a value between the limits: \n"
                "{}".format(hfw_limits)
            )
        if hfw < hfw_limits.min:
            raise ValueError(
                "{} ".format(hfw) + "horizontal field width is too small!\n"
                "Please choose a value between the limits: \n"
                "{}".format(hfw_limits)
            )
        elif hfw > hfw_limits.max:
            raise ValueError(
                "{} ".format(hfw) + "horizontal field width is too large!\n"
                "Please choose a value between the limits: \n"
                "{}".format(hfw_limits)
            )
        else:
            if hfw is np.nan:
                raise ValueError(
                    "{} horizontal field width ".format(hfw) + "is not a number!\n"
                    "Please choose a value between the limits: \n"
                    "{}".format(hfw_limits)
                )


def _validate_scanning_resolutions(
    microscope: SdbMicroscopeClient, scanning_resolutions: list[str]
) -> None:
    """Check that the user supplied scanning resolution values are valid.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.
    scanning_resolutions : list
        List of scanning resolutions, eg: ['1536x1024', '3072x2048']

    Raises
    ------
    ValueError
        Resolution not found in list of available scanning resolutions.
    """
    available_resolutions = (
        microscope.beams.ion_beam.scanning.resolution.available_values
    )
    microscope.beams.ion_beam.beam_current.available_values
    for resolution in scanning_resolutions:
        if resolution not in available_resolutions:
            raise ValueError(
                "{} not found ".format(resolution)
                + "in list of available scanning resolutions!\n"
                "Please choose one from the list: \n"
                "{}".format(available_resolutions)
            )


def _validate_scanning_rotation(microscope: SdbMicroscopeClient) -> None:
    """Check the microscope scan rotation is zero.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.

    Raises
    ------
    ValueError
        Raise an error to warn the user if the scan rotation is not zero.
    """
    rotation = microscope.beams.ion_beam.scanning.rotation.value
    if rotation is None:
        microscope.beams.ion_beam.scanning.rotation.value = 0
        rotation = microscope.beams.ion_beam.scanning.rotation.value
    if not np.isclose(rotation, 0.0):
        raise ValueError(
            "Ion beam scanning rotation must be 0 degrees."
            "\nPlease change your system settings and try again."
            "\nCurrent rotation value is {}".format(rotation)
        )


def _validate_stage_coordinate_system(microscope: SdbMicroscopeClient) -> None:
    """Ensure the stage coordinate system is RAW.

    Parameters
    ----------
    microscope : Connected Autoscrpt microscope instance.

    Notes
    -----
    The two available stage coordinate systems are:
    1. CoordinateSystem.RAW
        Coordinate system based solely on location of stage.
        This coordinate system is not affected by any adjustments and should
        bring stage to the exactly same position on a particular microscope.
    2. CoordinateSystem.SPECIMEN
        Coordinate system based on location on specimen.
        This coordinate system is affected by various additional adjustments
        that make it easier to navigate on a particular specimen. The most
        important one is link between Z coordinate and working distance.
        Specimen coordinate system is also used in XTUI stage control panel.

    Users have reported unexpected/unwanted behaviour with the operation of
    autolamella in cases where the SPECIMEN coordinate system is used
    (i.e. if the Z-Y link checkbox is ticked in the XT GUI). Avoiding this
    problem is why this validation check is run.
    """
    from autoscript_sdb_microscope_client.enumerations import CoordinateSystem

    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)
