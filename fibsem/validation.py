import logging

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from fibsem import calibration, acquire, movement
from fibsem.structures import BeamType, ImageSettings, MicroscopeState, CalibrationSettings, MicroscopeSettings
import numpy as np

# TODO: change to SystemSettings, CalibrationSettings, ImagingSettings
# TODO: change return type to list of warnings rather than reading the log...
def validate_initial_microscope_state(
    microscope: SdbMicroscopeClient, settings: dict
) -> None:
    """Set the initial microscope state to default, and validate other settings."""

    # TODO: add validation checks for dwell time and resolution
    logging.info(
        f"Electron voltage: {microscope.beams.electron_beam.high_voltage.value:.2f}"
    )
    logging.info(
        f"Electron current: {microscope.beams.electron_beam.beam_current.value:.2f}"
    )

    # set default microscope state
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
    microscope.beams.ion_beam.beam_current.value = settings["calibration"]["imaging"][
        "imaging_current"
    ]
    microscope.beams.ion_beam.horizontal_field_width.value = settings["calibration"][
        "imaging"
    ]["horizontal_field_width"]
    microscope.beams.ion_beam.scanning.resolution.value = settings["calibration"][
        "imaging"
    ]["resolution"]
    microscope.beams.ion_beam.scanning.dwell_time.value = settings["calibration"][
        "imaging"
    ]["dwell_time"]

    microscope.beams.electron_beam.horizontal_field_width.value = settings[
        "calibration"
    ]["imaging"]["horizontal_field_width"]
    microscope.beams.electron_beam.scanning.resolution.value = settings["calibration"][
        "imaging"
    ]["resolution"]
    microscope.beams.electron_beam.scanning.dwell_time.value = settings["calibration"][
        "imaging"
    ]["dwell_time"]

    # validate chamber state
    validate_chamber(microscope=microscope)

    # validate stage calibration (homed, linked)
    validate_stage_calibration(microscope=microscope)

    # validate needle calibration (needle calibration, retracted)
    validate_needle_calibration(microscope=microscope)

    # validate beam settings and calibration
    validate_beams_calibration(microscope=microscope, settings=settings)

    # validate scan rotation
    _validate_scanning_rotation(microscope=microscope) 




# TODO: finish this...
def validate_initial_microscope_state_v2(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings
) -> None:
    """Set the initial microscope state to default, and validate other settings."""

    # set default microscope state
    set_initial_microscope_state(microscope, settings)

    # validate chamber state
    validate_chamber(microscope=microscope)

    # validate stage calibration (homed, linked)
    validate_stage_calibration(microscope=microscope)

    # validate needle calibration (needle calibration, retracted)
    validate_needle_calibration(microscope=microscope)

    # validate beam settings and calibration
    validate_beams_calibration(microscope=microscope, settings=settings)

    # validate scan rotation
    _validate_scanning_rotation(microscope=microscope) 


# TODO: should this be MicroscopeState instead?
def set_initial_microscope_state(microscope: SdbMicroscopeClient, settings: MicroscopeSettings) ->  None:

    # set default microscope state
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
    
    microscope.beams.ion_beam.high_voltage.value = settings.system.ion_voltage
    microscope.beams.ion_beam.beam_current.value = settings.system.ion_current
    microscope.beams.ion_beam.horizontal_field_width.value = settings.image_settings.hfw
    microscope.beams.ion_beam.scanning.resolution.value = settings.image_settings.resolution
    microscope.beams.ion_beam.scanning.dwell_time.value = settings.image_settings.dwell_time

    microscope.beams.electron_beam.high_voltage.value = settings.system.electron_voltage
    microscope.beams.electron_beam.beam_current.value = settings.system.electron_current
    microscope.beams.electron_beam.horizontal_field_width.value = settings.image_settings.hfw
    microscope.beams.electron_beam.scanning.resolution.value = settings.image_settings.resolution
    microscope.beams.electron_beam.scanning.dwell_time.value = settings.image_settings.dwell_time

    return


def validate_milling_settings(stage_settings: dict, settings: dict) -> dict:
    # validation?
    if "milling_current" not in stage_settings:
        stage_settings["milling_current"] = settings["calibration"]["imaging"][
            "milling_current"
        ]
    if "cleaning_cross_section" not in stage_settings:
        stage_settings["cleaning_cross_section"] = False
    if "rotation" not in stage_settings:
        stage_settings["rotation"] = 0.0
    if "scan_direction" not in stage_settings:
        stage_settings["scan_direction"] = "TopToBottom"

    # remove list element from settings
    if "protocol_stages" in stage_settings:
        del stage_settings["protocol_stages"]

    return stage_settings


def validate_stage_calibration(microscope: SdbMicroscopeClient) -> None:

    if not microscope.specimen.stage.is_homed:
        logging.warning("Stage is not homed.")

    if not microscope.specimen.stage.is_linked:
        logging.warning("Stage is not linked.")

    logging.info("Stage calibration validation complete.")

    return


def validate_needle_calibration(microscope: SdbMicroscopeClient) -> None:

    if str(microscope.specimen.manipulator.state) == "Retracted":
        logging.info("Needle is retracted")
    else:
        logging.warning("Needle is inserted. Please retract before starting.")
        # movement.retract_needle() # TODO: decide whether or not to do this

    # TODO: calibrate needle?

    return

# TODO: change to SystemSettings, CalibrationSettings
def validate_beams_calibration(microscope: SdbMicroscopeClient, settings: dict) -> None:
    """Validate Beam Settings"""

    high_voltage = float(settings["system"]["high_voltage"])  # ion
    plasma_gas = str(settings["system"]["plasma_gas"]).capitalize()

    # TODO: check beam blanks?
    # TODO: check electron voltage?

    logging.info("Validating Electron Beam")
    if not microscope.beams.electron_beam.is_on:
        logging.warning("Electron Beam is not on, switching on now...")
        microscope.beams.electron_beam.turn_on()
        assert microscope.beams.electron_beam.is_on, "Unable to turn on Electron Beam."
        logging.warning("Electron Beam turned on.")

    microscope.imaging.set_active_view(1)
    if str(microscope.detector.type.value) != "ETD":
        logging.warning(
            f"Electron detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})"
        )
        if "ETD" in microscope.detector.type.available_values:
            microscope.detector.type.value = "ETD"
            logging.warning(
                f"Changed Electron detector type to {str(microscope.detector.type.value)}"
            )

    if str(microscope.detector.mode.value) != "SecondaryElectrons":
        logging.warning(
            f"Electron detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}"
        )
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = "SecondaryElectrons"
            logging.warning(
                f"Changed Electron detector mode to {str(microscope.detector.mode.value)}"
            )

    # working distances
    logging.info(
        f"EB Working Distance: {microscope.beams.electron_beam.working_distance.value:.4f}m"
    )
    if not np.isclose(
        microscope.beams.electron_beam.working_distance.value,
        settings["calibration"]["limits"]["eucentric_height_eb"],
        atol=settings["calibration"]["limits"]["eucentric_height_tolerance"],
    ):
        logging.warning(
            f"""Electron Beam is not close to eucentric height. It should be {settings['calibration']['limits']['eucentric_height_eb']}m
            (Currently is {microscope.beams.electron_beam.working_distance.value:.4f}m)"""
        )

    logging.info(
        f"E OPTICAL MODE: {str(microscope.beams.electron_beam.optical_mode.value)}"
    )
    logging.info(
        f"E OPTICAL MODES:  {str(microscope.beams.electron_beam.optical_mode.available_values)}"
    )

    # Validate Ion Beam
    logging.info("Validating Ion Beam")

    if not microscope.beams.ion_beam.is_on:
        logging.warning("Ion Beam is not on, switching on now...")
        microscope.beams.ion_beam.turn_on()
        assert microscope.beams.ion_beam.is_on, "Unable to turn on Ion Beam."
        logging.warning("Ion Beam turned on.")

    microscope.imaging.set_active_view(2)
    if str(microscope.detector.type.value) != "ETD":
        logging.warning(
            f"Ion detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})"
        )
        if "ETD" in microscope.detector.type.available_values:
            microscope.detector.type.value = "ETD"
            logging.warning(
                f"Changed Ion detector type to {str(microscope.detector.type.value)}"
            )

    if str(microscope.detector.mode.value) != "SecondaryElectrons":
        logging.warning(
            f"Ion detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}"
        )
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = "SecondaryElectrons"
            logging.warning(
                f"Changed Ion detector mode to {str(microscope.detector.mode.value)}"
            )

    # working distance
    logging.info(
        f"IB Working Distance: {microscope.beams.ion_beam.working_distance.value:.4f}m"
    )
    if not np.isclose(
        microscope.beams.ion_beam.working_distance.value,
        settings["calibration"]["limits"]["eucentric_height_ib"],
        atol=settings["calibration"]["limits"]["eucentric_height_tolerance"],
    ):
        logging.warning(
            f"Ion Beam is not close to eucentric height. It should be {settings['calibration']['limits']['eucentric_height_ib']}m \
        (Currently is {microscope.beams.ion_beam.working_distance.value:.4f}m)"
        )

    # validate high voltage
    high_voltage_limits = str(microscope.beams.ion_beam.high_voltage.limits)
    logging.info(f"Ion Beam High Voltage Limits are: {high_voltage_limits}")

    if microscope.beams.ion_beam.high_voltage.value != high_voltage:
        logging.warning(
            f"Ion Beam High Voltage should be {high_voltage}V (Currently {microscope.beams.ion_beam.high_voltage.value}V)"
        )

        if bool(microscope.beams.ion_beam.high_voltage.is_controllable):
            logging.warning(f"Changing Ion Beam High Voltage to {high_voltage}V.")
            microscope.beams.ion_beam.high_voltage.value = high_voltage
            assert (
                microscope.beams.ion_beam.high_voltage.value == high_voltage
            ), "Unable to change Ion Beam High Voltage"
            logging.warning(f"Ion Beam High Voltage Changed")

    # validate plasma gas
    if plasma_gas not in microscope.beams.ion_beam.source.plasma_gas.available_values:
        logging.warning("{plasma_gas} is not available as a plasma gas.")

    if microscope.beams.ion_beam.source.plasma_gas.value != plasma_gas:
        logging.warning(
            f"Plasma Gas is should be {plasma_gas} (Currently {microscope.beams.ion_beam.source.plasma_gas.value})"
        )

    # reset beam shifts
    acquire.reset_beam_shifts(microscope=microscope)



# TODO: change to SystemSettings, CalibrationSettings
def validate_beams_calibration_v2(microscope: SdbMicroscopeClient, settings: MicroscopeSettings) -> None:
    """Validate Beam Settings"""

    # TODO: check beam blanks?
    # TODO: check electron voltage?

    logging.info("Validating Electron Beam")
    if not microscope.beams.electron_beam.is_on:
        logging.warning("Electron Beam is not on, switching on now...")
        microscope.beams.electron_beam.turn_on()
        assert microscope.beams.electron_beam.is_on, "Unable to turn on Electron Beam."
        logging.warning("Electron Beam turned on.")

    microscope.imaging.set_active_view(1)
    if str(microscope.detector.type.value) != "ETD":
        logging.warning(
            f"Electron detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})"
        )
        if "ETD" in microscope.detector.type.available_values:
            microscope.detector.type.value = "ETD"
            logging.warning(
                f"Changed Electron detector type to {str(microscope.detector.type.value)}"
            )

    if str(microscope.detector.mode.value) != "SecondaryElectrons":
        logging.warning(
            f"Electron detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}"
        )
        if "SecondaryElectrons" in microscope.detector.mode.available_values:
            microscope.detector.mode.value = "SecondaryElectrons"
            logging.warning(
                f"Changed Electron detector mode to {str(microscope.detector.mode.value)}"
            )

    # working distances
    logging.info(
        f"EB Working Distance: {microscope.beams.electron_beam.working_distance.value:.4f}m"
    )
    if not np.isclose(
        microscope.beams.electron_beam.working_distance.value,
        settings.calibration.eucentric_height_eb,
        atol=settings.calibration.eucentric_height_tolerance,
    ):
        logging.warning(
            f"""Electron Beam is not close to eucentric height. It should be {settings.calibration.eucentric_height_eb}m
            (Currently is {microscope.beams.electron_beam.working_distance.value:.4f}m)"""
        )

    logging.info(
        f"Electron Optical Mode: {str(microscope.beams.electron_beam.optical_mode.value)}"
    )
    logging.info(
        f"Electron Optical Modes Available:  {str(microscope.beams.electron_beam.optical_mode.available_values)}"
    )

    # Validate Ion Beam
    logging.info("Validating Ion Beam")


    # TODO: Add modes to config?

    # if not microscope.beams.ion_beam.is_on:
    #     logging.warning("Ion Beam is not on, switching on now...")
    #     microscope.beams.ion_beam.turn_on()
    #     assert microscope.beams.ion_beam.is_on, "Unable to turn on Ion Beam."
    #     logging.warning("Ion Beam turned on.")

    # microscope.imaging.set_active_view(2)
    # if str(microscope.detector.type.value) != "ETD":
    #     logging.warning(
    #         f"Ion detector type is  should be ETD (Currently is {str(microscope.detector.type.value)})"
    #     )
    #     if "ETD" in microscope.detector.type.available_values:
    #         microscope.detector.type.value = "ETD"
    #         logging.warning(
    #             f"Changed Ion detector type to {str(microscope.detector.type.value)}"
    #         )

    # if str(microscope.detector.mode.value) != "SecondaryElectrons":
    #     logging.warning(
    #         f"Ion detector mode is should be SecondaryElectrons (Currently is {str(microscope.detector.mode.value)}"
    #     )
    #     if "SecondaryElectrons" in microscope.detector.mode.available_values:
    #         microscope.detector.mode.value = "SecondaryElectrons"
    #         logging.warning(
    #             f"Changed Ion detector mode to {str(microscope.detector.mode.value)}"
    #         )

    # # working distance
    # logging.info(
    #     f"IB Working Distance: {microscope.beams.ion_beam.working_distance.value:.4f}m"
    # )
    # if not np.isclose(
    #     microscope.beams.ion_beam.working_distance.value,
    #     settings["calibration"]["limits"]["eucentric_height_ib"],
    #     atol=settings["calibration"]["limits"]["eucentric_height_tolerance"],
    # ):
    #     logging.warning(
    #         f"Ion Beam is not close to eucentric height. It should be {settings['calibration']['limits']['eucentric_height_ib']}m \
    #     (Currently is {microscope.beams.ion_beam.working_distance.value:.4f}m)"
    #     )

    # # validate high voltage
    # high_voltage_limits = str(microscope.beams.ion_beam.high_voltage.limits)
    # logging.info(f"Ion Beam High Voltage Limits are: {high_voltage_limits}")

    # if microscope.beams.ion_beam.high_voltage.value != high_voltage:
    #     logging.warning(
    #         f"Ion Beam High Voltage should be {high_voltage}V (Currently {microscope.beams.ion_beam.high_voltage.value}V)"
    #     )

    #     if bool(microscope.beams.ion_beam.high_voltage.is_controllable):
    #         logging.warning(f"Changing Ion Beam High Voltage to {high_voltage}V.")
    #         microscope.beams.ion_beam.high_voltage.value = high_voltage
    #         assert (
    #             microscope.beams.ion_beam.high_voltage.value == high_voltage
    #         ), "Unable to change Ion Beam High Voltage"
    #         logging.warning(f"Ion Beam High Voltage Changed")

    # validate plasma gas
    plasma_gas = settings.system.plasma_gas.capitalize()
    if plasma_gas not in microscope.beams.ion_beam.source.plasma_gas.available_values:
        logging.warning("{plasma_gas} is not available as a plasma gas.")

    if microscope.beams.ion_beam.source.plasma_gas.value != plasma_gas:
        logging.warning(
            f"Plasma Gas is should be {plasma_gas} (Currently {microscope.beams.ion_beam.source.plasma_gas.value})"
        )

    # reset beam shifts
    acquire.reset_beam_shifts(microscope=microscope)







def validate_chamber(microscope: SdbMicroscopeClient) -> None:
    """Validate the state of the chamber"""

    logging.info(
        f"Validating Vacuum Chamber State: {str(microscope.vacuum.chamber_state)}"
    )
    if not str(microscope.vacuum.chamber_state) == "Pumped":
        logging.warning(
            f"Chamber vacuum state should be Pumped (Currently is {str(microscope.vacuum.chamber_state)})"
        )

    logging.info(
        f"Validating Vacuum Chamber Pressure: {microscope.state.chamber_pressure.value:.6f} mbar"
    )
    if microscope.state.chamber_pressure.value >= 1e-4:
        logging.warning(
            f"Chamber pressure is too high, please pump the system (Currently {microscope.state.chamber_pressure.value:.6f} mbar)"
        )

def validate_stage_height_for_needle_insertion(
    microscope: SdbMicroscopeClient, settings: dict) -> bool:

    stage = microscope.specimen.stage
    stage_height_limit = settings["calibration"]["limits"]["stage_height_limit"]

    valid_stage_height = True

    if stage.current_position.z < stage_height_limit:

        # Unable to insert the needle if the stage height is below this limit (3.7e-3)
        logging.warning(f"Calibration error detected: stage position height")
        logging.warning(f"Stage Position: {stage.current_position}")

        valid_stage_height = False

    return valid_stage_height

def validate_stage_height_for_needle_insertion_v2(
    microscope: SdbMicroscopeClient, settings: MicroscopeSettings) -> bool:

    stage = microscope.specimen.stage
    needle_stage_height_limit = settings.calibration.needle_stage_height_limit

    # Unable to insert the needle if the stage height is below this limit (3.7e-3)
    return bool(stage.current_position.z < needle_stage_height_limit)



# TODO: change to calibrationSettings
def validate_focus(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    link: bool = True,
) -> bool:

    # check focus distance is within tolerance
    if link:
        calibration.auto_link_stage(microscope)  # TODO: remove?

    return check_working_distance_is_within_tolerance(
        microscope, settings=settings, beam_type=BeamType.ELECTRON
    )

def check_working_distance_is_within_tolerance(
    microscope: SdbMicroscopeClient, settings: dict, beam_type: BeamType = BeamType.ELECTRON
) -> bool:

    if beam_type is BeamType.ELECTRON:
        working_distance = microscope.beams.electron_beam.working_distance.value
        eucentric_height = settings["calibration"]["limits"]["eucentric_height_eb"]
        eucentric_tolerance = settings["calibration"]["limits"][
            "eucentric_height_tolerance"
        ]

    if beam_type is BeamType.ION:
        working_distance = microscope.beams.electron_beam.working_distance.value
        eucentric_height = settings["calibration"]["limits"]["eucentric_height_ib"]
        eucentric_tolerance = settings["calibration"]["limits"][
            "eucentric_height_tolerance"
        ]

    return np.isclose(working_distance, eucentric_height, atol=eucentric_tolerance)




# validate microscope settings
import numpy as np

def _validate_application_files(microscope: SdbMicroscopeClient, application_files: list[str]) -> None:
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


def _validate_dwell_time(microscope: SdbMicroscopeClient, dwell_times: list[float]) -> None:
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

def _validate_electron_beam_currents(microscope: SdbMicroscopeClient, electron_beam_currents: list[float]) -> None:
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

def _validate_ion_beam_currents(microscope: SdbMicroscopeClient, ion_beam_currents: list[float]) -> None:
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
        if beam_current <= min(available_ion_beam_currents) or beam_current >= max(available_ion_beam_currents):
            raise ValueError(
                "{} not found ".format(beam_current)
                + "in list of available ion beam currents!\n"
                "Please choose one from the list: \n"
                "{}".format(available_ion_beam_currents)
            )


def _validate_horizontal_field_width(microscope: SdbMicroscopeClient, horizontal_field_widths: list[float]) -> None:
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


def _validate_scanning_resolutions(microscope: SdbMicroscopeClient, scanning_resolutions: list[str]) -> None:
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

