import logging
from datetime import datetime

import numpy as np

from fibsem.config import load_microscope_manufacturer
manufacturer = load_microscope_manufacturer()

if manufacturer == "Thermo":
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.enumerations import (
        CoordinateSystem,
        ManipulatorCoordinateSystem,
    )
    from autoscript_sdb_microscope_client.structures import (
        StagePosition,
        Rectangle,
        RunAutoFocusSettings,
    )

from fibsem import acquire, movement
from fibsem.structures import (
    BeamSettings,
    MicroscopeState,
    BeamType,
    ImageSettings,
    MicroscopeSettings,
    BeamSystemSettings,
)
from fibsem.microscope import FibsemMicroscope

from pathlib import Path
import skimage
from skimage.morphology import disk
from skimage.filters.rank import gradient


def auto_link_stage(microscope: FibsemMicroscope, hfw: float = 150e-6) -> None:
    """Automatically focus and link sample stage z-height.

    Notes:
        - Focusing determines the working distance (focal distance) of the beam
        - Relinking is required whenever there is a significant change in vertical distance, i.e. moving
          from the landing grid to the sample grid.
        - Linking determines the specimen coordinate system, as it is defined as the relative dimensions of the top of stage
          to the instruments.
    """
    if manufacturer == "Tescan":
        raise NotImplementedError

    microscope.connection.imaging.set_active_view(BeamType.ELECTRON.value)
    original_hfw = microscope.connection.beams.electron_beam.horizontal_field_width.value
    microscope.connection.beams.electron_beam.horizontal_field_width.value = hfw
    microscope.autocontrast(beam_type=BeamType.ELECTRON)
    microscope.connection.auto_functions.run_auto_focus()
    microscope.connection.specimen.stage.link()
    # NOTE: replace with auto_focus_and_link if performance of focus is poor
    # # Restore original settings
    microscope.connection.beams.electron_beam.horizontal_field_width.value = original_hfw


def auto_focus_beam(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings, # NOTE: This isn't used and is a mandatory argument
    mode: str = "default",
    wd_delta: float = 0.05e-3,
    steps: int = 5,
    reduced_area: Rectangle = Rectangle(0.3, 0.3, 0.4, 0.4),
    focus_image_settings: ImageSettings = None,
) -> None:

    if manufacturer == "Tescan":
        raise NotImplementedError


    if mode == "default":
        microscope.connection.imaging.set_active_device(BeamType.ELECTRON.value)
        microscope.connection.imaging.set_active_view(BeamType.ELECTRON.value)  # set to Ebeam

        focus_settings = RunAutoFocusSettings()
        microscope.connection.auto_functions.run_auto_focus()

    if mode == "sharpness":

        if focus_image_settings is None:
            focus_image_settings = ImageSettings(
                resolution=(768, 512), 
                dwell_time=200e-9,
                hfw=50e-6,
                beam_type=BeamType.ELECTRON,
                save=False,
                autocontrast=True,
                gamma_enabled=False,
                label=None,
            )

        current_wd = microscope.connection.beams.electron_beam.working_distance.value
        logging.info(f"sharpness (accutance) based auto-focus routine")

        logging.info(f"initial working distance: {current_wd:.2e}")

        min_wd = current_wd - (steps * wd_delta / 2)
        max_wd = current_wd + (steps * wd_delta / 2)

        working_distances = np.linspace(min_wd, max_wd, steps + 1)

        # loop through working distances and calculate the sharpness (acutance)
        # highest acutance is best focus
        sharpeness_metric = []
        for i, wd in enumerate(working_distances):

            logging.info(f"Img {i}: {wd:.2e}")
            microscope.connection.beams.electron_beam.working_distance.value = wd

            img = acquire.new_image(
                microscope, focus_image_settings, reduced_area=reduced_area
            )

            # sharpness (Acutance: https://en.wikipedia.org/wiki/Acutance
            out = gradient(skimage.filters.median(np.copy(img.data)), disk(5))

            sharpness = np.mean(out)
            sharpeness_metric.append(sharpness)

        # select working distance with max acutance
        idx = np.argmax(sharpeness_metric)

        pairs = list(zip(working_distances, sharpeness_metric))
        logging.info([f"{wd:.2e}: {metric:.4f}" for wd, metric in pairs])
        logging.info(
            f"{idx}, {working_distances[idx]:.2e}, {sharpeness_metric[idx]:.4f}"
        )

        # reset working distance
        microscope.connection.beams.electron_beam.working_distance.value = working_distances[idx]

        # NOTE: Why is this commented out?
        # run fine auto focus and link
        # microscope.imaging.set_active_device(BeamType.ELECTRON.value)
        # microscope.imaging.set_active_view(BeamType.ELECTRON.value)  # set to Ebeam
        # microscope.auto_functions.run_auto_focus()
        # microscope.specimen.stage.link()

    if mode == "dog": # NOTE: Why
        # TODO: implement difference of gaussian based auto-focus

        pass

    return


def auto_charge_neutralisation(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    discharge_settings: ImageSettings = None,
    n_iterations: int = 10,
) -> None:

    # take sequence of images quickly,

    # use preset settings if not defined
    if discharge_settings is None:
        discharge_settings = ImageSettings(
            resolution=(768,512),
            dwell_time=200e-9,
            hfw=image_settings.hfw,
            beam_type=BeamType.ELECTRON,
            save=False,
            autocontrast=False,
            gamma_enabled=False,
            label=None,
        )

    for i in range(n_iterations):
        acquire.new_image(microscope, discharge_settings)

    # take image
    acquire.new_image(microscope, image_settings)

    logging.info(f"BAM! and the charge is gone!")  # important information


def auto_needle_calibration(
    microscope: FibsemMicroscope, settings: MicroscopeSettings, validate: bool = True
):

    if manufacturer == "Tescan":
        raise NotImplementedError

    # set coordinate system
    microscope.connection.specimen.manipulator.set_default_coordinate_system(
        ManipulatorCoordinateSystem.STAGE
    )

    # current working distance
    wd = microscope.connection.beams.electron_beam.working_distance.value
    needle_wd_eb = 4.0e-3

    # focus on the needle
    microscope.connection.beams.electron_beam.working_distance.value = needle_wd_eb
    microscope.connection.specimen.stage.link()

    settings.image.hfw = 2700e-6
    acquire.take_reference_images(microscope, settings.image)

    # very low res alignment
    hfws = [2700e-6, 900e-6, 400e-6, 150e-6]
    for hfw in hfws:
        settings.image.hfw = hfw
        align_needle_to_eucentric_position(microscope, settings, validate=validate)

    # restore working distance
    microscope.connection.beams.electron_beam.working_distance.value = wd
    microscope.connection.specimen.stage.link()

    logging.info(f"Finished automatic needle calibration.")


def align_needle_to_eucentric_position(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    validate: bool = False,
) -> None:
    """Move the needle to the eucentric position, and save the updated position to disk

    Args:
        microscope (FibsemMicroscope): OpenFIBSEM microscope instance
        settings (MicroscopeSettings): microscope settings
        validate (bool, optional): validate the alignment. Defaults to False.
    """

    if manufacturer == "Tescan":
        raise NotImplementedError

    from fibsem.ui import windows as fibsem_ui_windows
    from fibsem.detection.utils import FeatureType, Feature
    from fibsem.detection import detection

    # take reference images
    settings.image.save = False
    settings.image.beam_type = BeamType.ELECTRON

    det = fibsem_ui_windows.detect_features_v2(
        microscope=microscope.connection,
        settings=settings,
        features=[
            Feature(FeatureType.NeedleTip, None),
            Feature(FeatureType.ImageCentre, None),
        ],
        validate=validate,
    )
    detection.move_based_on_detection(
        microscope.connection, settings, det, beam_type=settings.image.beam_type
    )

    # take reference images
    settings.image.save = False
    settings.image.beam_type = BeamType.ION

    image = acquire.new_image(microscope, settings.image)

    det = fibsem_ui_windows.detect_features_v2(
        microscope=microscope.connection,
        settings=settings,
        features=[
            Feature(FeatureType.NeedleTip, None),
            Feature(FeatureType.ImageCentre, None),
        ],
        validate=validate,
    )
    detection.move_based_on_detection(
        microscope.connection, settings, det, beam_type=settings.image.beam_type, move_x=False
    )

    # take image
    acquire.take_reference_images(microscope, settings.image)


def auto_home_and_link(
    microscope: FibsemMicroscope, state: MicroscopeState = None
) -> None:

    if manufacturer == "Tescan":
        raise NotImplementedError

    import os
    from fibsem import utils, config

    # home the stage
    logging.info(f"Homing stage...")
    microscope.connection.specimen.stage.home()

    # if no state provided, use the default
    if state is None:
        path = os.path.join(config.CONFIG_PATH, "calibrated_state.yaml")
        state = MicroscopeState.__from_dict__(utils.load_yaml(path))

    # move to saved linked state
    microscope.set_microscope_state(state)

    # link
    logging.info("Linking stage...")
    microscope.autocontrast(beam_type=BeamType.ELECTRON)
    microscope.connection.auto_functions.run_auto_focus()
    microscope.connection.specimen.stage.link()


def auto_home_and_link_v2(
    microscope: FibsemMicroscope, state: MicroscopeState = None
) -> None:

    if manufacturer == "Tescan":
        raise NotImplementedError
    # home the stage and return the linked state

    if state is None:
        state = microscope.get_current_microscope_state()

    # home the stage
    logging.info(f"Homing stage...")
    microscope.connection.specimen.stage.home()

    # move to saved linked state
    microscope.set_microscope_state(state)

    # relink (set state also links...)
    microscope.connection.specimen.stage.link()


# STATE MANAGEMENT


def get_raw_stage_position(microscope: FibsemMicroscope) -> StagePosition:
    """Get the current stage position in raw coordinate system, and switch back to specimen"""

    if manufacturer == "Tescan":
        raise NotImplementedError

    microscope.connection.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)
    stage_position = microscope.connection.specimen.stage.current_position
    microscope.connection.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    return stage_position

def get_current_beam_system_state(microscope: FibsemMicroscope, beam_type: BeamType):

    if manufacturer == "Tescan":
        raise NotImplementedError

    if beam_type is BeamType.ELECTRON:
        microscope_beam = microscope.connection.beams.electron_beam
    if beam_type is BeamType.ION:
        microscope_beam = microscope.connection.beams.ion_beam

    # set beam active view and device
    microscope.connection.imaging.set_active_view(beam_type.value)
    microscope.connection.imaging.set_active_device(beam_type.value)

    # get current beam settings
    voltage = microscope_beam.high_voltage.value
    current = microscope_beam.beam_current.value
    detector_type = microscope.connection.detector.type.value
    detector_mode = microscope.connection.detector.mode.value

    if beam_type is BeamType.ION:
        eucentric_height = 16.5e-3
        plasma_gas = microscope_beam.source.plasma_gas.value
    else:
        eucentric_height = 4.0e-3
        plasma_gas = None

    return BeamSystemSettings(
        beam_type=beam_type,
        voltage=voltage,
        current=current,
        detector_type=detector_type,
        detector_mode=detector_mode,
        eucentric_height=eucentric_height,
        plasma_gas=plasma_gas,
    )
