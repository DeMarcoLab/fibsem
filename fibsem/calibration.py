import logging
from datetime import datetime

import numpy as np

try:
    from autoscript_sdb_microscope_client.enumerations import (
        CoordinateSystem, ManipulatorCoordinateSystem)
    from autoscript_sdb_microscope_client.structures import (
        Rectangle, RunAutoFocusSettings, StagePosition)

    THERMO = True
except:
    THERMO = False
try:
    import tescanautomation

    TESCAN = True
except:
    TESCAN = False

from pathlib import Path

import skimage

from fibsem import acquire
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamSettings, BeamSystemSettings, BeamType,
                               FibsemRectangle, FibsemStagePosition,
                               ImageSettings, MicroscopeSettings,
                               MicroscopeState)
def auto_focus_beam(
    microscope: FibsemMicroscope,
    beam_type: BeamType,
    mode: str = "default",
    focus_image_settings: ImageSettings = None,
    step_size: float = 0.05e-3,
    num_steps: int = 5,
) -> None:

    if mode == "default":
        microscope.auto_focus(beam_type=beam_type)

    if mode == "sharpness":
        
        from skimage.filters.rank import gradient
        from skimage.morphology import disk

        if focus_image_settings is None:
            focus_image_settings = ImageSettings(
                resolution=[768, 512],
                dwell_time=200e-9,
                hfw=50e-6,
                beam_type=beam_type,
                save=False,
                autocontrast=True,
                gamma_enabled=False,
                label=None,
                reduced_area=FibsemRectangle(0.3, 0.3, 0.4, 0.4),
            )

        # get current working distance
        current_wd = microscope.get("working_distance", beam_type)

        logging.info(f"sharpness (accutance) based auto-focus routine")
        logging.info(f"initial working distance: {current_wd:.2e}")

        # define working distance range
        min_wd = current_wd - (num_steps * step_size / 2)
        max_wd = current_wd + (num_steps * step_size / 2)
        wds = np.linspace(min_wd, max_wd, num_steps + 1)

        # loop through working distances and calculate the sharpness (acutance)
        # highest acutance is best focus
        sharpeness_metric = []
        for i, wd in enumerate(wds):

            logging.info(f"image {i}: {wd:.2e}")
            microscope.set("working_distance", wd, beam_type)

            img = acquire.new_image(microscope, focus_image_settings)

            # sharpness (Acutance: https://en.wikipedia.org/wiki/Acutance)
            sharpness = np.mean(gradient(skimage.filters.median(np.copy(img.data)), disk(5)))
            sharpeness_metric.append(sharpness)

        # select working distance with max acutance
        idx = np.argmax(sharpeness_metric)

        pairs = list(zip(wds, sharpeness_metric))
        logging.info([f"{wd:.2e}: {metric:.4f}" for wd, metric in pairs])
        logging.info(f"{idx}, {wds[idx]:.2e}, {sharpeness_metric[idx]:.4f}")

        # set working distance
        microscope.set(
            key="working_distance",
            value=wds[idx],
            beam_type=beam_type,
        )

    if mode == "dog":
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
            resolution=(768, 512),
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


# def auto_needle_calibration(
#     microscope: FibsemMicroscope, settings: MicroscopeSettings, validate: bool = True
# ):

#     if TESCAN:
#         raise NotImplementedError

#     # set coordinate system
#     microscope.connection.specimen.manipulator.set_default_coordinate_system(
#         ManipulatorCoordinateSystem.STAGE
#     )

#     # current working distance
#     wd = microscope.connection.beams.electron_beam.working_distance.value
#     needle_wd_eb = 4.0e-3

#     # focus on the needle
#     microscope.connection.beams.electron_beam.working_distance.value = needle_wd_eb
#     microscope.connection.specimen.stage.link()

#     settings.image.hfw = 2700e-6
#     acquire.take_reference_images(microscope, settings.image)

#     # very low res alignment
#     hfws = [2700e-6, 900e-6, 400e-6, 150e-6]
#     for hfw in hfws:
#         settings.image.hfw = hfw
#         align_needle_to_eucentric_position(microscope, settings, validate=validate)

#     # restore working distance
#     microscope.connection.beams.electron_beam.working_distance.value = wd
#     microscope.connection.specimen.stage.link()

#     logging.info(f"Finished automatic needle calibration.")


# def align_needle_to_eucentric_position(
#     microscope: FibsemMicroscope,
#     settings: MicroscopeSettings,
#     validate: bool = False,
# ) -> None:
#     """Move the needle to the eucentric position, and save the updated position to disk

#     Args:
#         microscope (FibsemMicroscope): OpenFIBSEM microscope instance
#         settings (MicroscopeSettings): microscope settings
#         validate (bool, optional): validate the alignment. Defaults to False.
#     """

#     if TESCAN:
#         raise NotImplementedError

#     from fibsem.ui import windows as fibsem_ui_windows
#     from fibsem.detection.utils import FeatureType, Feature
#     from fibsem.detection import detection

#     # take reference images
#     settings.image.save = False
#     settings.image.beam_type = BeamType.ELECTRON

#     det = fibsem_ui_windows.detect_features_v2(
#         microscope=microscope.connection,
#         settings=settings,
#         features=[
#             Feature(FeatureType.NeedleTip, None),
#             Feature(FeatureType.ImageCentre, None),
#         ],
#         validate=validate,
#     )
#     detection.move_based_on_detection(
#         microscope.connection, settings, det, beam_type=settings.image.beam_type
#     )

#     # take reference images
#     settings.image.save = False
#     settings.image.beam_type = BeamType.ION

#     image = acquire.new_image(microscope, settings.image)

#     det = fibsem_ui_windows.detect_features_v2(
#         microscope=microscope.connection,
#         settings=settings,
#         features=[
#             Feature(FeatureType.NeedleTip, None),
#             Feature(FeatureType.ImageCentre, None),
#         ],
#         validate=validate,
#     )
#     detection.move_based_on_detection(
#         microscope.connection, settings, det, beam_type=settings.image.beam_type, move_x=False
#     )

#     # take image
#     acquire.take_reference_images(microscope, settings.image)


def auto_home_and_link_v2(
    microscope: FibsemMicroscope, state: MicroscopeState = None
) -> None:

    # home the stage and return the linked state
    if state is None:
        state = microscope.get_current_microscope_state()

    # home the stage
    microscope.home()

    # move to saved linked state
    microscope.set_microscope_state(state)

