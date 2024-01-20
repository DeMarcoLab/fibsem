import logging
from datetime import datetime
import numpy as np
from pathlib import Path

import skimage

from fibsem import acquire, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamSettings, BeamSystemSettings, BeamType,
                               FibsemRectangle, FibsemStagePosition,
                               ImageSettings, MicroscopeSettings, FibsemImage,
                               MicroscopeState)
from fibsem.detection.detection import NeedleTip, ImageCentre
from fibsem import config as cfg
def auto_focus_beam(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    beam_type: BeamType,
    metric_fn = None, # function to calculate focus metric
    focus_image_settings: ImageSettings = None,
    step_size: float = 0.05e-3,
    num_steps: int = 5,
    kwargs: dict = {},
    verbose: bool = False,
) -> None:

    # TODO: @patrickcleeve2 this could be generalised further if we specify the parameter to sweep through too...
    # e.g. microscope.set("working_distance", value, beam_type) for auto focus
    # e.g. microscope.set("beam_current", value, beam_type) for auto stigmation
    # e.g. microscope.set("beam_current", value, beam_type) for auto beam current
    # might be too much generalisation though...
    # also need to specify a default function

    if metric_fn is None:
        # run the default autofocus routine
        microscope.auto_focus(beam_type=beam_type)
        return
        
    if focus_image_settings is None:
        # use preset settings if not defined
        focus_image_settings = ImageSettings(
            resolution=[768, 512],
            dwell_time=200e-9,
            hfw=100e-6,
            beam_type=beam_type,
            save=True,
            path=settings.image.path,
            autocontrast=True,
            autogamma=False,
            filename=f"{utils.current_timestamp()}_",
            reduced_area=FibsemRectangle(0.3, 0.3, 0.4, 0.4),
        )

    # get current working distance
    current_wd = microscope.get("working_distance", beam_type)

    if verbose:
        logging.info(f"{metric_fn.__name__} based auto-focus routine")
        logging.info(f"doc: {metric_fn.__doc__}")
        logging.info(f"initial working distance: {current_wd:.2e}")

    # define working distance range
    min_wd = current_wd - (num_steps * step_size / 2)
    max_wd = current_wd + (num_steps * step_size / 2)
    wds = np.linspace(min_wd, max_wd, num_steps + 1)

    # loop through working distances and calculate the sharpness (acutance)
    # highest acutance is best focus
    metrics = []
    for i, wd in enumerate(wds):

        logging.info(f"image {i}: {wd:.2e}")
        microscope.set("working_distance", wd, beam_type)

        focus_image_settings.filename = f"{utils.current_timestamp()}_sharpness_{i}"
        img = acquire.new_image(microscope, focus_image_settings)

        # calculate focus metric 
        metric = metric_fn(img, **kwargs)
        metrics.append(metric)

    # select working distance with highest metric
    idx = np.argmax(metrics)

    if verbose:
        pairs = list(zip(wds, metrics))
        logging.info([f"{wd:.2e}: {metric:.4f}" for wd, metric in pairs])
        logging.info(f"{idx}, {wds[idx]:.2e}, {metrics[idx]:.4f}")

    # set working distance
    microscope.set(
        key="working_distance",
        value=wds[idx],
        beam_type=beam_type,
    )


    return

def _sharpness(img:FibsemImage, **kwargs) -> float:
    """Calculate sharpness (accutance) of an image.
    (Acutance: https://en.wikipedia.org/wiki/Acutance)

    Args:
        img (FibsemImage): _description_
        disk_size (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    from skimage.filters.rank import gradient
    from skimage.morphology import disk
    disk_size = kwargs.get("disk_size", 5)
    logging.info(f"calculating sharpness (accutance) of image {img}: {disk_size}")
    return np.mean(gradient(skimage.filters.median(np.copy(img.data)), disk(disk_size)))

def _dog(img: FibsemImage, **kwargs) -> float:
    """Calculate difference of gaussian (DoG) of an image.

    Args:
        img (FibsemImage): _description_

    Returns:
        _type_: _description_
    """
    low = kwargs.get("low", 3)
    high = kwargs.get("high", 9)
    from skimage.filters import difference_of_gaussians
    return np.mean(difference_of_gaussians(np.copy(img.data), low, high))

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
            resolution=[768, 512],
            dwell_time=200e-9,
            hfw=image_settings.hfw,
            beam_type=BeamType.ELECTRON,
            save=False,
            autocontrast=False,
            autogamma=False,
            filename=None,
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

#     from fibsem.ui import windows as fibsem_ui_windows
#     from fibsem.detection import detection

#     # take reference images
#     settings.image.save = False
#     settings.image.beam_type = BeamType.ELECTRON

#     det = fibsem_ui_windows.detect_features_v2(
#         microscope=microscope,
#         settings=settings,
#         features=[
#             NeedleTip(),
#             ImageCentre(),
#         ],
#         validate=validate,
#     )
#     detection.move_based_on_detection(
#         microscope, settings, det, beam_type=settings.image.beam_type
#     )

#     # take reference images
#     settings.image.save = False
#     settings.image.beam_type = BeamType.ION

#     image = acquire.new_image(microscope, settings.image)

#     det = fibsem_ui_windows.detect_features_v2(
#         microscope=microscope,
#         settings=settings,
#         features=[
#             NeedleTip(),
#             ImageCentre(),
#         ],
#         validate=validate,
#     )
#     detection.move_based_on_detection(
#         microscope, settings, det, beam_type=settings.image.beam_type, move_x=False
#     )

#     # take image
#     acquire.take_reference_images(microscope, settings.image)


def auto_home_and_link_v2(
    microscope: FibsemMicroscope, state: MicroscopeState = None
) -> None:

    # home the stage and return the linked state
    if state is None:
        state = microscope.get_microscope_state()

    # home the stage
    microscope.home_stage()

    # move to saved linked state
    microscope.set_microscope_state(state)


def _calibrate_manipulator_thermo(microscope:FibsemMicroscope, settings:MicroscopeSettings, parent_ui = None):
    from fibsem.detection import detection
    from fibsem.segmentation.model import load_model
    import matplotlib.pyplot as plt

    from autolamella.workflows.ui import update_detection_ui, ask_user

    if parent_ui:
        ret = ask_user(parent_ui, 
            msg="Please complete the EasyLift alignment procedure in the xT UI until Step 5. Press Continue to proceed.",
            pos="Continue", neg="Cancel")
        if ret is False:
            return
    else:
        input("Please complete the EasyLift alignment procedure in the xT UI until Step 5. Press Enter to proceed.")


    def align_manipulator_to_eucentric(microsscope: FibsemMicroscope, settings:MicroscopeSettings, parent_ui, validate: bool) -> None:
        return NotImplemented

    settings.protocol["options"].get("checkpoint", cfg.__DEFAULT_CHECKPOINT__)
    model = load_model(settings.protocol["options"]["checkpoint"])
    settings.image.autocontrast = True

    hfws = [2000e-6, 900e-6, 400e-6, 150e-6]

    # set working distance
    wd = microscope.get("working_distance", BeamType.ELECTRON)
    microscope.set("working_distance", microscope.system.electron.eucentric_height, BeamType.ELECTRON)

    for hfw in hfws:
        for beam_type in [BeamType.ELECTRON, BeamType.ION]:
            settings.image.hfw = hfw
            settings.image.beam_type = beam_type

            features = [detection.NeedleTip(), detection.ImageCentre()] if np.isclose(microscope.get("scan_rotation", beam_type), 0) else [detection.NeedleTipBottom(), detection.ImageCentre()]
            
            if parent_ui:
                det = update_detection_ui(microscope, settings, features, parent_ui, validate = True, msg = f"Confirm Feature Detection. Press Continue to proceed.")
            else:
                image = acquire.new_image(microscope, settings.image)
                det = detection.detect_features(image, model, features=features, pixelsize=image.metadata.pixel_size.x)
                detection.plot_detection(det)
                ret  = input("continue? (y/n)")
                
                if ret != "y":
                    return

            move_x = bool(beam_type == BeamType.ELECTRON) # ION calibration only in z
            detection.move_based_on_detection(microscope, settings, det, beam_type, move_x=move_x, _move_system="manipulator")

    # restore working distance
    microscope.set("working_distance", wd, BeamType.ELECTRON)

    if parent_ui:
        ask_user(parent_ui, 
            msg="Alignment of EasyLift complete. Please complete the procedure in xT UI. Press Continue to proceed.",
            pos="Continue")
    print(f"The manipulator should now be centred in both beams.")