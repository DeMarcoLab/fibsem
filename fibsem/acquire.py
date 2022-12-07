import logging

import numpy as np
import os

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    GrabFrameSettings,
    Rectangle,
    RunAutoCbSettings,
)
from skimage import exposure

from fibsem import utils
from fibsem.structures import (
    BeamType,
    GammaSettings,
    ImageSettings,
    ReferenceImages,
    FibsemImage,
)
from fibsem.FibsemMicroscope import FibsemMicroscope

from fibsem import calibration



"""==================== Deprecated Functions ===================="""


def autocontrast(microscope: SdbMicroscopeClient, beam_type=BeamType.ELECTRON) -> None:
    """Automatically adjust the microscope image contrast."""
    microscope.imaging.set_active_view(beam_type.value)

    cb_settings = RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    logging.debug("automatically adjusting contrast...")
    microscope.auto_functions.run_auto_cb()  # cb_settings, TODO: pass through settings


def last_image(
    microscope: SdbMicroscopeClient, beam_type: BeamType = BeamType.ELECTRON
) -> AdornedImage:
    """Get the last previously acquired image.

    Args:
        microscope (SdbMicroscopeClient):  autoscript microscope instance
        beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.

    Returns:
        AdornedImage: last image
    """

    microscope = microscope.connection

    microscope.imaging.set_active_view(beam_type.value)
    microscope.imaging.set_active_device(beam_type.value)
    image = microscope.imaging.get_image()
    state = calibration.get_current_microscope_state(microscope)
    image_settings = ImageSettings(
        resolution=f"{image.width}x{image.height}",
        dwell_time=image.metadata.scan_settings.dwell_time,
        hfw=image.width * image.metadata.binary_result.pixel_size.x,
        autocontrast=True,
        beam_type=BeamType.ELECTRON,
        gamma=GammaSettings(),
        save=False,
        save_path="path",
        label=utils.current_timestamp(),
        reduced_area=None,
    )
    fibsem_img = FibsemImage.fromAdornedImage(image, image_settings, state)
    return fibsem_img


def acquire_image(
    microscope: SdbMicroscopeClient,
    settings: GrabFrameSettings = None,
    beam_type: BeamType = BeamType.ELECTRON,
) -> AdornedImage:
    """Acquire a new image.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        settings (GrabFrameSettings, optional): frame grab settings. Defaults to None.
        beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.

    Returns:
        AdornedImage: new image
    """
    logging.info(f"acquiring new {beam_type.name} image.")
    microscope.imaging.set_active_view(beam_type.value)
    microscope.imaging.set_active_device(beam_type.value)
    image = microscope.imaging.grab_frame(settings)

    return image

def reset_beam_shifts(microscope: SdbMicroscopeClient):
    """Set the beam shift to zero for the electron and ion beams

    Args:
        microscope (SdbMicroscopeClient): Autoscript microscope object
    """
    from autoscript_sdb_microscope_client.structures import Point

    # reset zero beamshift
    logging.debug(
        f"reseting ebeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value}"
    )
    microscope.beams.electron_beam.beam_shift.value = Point(0, 0)
    logging.debug(
        f"reseting ibeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value}"
    )
    microscope.beams.ion_beam.beam_shift.value = Point(0, 0)
    logging.debug(f"reset beam shifts to zero complete")

"""====================================================== """


def take_reference_images(
    microscope: FibsemMicroscope, image_settings: ImageSettings
) -> list[FibsemImage]:
    """Take a reference image using both beams

    Args:
        microscope (FibsemMicroscope): fibsem microscope instance
        image_settings (ImageSettings): imaging settings

    Returns:
        list[AdornedImage]: electron and ion reference image pair
    """
    tmp_beam_type = image_settings.beam_type
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = new_image(microscope, image_settings)
    # state_eb = calibration.get_current_microscope_state(microscope)
    image_settings.beam_type = BeamType.ION
    ib_image = new_image(microscope, image_settings)
    # state_ib = calibration.get_current_microscope_state(microscope)
    image_settings.beam_type = tmp_beam_type  # reset to original beam type

    return eb_image, ib_image
    # return FibsemImage.fromAdornedImage(eb_image, image_settings, state_eb), FibsemImage.fromAdornedImage(ib_image, image_settings, state_ib)


def take_set_of_reference_images(
    microscope: SdbMicroscopeClient,
    image_settings: ImageSettings,
    hfws: tuple[float],
    label: str = "ref_image",
) -> ReferenceImages:
    """Take a set of reference images at low and high magnification"""

    # force save
    image_settings.save = True

    image_settings.hfw = hfws[0]
    image_settings.label = f"{label}_low_res"
    low_eb, low_ib = take_reference_images(microscope, image_settings)

    image_settings.hfw = hfws[1]
    image_settings.label = f"{label}_high_res"
    high_eb, high_ib = take_reference_images(microscope, image_settings)

    reference_images = ReferenceImages(low_eb, high_eb, low_ib, high_ib)

    return reference_images


def auto_gamma(image: FibsemImage, settings: GammaSettings) -> FibsemImage:
    """Automatic gamma correction"""
    std = np.std(image.data)  # unused variable?
    mean = np.mean(image.data)
    diff = mean - 255 / 2.0
    gam = np.clip(
        settings.min_gamma, 1 + diff * settings.scale_factor, settings.max_gamma
    )
    if abs(diff) < settings.threshold:
        gam = 1.0
    logging.debug(
        f"AUTO_GAMMA | {image.metadata.image_settings.beam_type} | {diff:.3f} | {gam:.3f}"
    )
    image_data = exposure.adjust_gamma(image.data, gam)

    return FibsemImage(data=image_data, metadata=image.metadata)


def new_image(
    microscope: FibsemMicroscope,
    settings: ImageSettings,
    reduced_area: Rectangle = None,
) -> FibsemImage:
    """Apply the image settings and take a new image

    Args:
        microscope (FibsemMicroscope): fibsem microscope client connection
        settings (ImageSettings): image settings to take the image with
        reduced_area (Rectangle, optional): image with the reduced area . Defaults to None.

    Returns:
            FibsemImage: new image
            String: filename of saved FibsemImage
    """
   

    # set label
    if settings.beam_type is BeamType.ELECTRON:
        label = f"{settings.label}_eb"

    if settings.beam_type is BeamType.ION:
        label = f"{settings.label}_ib"

    # run autocontrast
    if settings.autocontrast:
        microscope.autocontrast(beam_type=settings.beam_type)

    # acquire the image
    image = microscope.acquire_image(
        image_settings=settings,
    )

    # apply gamma correction
    if settings.gamma.enabled:
        image = auto_gamma(image, settings.gamma)

    # save image
    if settings.save:
        filename = os.path.join(settings.save_path, label)
        image.save(save_path=filename)

    return image


