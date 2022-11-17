import logging

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    GrabFrameSettings,
    Rectangle,
    RunAutoCbSettings,
)
from skimage import exposure

from fibsem import utils
from fibsem.structures import BeamType, GammaSettings, ImageSettings, ReferenceImages
from fibsem.fibsemImage import fibsemImage



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


def take_reference_images(
    microscope: SdbMicroscopeClient, image_settings: ImageSettings
) -> list[AdornedImage]:
    """Take a reference image using both beams

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        image_settings (ImageSettings): imaging settings

    Returns:
        list[AdornedImage]: electron and ion reference image pair
    """
    tmp_beam_type = image_settings.beam_type
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = new_image(microscope, image_settings)
    image_settings.beam_type = BeamType.ION
    ib_image = new_image(microscope, image_settings)
    image_settings.beam_type = tmp_beam_type  # reset to original beam type
    return eb_image, ib_image


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


def auto_gamma(image: AdornedImage, settings: GammaSettings) -> AdornedImage:
    """Automatic gamma correction"""
    std = np.std(image.data) # unused variable?
    mean = np.mean(image.data)
    diff = mean - 255 / 2.0
    gam = np.clip(
        settings.min_gamma, 1 + diff * settings.scale_factor, settings.max_gamma
    )
    if abs(diff) < settings.threshold:
        gam = 1.0
    logging.debug(
        f"AUTO_GAMMA | {image.metadata.acquisition.beam_type} | {diff:.3f} | {gam:.3f}"
    )
    image_data = exposure.adjust_gamma(image.data, gam)
    
    return AdornedImage(data=image_data, metadata=image.metadata)

def new_image(
    microscope: SdbMicroscopeClient,
    settings: ImageSettings,
    reduced_area: Rectangle = None,
) -> AdornedImage:
    """Apply the image settings and take a new image

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client connection
        settings (ImageSettings): image settings to take the image with
        reduced_area (Rectangle, optional): image with the reduced area . Defaults to None.

    Returns:
            AdornedImage: new image
    """

    # set frame settings
    frame_settings = GrabFrameSettings(
        resolution=settings.resolution,
        dwell_time=settings.dwell_time,
        reduced_area=reduced_area,
    )

    # set horizontal field width
    if settings.beam_type is BeamType.ELECTRON:
        hfw_limits = microscope.beams.electron_beam.horizontal_field_width.limits
        settings.hfw = np.clip(settings.hfw, hfw_limits.min, hfw_limits.max)
        microscope.beams.electron_beam.horizontal_field_width.value = settings.hfw
        label = f"{settings.label}_eb"

    if settings.beam_type is BeamType.ION:
        hfw_limits = microscope.beams.ion_beam.horizontal_field_width.limits
        settings.hfw = np.clip(settings.hfw, hfw_limits.min, hfw_limits.max)
        microscope.beams.ion_beam.horizontal_field_width.value = settings.hfw
        label = f"{settings.label}_ib"

    # run autocontrast
    if settings.autocontrast:
        autocontrast(microscope, beam_type=settings.beam_type)

    # acquire the image
    image = acquire_image(
        microscope=microscope,
        settings=frame_settings,
        beam_type=settings.beam_type,
    )

    # apply gamma correction
    if settings.gamma.enabled:
        image = auto_gamma(image, settings.gamma)

    # save image
    if settings.save:
        utils.save_image(image=image, save_path=settings.save_path, label=label)
    
    return image

def new_fibsemImage(
    microscope, settings: ImageSettings, reduced_area = None
) -> fibsemImage:
    if type(microscope) == SdbMicroscopeClient:
        image = new_image(microscope, settings, reduced_area)
        fibsem_image = fibsemImage()
        fibsem_image.convert_adorned_to_fibsemImage(image, settings)
    else:
        fibsem_image = None
    return fibsem_image

def last_fibsemImage(
    microscope, beam_type
) -> fibsemImage:
    if type(microscope) == SdbMicroscopeClient:
        image = last_image(microscope, beam_type)
        fibsem_image = fibsemImage()
        fibsem_image.convert_adorned_to_fibsemImage(image) # Cannot access metadata
    else:
        fibsem_image = None
    return fibsem_image

def last_image(
    microscope: SdbMicroscopeClient, beam_type: BeamType =BeamType.ELECTRON
) -> AdornedImage:
    """Get the last previously acquired image.

    Args:
        microscope (SdbMicroscopeClient):  autoscript microscope instance
        beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.

    Returns:
        AdornedImage: last image
    """


    microscope.imaging.set_active_view(beam_type.value)
    microscope.imaging.set_active_device(beam_type.value)
    image = microscope.imaging.get_image()
    return image


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
