import logging

import numpy as np
import os

from skimage import exposure

from fibsem.structures import (
    BeamType,
    GammaSettings,
    ImageSettings,
    ReferenceImages,
    FibsemImage,
    FibsemRectangle,
)
from fibsem.FibsemMicroscope import FibsemMicroscope



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
    microscope: FibsemMicroscope,
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
    reduced_area: FibsemRectangle = None,
) -> FibsemImage:
    """Apply the image settings and take a new image

    Args:
        microscope (FibsemMicroscope): fibsem microscope client connection
        settings (ImageSettings): image settings to take the image with
        reduced_area (FibsemRectangle, optional): image with the reduced area . Defaults to None.

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


