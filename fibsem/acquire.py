import logging

import numpy as np
import os

from skimage import exposure

from fibsem.structures import (
    BeamType,
    ImageSettings,
    ReferenceImages,
    FibsemImage,
    FibsemRectangle,
)
from fibsem.microscope import FibsemMicroscope


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


def auto_gamma(image: FibsemImage, min_gamma: float = 0.0,max_gamma: float = 2.0, scale_factor: float = 0.1,gamma_threshold: int = 45 ) -> FibsemImage:
    """Automatic gamma correction"""
    std = np.std(image.data)  # unused variable?
    mean = np.mean(image.data)
    diff = mean - 255 / 2.0
    gam = np.clip(
        min_gamma, 1 + diff * scale_factor, max_gamma
    )
    if abs(diff) < gamma_threshold:
        gam = 1.0
    logging.debug(
        f"AUTO_GAMMA | {image.metadata.image_settings.beam_type} | {diff:.3f} | {gam:.3f}"
    )
    image_data = exposure.adjust_gamma(image.data, gam)

    return FibsemImage(data=image_data, metadata=image.metadata)


def new_image(
    microscope: FibsemMicroscope,
    settings: ImageSettings,
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

    if settings.gamma_enabled:
        image = auto_gamma(image)

    # save image
    if settings.save:
        filename = os.path.join(settings.save_path, label)
        image.save(save_path=filename)

    return image
