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
    """
    Acquires a pair of electron and ion reference images using the specified imaging settings and
    a FibsemMicroscope instance.

    Args:
        microscope (FibsemMicroscope): A FibsemMicroscope instance for imaging.
        image_settings (ImageSettings): An ImageSettings object with the desired imaging parameters.

    Returns:
        A list containing a pair of FibsemImage objects, representing the electron and ion reference
        images acquired using the specified microscope and image settings.

    Notes:
        - This function temporarily changes the `image_settings.beam_type` to `BeamType.ELECTRON`
          and then `BeamType.ION` to acquire the electron and ion reference images, respectively.
          It resets the `image_settings.beam_type` to the original value after acquiring the images.
        - The `FibsemImage` objects in the returned list contain the image data as numpy arrays,
          as well as other image metadata.
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
   


def take_set_of_reference_images(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    hfws: tuple[float],
    label: str = "ref_image",
) -> ReferenceImages:
    """
    Takes a set of reference images at low and high magnification using a FibsemMicroscope.
    The image settings and half-field widths for the low- and high-resolution images are
    specified using an ImageSettings object and a tuple of two floats, respectively.
    The optional label parameter can be used to customize the image labels.

    Args:
        microscope (FibsemMicroscope): A FibsemMicroscope object to acquire the images from.
        image_settings (ImageSettings): An ImageSettings object with the desired imaging parameters.
        hfws (Tuple[float, float]): A tuple of two floats specifying the half-field widths (in microns)
            for the low- and high-resolution images, respectively.
        label (str, optional): A label to be included in the image filenames. Defaults to "ref_image".

    Returns:
        A ReferenceImages object containing the low- and high-resolution electron and ion beam images.

    Notes:
        This function sets image_settings.save to True before taking the images.
        The returned ReferenceImages object contains the electron and ion beam images as FibsemImage objects.
    """
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


def auto_gamma(
    image: FibsemImage,
    min_gamma: float = 0.15,
    max_gamma: float = 1.8,
    scale_factor: float = 0.01,
    gamma_threshold: int = 45
) -> FibsemImage:
    """
    Applies automatic gamma correction to the input `FibsemImage`.

    Args:
        image (FibsemImage): The input `FibsemImage` to apply gamma correction to.
        min_gamma (float): The minimum gamma value allowed in the correction. Defaults to 0.15.
        max_gamma (float): The maximum gamma value allowed in the correction. Defaults to 1.8.
        scale_factor (float): A scaling factor to adjust the gamma correction range based on the image
            brightness. Defaults to 0.01.
        gamma_threshold (int): The maximum threshold of brightness difference from the mid-gray value
            (i.e., 128) before the gamma value is forced to 1.0. Defaults to 45.

    Returns:
        A new `FibsemImage` object containing the gamma-corrected image data, with the same metadata
        as the input image.

    Notes:
        - This function applies gamma correction to the input image using the `skimage.exposure.adjust_gamma`
          function, with the gamma value computed based on the mean intensity of the image.
        - If the difference between the mean image intensity and the mid-gray value (i.e., 128) is greater
          than the specified `gamma_threshold`, the gamma value is forced to 1.0 to avoid over-correction.
        - The `FibsemImage` object in the returned list contains the gamma-corrected image data as a
          numpy array, as well as other image metadata.
    """

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
    """Apply the given image settings and acquire a new image.

    Args:
        microscope (FibsemMicroscope): The FibsemMicroscope instance used to acquire the image.
        settings (ImageSettings): The image settings used to acquire the image.

    Returns:
        FibsemImage: The acquired image.
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
