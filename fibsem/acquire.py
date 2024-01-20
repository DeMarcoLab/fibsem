
import os


from fibsem.structures import (
    BeamType,
    ImageSettings,
    ReferenceImages,
    FibsemImage,
    FibsemRectangle,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.imaging import autogamma


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

    # set filename
    if settings.beam_type is BeamType.ELECTRON:
        filename = f"{settings.filename}_eb"

    if settings.beam_type is BeamType.ION:
        filename = f"{settings.filename}_ib"

    # run autocontrast
    if settings.autocontrast:
        microscope.autocontrast(beam_type=settings.beam_type)

    # acquire the image
    image = microscope.acquire_image(
        image_settings=settings,
    )

    if settings.autogamma:
        image = autogamma.auto_gamma(image, method="autogamma")

    # save image
    if settings.save:
        filename = os.path.join(settings.path, filename)
        image.save(path=filename)

    return image

def acquire_image(microscope:FibsemMicroscope, settings:ImageSettings) -> FibsemImage:
    """ passthrough for new_image to match internal api"""
    return new_image(microscope, settings)

def last_image(microscope: FibsemMicroscope, beam_type: BeamType) -> FibsemImage:
    """_summary_

    Args:
        microscope (FibsemMicroscope): microscope instance
        beam_type (BeamType): beam type for image

    Returns:
        FibsemImage: last image acquired by the microscope
    """
    return microscope.last_image(beam_type=beam_type)



def take_reference_images(
    microscope: FibsemMicroscope, image_settings: ImageSettings
) -> tuple[FibsemImage, FibsemImage]:
    """
    Acquires a pair of electron and ion reference images using the specified imaging settings and
    a FibsemMicroscope instance.

    Args:
        microscope (FibsemMicroscope): A FibsemMicroscope instance for imaging.
        image_settings (ImageSettings): An ImageSettings object with the desired imaging parameters.

    Returns:
        A tuple containing a pair of FibsemImage objects, representing the electron and ion reference
        images acquired using the specified microscope and image settings.

    """
    import time 
    from fibsem.microscope import TescanMicroscope

    tmp_beam_type = image_settings.beam_type
    
    # acquire electron image
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = acquire_image(microscope, image_settings)
    
    # acquire ion image
    image_settings.beam_type = BeamType.ION
    if isinstance(microscope, TescanMicroscope):
        time.sleep(3)
    ib_image = acquire_image(microscope, image_settings)
    image_settings.beam_type = tmp_beam_type  # reset to original beam type

    return eb_image, ib_image
   

def take_set_of_reference_images(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    hfws: tuple[float],
    filename: str = "ref_image",
) -> ReferenceImages:
    """
    Takes a set of reference images at low and high magnification using a FibsemMicroscope.
    The image settings and half-field widths for the low- and high-resolution images are
    specified using an ImageSettings object and a tuple of two floats, respectively.
    The optional filename parameter can be used to customize the image labels.

    Args:
        microscope (FibsemMicroscope): A FibsemMicroscope object to acquire the images from.
        image_settings (ImageSettings): An ImageSettings object with the desired imaging parameters.
        hfws (Tuple[float, float]): A tuple of two floats specifying the horizontal field widths (in microns)
            for the low- and high-resolution images, respectively.
        filename (str, optional): A filename to be included in the image filenames. Defaults to "ref_image".

    Returns:
        A ReferenceImages object containing the low- and high-resolution electron and ion beam images.

    Notes:
        This function sets image_settings.save to True before taking the images.
        The returned ReferenceImages object contains the electron and ion beam images as FibsemImage objects.
    """
    # force save
    image_settings.save = True

    image_settings.hfw = hfws[0]
    image_settings.filename = f"{filename}_low_res"
    low_eb, low_ib = take_reference_images(microscope, image_settings)

    image_settings.hfw = hfws[1]
    image_settings.filename = f"{filename}_high_res"
    high_eb, high_ib = take_reference_images(microscope, image_settings)

    reference_images = ReferenceImages(low_eb, high_eb, low_ib, high_ib)


    # more flexible version
    # reference_images = []
    # for i, hfw in enumerate(hfws):
    #     image_settings.hfw = hfw
    #     image_settings.filename = f"{filename}_res_{i:02d}"
    #     eb_image, ib_image = take_reference_images(microscope, image_settings)
    #     reference_images.append([eb_image, ib_image])

    return reference_images



