import logging
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                         GrabFrameSettings,
                                                         Rectangle,
                                                         RunAutoCbSettings)
from skimage import exposure
from fibsem.structures import BeamType, GammaSettings, ImageSettings
from fibsem import utils


def autocontrast(microscope: SdbMicroscopeClient, beam_type=BeamType.ELECTRON) -> None:
    """Automatically adjust the microscope image contrast."""
    microscope.imaging.set_active_view(beam_type.value)

    RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  # low resolution, so as not to damage the sample
        number_of_frames=5,
    )
    logging.info("automatically adjusting contrast...")
    microscope.auto_functions.run_auto_cb()


def take_reference_images(
    microscope: SdbMicroscopeClient, image_settings: ImageSettings
) -> list[AdornedImage]:
    tmp_beam_type = image_settings.beam_type
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = new_image(microscope, image_settings)
    image_settings.beam_type = BeamType.ION
    ib_image = new_image(microscope, image_settings)
    image_settings.beam_type = tmp_beam_type  # reset to original beam type
    return eb_image, ib_image


def gamma_correction(image: AdornedImage, settings: GammaSettings) -> AdornedImage:
    """Automatic gamma correction"""
    std = np.std(image.data)
    mean = np.mean(image.data)
    diff = mean - 255 / 2.0
    gam = np.clip(
        settings.min_gamma, 1 + diff * settings.scale_factor, settings.max_gamma
    )
    if abs(diff) < settings.threshold:
        gam = 1.0
    logging.info(
        f"GAMMA_CORRECTION | {image.metadata.acquisition.beam_type} | {diff:.3f} | {gam:.3f}"
    )
    image_data = exposure.adjust_gamma(image.data, gam)
    reference = AdornedImage(data=image_data)
    reference.metadata = image.metadata
    image = reference
    return image

# TODO: change set_active_view to set_active_device... for better stability
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
            AdornedImage: new autoscript adorned image
    """
    frame_settings = GrabFrameSettings(
        resolution=settings.resolution,
        dwell_time=settings.dwell_time,
        reduced_area=reduced_area,
    )

    if settings.beam_type == BeamType.ELECTRON:
        hfw_limits = microscope.beams.electron_beam.horizontal_field_width.limits
        settings.hfw = np.clip(settings.hfw, hfw_limits.min, hfw_limits.max)
        microscope.beams.electron_beam.horizontal_field_width.value = settings.hfw
        label = settings.label + "_eb"
    if settings.beam_type == BeamType.ION:
        hfw_limits = microscope.beams.ion_beam.horizontal_field_width.limits
        settings.hfw = np.clip(settings.hfw, hfw_limits.min, hfw_limits.max)
        microscope.beams.ion_beam.horizontal_field_width.value = settings.hfw
        label = settings.label + "_ib"

    if settings.autocontrast:
        autocontrast(microscope, beam_type=settings.beam_type)

    image = acquire_image(
        microscope=microscope, settings=frame_settings, beam_type=settings.beam_type,
    )

    # apply gamma correction
    if settings.gamma.enabled:

        # gamma parameters
        image = gamma_correction(image, settings.gamma)

    if settings.save:
        utils.save_image(image=image, save_path=settings.save_path, label=label)
    return image


def last_image(
    microscope: SdbMicroscopeClient, beam_type=BeamType.ELECTRON
) -> AdornedImage:
    """Get the last previously acquired ion or electron beam image.

    Parameters
    ----------
    microscope : Autoscript microscope object.
    beam_type :

    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    microscope.imaging.set_active_view(beam_type.value)
    image = microscope.imaging.get_image()
    return image


def acquire_image(
    microscope: SdbMicroscopeClient,
    settings: GrabFrameSettings = None,
    beam_type: BeamType = BeamType.ELECTRON,
) -> AdornedImage:
    """Take new electron or ion beam image.
    Returns
    -------
    AdornedImage
        If the returned AdornedImage is named 'image', then:
        image.data = a numpy array of the image pixels
        image.metadata.binary_result.pixel_size.x = image pixel size in x
        image.metadata.binary_result.pixel_size.y = image pixel size in y
    """
    logging.info(f"acquiring new {beam_type.name} image.")
    microscope.imaging.set_active_view(beam_type.value)
    if settings is not None:
        image = microscope.imaging.grab_frame(settings)
    else:
        image = microscope.imaging.grab_frame()
    return image



def update_image_settings_v3(
    settings: dict,
    resolution=None,
    dwell_time=None,
    hfw=None,
    autocontrast=None,
    beam_type=None,
    gamma=None,
    save=None,
    label=None,
    path=None,
):
    """Update image settings. Uses default values if not supplied

    Args:
        settings (dict): the default settings dictionary
        resolution (str, optional): image resolution. Defaults to None.
        dwell_time (float, optional): image dwell time. Defaults to None.
        hfw (float, optional): image horizontal field width. Defaults to None.
        autocontrast (bool, optional): use autocontrast. Defaults to None.
        beam_type (BeamType, optional): beam type to image with (Electron, Ion). Defaults to None.
        gamma (GammaSettings, optional): gamma correction settings. Defaults to None.
        save (bool, optional): save the image. Defaults to None.
        label (str, optional): image filename . Defaults to None.
        save_path (Path, optional): directory to save image. Defaults to None.
    """
    gamma_settings = GammaSettings(
        enabled=settings["calibration"]["gamma"]["enabled"],
        min_gamma=settings["calibration"]["gamma"]["min_gamma"],
        max_gamma=settings["calibration"]["gamma"]["max_gamma"],
        scale_factor=settings["calibration"]["gamma"]["scale_factor"],
        threshold=settings["calibration"]["gamma"]["threshold"],
    )

    # new image_settings
    image_settings = ImageSettings(
        resolution=settings["calibration"]["imaging"]["resolution"]
        if resolution is None
        else resolution,
        dwell_time=settings["calibration"]["imaging"]["dwell_time"]
        if dwell_time is None
        else dwell_time,
        hfw=settings["calibration"]["imaging"]["horizontal_field_width"]
        if hfw is None
        else hfw,
        autocontrast=settings["calibration"]["imaging"]["autocontrast"]
        if autocontrast is None
        else autocontrast,
        beam_type=BeamType.ELECTRON if beam_type is None else beam_type,
        gamma=gamma_settings if gamma is None else gamma,
        save=bool(settings["calibration"]["imaging"]["save"]) if save is None else save,
        save_path="" if path is None else path, # TODO: change to os.getcwd?
        label=utils.current_timestamp() if label is None else label,
    )

    return image_settings


def reset_beam_shifts(microscope: SdbMicroscopeClient):
    """Set the beam shift to zero for the electron and ion beams

    Args:
        microscope (SdbMicroscopeClient): Autoscript microscope object
    """
    from autoscript_sdb_microscope_client.structures import (GrabFrameSettings,
                                                             Point)

    # reset zero beamshift
    logging.info(
        f"reseting ebeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value} "
    )
    microscope.beams.electron_beam.beam_shift.value = Point(0, 0)
    logging.info(
        f"reseting ibeam shift to (0, 0) from: {microscope.beams.electron_beam.beam_shift.value} "
    )
    microscope.beams.ion_beam.beam_shift.value = Point(0, 0)
    logging.info(f"reset beam shifts to zero complete")
