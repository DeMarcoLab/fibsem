import logging

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    MoveSettings,
    Rectangle,
    StagePosition,
)
from scipy import fftpack

from fibsem import acquire, calibration, movement, utils, validation
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.structures import (
    BeamType,
    ImageSettings,
    MicroscopeSettings,
    ReferenceImages,
    FibsemImage,
)


def auto_eucentric_correction(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    image_settings: ImageSettings,
    tilt_degrees: int = 25,
    xcorr_limit: int = 250,
) -> None:

    image_settings.save = False
    image_settings.beam_type = BeamType.ELECTRON
    calibration.auto_charge_neutralisation(microscope, image_settings)

    for hfw in [400e-6, 150e-6, 80e-6, 80e-6]:
        image_settings.hfw = hfw

        correct_stage_eucentric_alignment(
            microscope,
            settings,
            image_settings,
            tilt_degrees=tilt_degrees,
            xcorr_limit=xcorr_limit,
        )


def correct_stage_eucentric_alignment(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    image_settings: ImageSettings,
    tilt_degrees: float = 52,
    xcorr_limit: int = 250,
) -> None:

    # TODO: does the direction of tilt change this?

    # take images
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)

    # tilt stretch to match feature sizes
    ib_image = image_utils.cosine_stretch(ib_image, tilt_degrees)

    # cross correlate
    lp_px = 128  # int(max(ib_image.data.shape) / 12)
    hp_px = 6  # int(max(ib_image.data.shape) / 256)
    sigma = 6

    ref_mask = masks.create_circle_mask(eb_image.data.shape, radius=64)

    dx, dy, xcorr = shift_from_crosscorrelation(
        eb_image,
        ib_image,
        lowpass=lp_px,
        highpass=hp_px,
        sigma=sigma,
        use_rect_mask=True,
        ref_mask=ref_mask,
        xcorr_limit=xcorr_limit,
    )

    # move vertically to correct eucentric position
    movement.move_stage_eucentric_correction(microscope, settings, dy)


def coarse_eucentric_alignment(
    microscope: SdbMicroscopeClient,
    hfw: float = 30e-6,
    eucentric_height: float = 3.91e-3,
) -> None:

    # focus and link stage
    calibration.auto_link_stage(microscope, hfw=hfw)

    # move to eucentric height
    stage = microscope.specimen.stage
    move_settings = MoveSettings(link_z_y=True)
    z_move = StagePosition(z=eucentric_height, coordinate_system="Specimen")
    stage.absolute_move(z_move, move_settings)


def beam_shift_alignment(
    microscope: SdbMicroscopeClient,
    image_settings: ImageSettings,
    ref_image: FibsemImage,
    reduced_area: Rectangle,
):
    """Align the images by adjusting the beam shift, instead of moving the stage
            (increased precision, lower range)
        NOTE: only shift the ion beam, never electron

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client
        image_settings (acquire.ImageSettings): settings for taking image
        ref_image (AdornedImage): reference image to align to
        reduced_area (Rectangle): The reduced area to image with.
    """

    # # align using cross correlation
    new_image = acquire.new_image(
        microscope, settings=image_settings, reduced_area=reduced_area
    )
    dx, dy, _ = shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )

    # adjust beamshift
    microscope.beams.ion_beam.beam_shift.value += (-dx, dy)


def correct_stage_drift(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    reference_images: ReferenceImages,
    alignment: tuple(BeamType) = (BeamType.ELECTRON, BeamType.ELECTRON),
    rotate: bool = False,
    use_ref_mask: bool = False,
    mask_scale: int = 4,
    xcorr_limit: tuple = None,
    constrain_vertical: bool = False,
) -> bool:
    """Correct the stage drift by crosscorrelating low-res and high-res reference images"""

    # set reference images
    if alignment[0] is BeamType.ELECTRON:
        ref_lowres, ref_highres = (
            reference_images.low_res_eb,
            reference_images.high_res_eb,
        )
    if alignment[0] is BeamType.ION:
        ref_lowres, ref_highres = (
            reference_images.low_res_ib,
            reference_images.high_res_ib,
        )

    if xcorr_limit is None:
        xcorr_limit = (None, None)

    # rotate reference
    if rotate:
        ref_lowres = image_utils.rotate_image(ref_lowres)
        ref_highres = image_utils.rotate_image(ref_highres)

    # align lowres, then highres
    for i, ref_image in enumerate([ref_lowres, ref_highres]):

        if use_ref_mask:
            ref_mask = masks.create_lamella_mask(
                ref_image,
                settings.protocol["lamella"],
                scale=mask_scale,
                use_trench_height=True,
            )  # TODO: refactor, liftout specific
        else:
            ref_mask = None

        # take new images
        # set new image settings (same as reference)
        settings.image = utils.match_image_settings(
            ref_image, settings.image, beam_type=alignment[1]
        )
        new_image = acquire.new_image(microscope, settings.image)

        # crosscorrelation alignment
        ret = align_using_reference_images(
            microscope,
            settings,
            ref_image,
            new_image,
            ref_mask=ref_mask,
            xcorr_limit=xcorr_limit[i],
            constrain_vertical=constrain_vertical,
        )

        if ret is False:
            break  # cross correlation has failed...

    return ret


def align_using_reference_images(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    ref_image: FibsemImage,
    new_image: FibsemImage,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None,
    constrain_vertical: bool = False,
) -> bool:

    # get beam type
    ref_beam_type = BeamType[ref_image.metadata.image_settings.beam_type.upper()]
    new_beam_type = BeamType[new_image.metadata.image_settings.beam_type.upper()]

    logging.info(
        f"aligning {ref_beam_type.name} reference image to {new_beam_type.name}."
    )
    sigma = 6
    hp_px = 8
    lp_px = 128  # MAGIC_NUMBER

    dx, dy, xcorr = shift_from_crosscorrelation(
        ref_image,
        new_image,
        lowpass=lp_px,
        highpass=hp_px,
        sigma=sigma,
        use_rect_mask=True,
        ref_mask=ref_mask,
        xcorr_limit=xcorr_limit,
    )

    shift_within_tolerance = validation.check_shift_within_tolerance(
        dx=dx, dy=dy, ref_image=ref_image, limit=0.5
    )

    if shift_within_tolerance:

        # vertical constraint = eucentric movement
        if constrain_vertical:
            movement.move_stage_eucentric_correction(
                microscope, settings=settings, dy=-dy
            )  # FLAG_TEST
        else:
            # move the stage
            movement.move_stage_relative_with_corrected_movement(
                microscope,
                settings,
                dx=dx,
                dy=-dy,
                beam_type=new_beam_type,
            )

    return shift_within_tolerance


def shift_from_crosscorrelation(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    lowpass: int = 128,
    highpass: int = 6,
    sigma: int = 6,
    use_rect_mask: bool = False,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None,
) -> tuple[float, float, np.ndarray]:

    # get pixel_size
    pixelsize_x = new_image.metadata.pixel_size.x
    pixelsize_y = new_image.metadata.pixel_size.y

    # normalise both images
    ref_data_norm = image_utils.normalise_image(ref_image)
    new_data_norm = image_utils.normalise_image(new_image)

    # cross-correlate normalised images
    if use_rect_mask:
        rect_mask = masks._mask_rectangular(new_data_norm.shape)
        ref_data_norm = rect_mask * ref_data_norm
        new_data_norm = rect_mask * new_data_norm

    if ref_mask is not None:
        ref_data_norm = ref_mask * ref_data_norm  # mask the reference

    # bandpass mask
    bandpass = masks.create_bandpass_mask(
        shape=ref_data_norm.shape, lp=lowpass, hp=highpass, sigma=sigma
    )

    # crosscorrelation
    xcorr = crosscorrelation_v2(ref_data_norm, new_data_norm, bandpass=bandpass)

    # limit xcorr range
    if xcorr_limit:
        xcorr = masks.apply_circular_mask(xcorr, xcorr_limit)

    # calculate maximum crosscorrelation
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)  # TODO: backwards
    cen = np.asarray(xcorr.shape) / 2
    err = np.array(cen - [maxX, maxY], int)

    # calculate shift in metres
    x_shift = err[1] * pixelsize_x
    y_shift = err[0] * pixelsize_y  # this could be the issue?

    logging.debug(f"cross-correlation:")
    logging.debug(f"pixelsize: x: {pixelsize_x:.2e}, y: {pixelsize_y:.2e}")
    logging.debug(f"maxX: {maxX}, {maxY}, centre: {cen}")
    logging.debug(f"x: {err[1]}px, y: {err[0]}px")
    logging.debug(f"x: {x_shift:.2e}m, y: {y_shift:.2e} meters")

    # metres
    return x_shift, y_shift, xcorr


def crosscorrelation_v2(
    img1: np.ndarray, img2: np.ndarray, bandpass: np.ndarray = None
) -> np.ndarray:
    """Cross-correlate images (fourier convolution matching)

    Args:
        img1 (np.ndarray): reference_image
        img2 (np.ndarray): new image
        bandpass (np.ndarray)

    Returns:
        np.ndarray: crosscorrelation map
    """
    if img1.shape != img2.shape:
        err = (
            f"Image 1 {img1.shape} and Image 2 {img2.shape} need to have the same shape"
        )
        logging.error(err)
        raise ValueError(err)

    if bandpass is None:
        bandpass = np.ones_like(img1)

    n_pixels = img1.shape[0] * img1.shape[1]

    img1ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img1)))
    tmp = img1ft * np.conj(img1ft)
    img1ft = n_pixels * img1ft / np.sqrt(tmp.sum())

    img2ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img2)))
    img2ft[0, 0] = 0
    tmp = img2ft * np.conj(img2ft)

    img2ft = n_pixels * img2ft / np.sqrt(tmp.sum())

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    # ax[0].imshow(np.fft.ifft2(img1ft).real)
    # ax[1].imshow(np.fft.ifft2(img2ft).real)
    # plt.show()

    # plt.title("Power Spectra")
    # plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(img1)))))
    # plt.show()

    xcorr = np.real(np.fft.fftshift(np.fft.ifft2(img1ft * np.conj(img2ft))))

    return xcorr
