import logging

import numpy as np

from scipy import fftpack

from fibsem import acquire, calibration, utils, validation
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.structures import (
    BeamType,
    ImageSettings,
    MicroscopeSettings,
    ReferenceImages,
    FibsemImage,
    FibsemRectangle,
)
from fibsem.microscope import FibsemMicroscope
from typing import Union


def auto_eucentric_correction(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    image_settings: ImageSettings,
    tilt_degrees: int = 25,
    xcorr_limit: int = 250,
) -> None:

    raise NotImplementedError

    image_settings.save = False
    image_settings.beam_type = BeamType.ELECTRON
    calibration.auto_charge_neutralisation(
        microscope.connection, image_settings
    )  # TODO: need to change this function

    for hfw in [400e-6, 150e-6, 80e-6, 80e-6]:
        image_settings.hfw = hfw

        correct_stage_eucentric_alignment(
            microscope,
            settings,
            image_settings,
            tilt_degrees=tilt_degrees,
            xcorr_limit=xcorr_limit,
        )


def beam_shift_alignment(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    ref_image: FibsemImage,
    reduced_area: FibsemRectangle = None,
):
    """Aligns the images by adjusting the beam shift instead of moving the stage.

    This method uses cross-correlation between the reference image and a new image to calculate the
    optimal beam shift for alignment. This approach offers increased precision, but a lower range
    compared to stage movement.

    NOTE: Only shift the ion beam, never the electron beam.

    Args:
        microscope (FibsemMicroscope): An OpenFIBSEM microscope client.
        image_settings (acquire.ImageSettings): Settings for taking the image.
        ref_image (FibsemImage): The reference image to align to.
        reduced_area (FibseRectangle): The reduced area to image with.

    Raises:
        ValueError: If `image_settings.beam_type` is not set to `BeamType.ION`.

    """
    image_settings.reduced_area = reduced_area
    new_image = acquire.new_image(
        microscope, settings=image_settings
    )
    dx, dy, _ = shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )

    # adjust beamshift 
    microscope.beam_shift(dx, dy, image_settings.beam_type)


def correct_stage_drift(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    reference_images: ReferenceImages,
    alignment: tuple[BeamType, BeamType] = (BeamType.ELECTRON, BeamType.ELECTRON),
    rotate: bool = False,
    use_ref_mask: bool = False,
    mask_scale: int = 4,
    xcorr_limit: Union[tuple[int, int], None] = None,
    constrain_vertical: bool = False,
) -> bool:
    """Corrects the stage drift by aligning low- and high-resolution reference images
    using cross-correlation.

    Args:
        microscope (FibsemMicroscope): The microscope used for image acquisition.
        settings (MicroscopeSettings): The settings used for image acquisition.
        reference_images (ReferenceImages): A container of low- and high-resolution
            reference images.
        alignment (tuple[BeamType, BeamType], optional): A tuple of two `BeamType`
            objects, specifying the beam types used for the alignment of low- and
            high-resolution images, respectively. Defaults to (BeamType.ELECTRON,
            BeamType.ELECTRON).
        rotate (bool, optional): Whether to rotate the reference images before
            alignment. Defaults to False.
        use_ref_mask (bool, optional): Whether to apply a mask to the reference images
            before alignment. Defaults to False.
        mask_scale (int, optional): The scale factor used for creating the mask. Defaults
            to 4.
        xcorr_limit (tuple[int, int] | None, optional): A tuple of two integers that
            represent the minimum and maximum cross-correlation values allowed for the
            alignment. If not specified, the values are set to (None, None), which means
            there are no limits. Defaults to None.
        constrain_vertical (bool, optional): Whether to constrain the alignment to the
            vertical axis. Defaults to False.

    Returns:
        bool: True if the stage drift correction was successful, False otherwise.
    """

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
        # settings.image = utils.match_image_settings(
        #     ref_image, settings.image, beam_type=alignment[1]
        # )
        settings.image = ImageSettings.fromFibsemImage(ref_image)
        settings.image.beam_type = alignment[1]
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
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    ref_image: FibsemImage,
    new_image: FibsemImage,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None,
    constrain_vertical: bool = False,
) -> bool:
    """
    Uses cross-correlation to align a new image to a reference image.

    Args:
        microscope: A FibsemMicroscope instance representing the microscope being used.
        settings: A MicroscopeSettings instance representing the settings for the imaging session.
        ref_image: A FibsemImage instance representing the reference image to which the new image will be aligned.
        new_image: A FibsemImage instance representing the new image that will be aligned to the reference image.
        ref_mask: A numpy array representing a mask to apply to the reference image during alignment. Default is None.
        xcorr_limit: An integer representing the limit for the cross-correlation coefficient. If the coefficient is below
            this limit, alignment will fail. Default is None.
        constrain_vertical: A boolean indicating whether to constrain movement to the vertical axis. If True, movement
            will be restricted to the vertical axis, which is useful for eucentric movement. If False, movement will be
            allowed on both the X and Y axes. Default is False.

    Returns:
        A boolean indicating whether the alignment was successful. True if the alignment was successful, False otherwise.
    """
    # get beam type
    ref_beam_type = BeamType[ref_image.metadata.image_settings.beam_type.name.upper()]
    new_beam_type = BeamType[new_image.metadata.image_settings.beam_type.name.upper()]

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

    shift_within_tolerance = (
        validation.check_shift_within_tolerance(  # TODO: Abstract validation.py
            dx=dx, dy=dy, ref_image=ref_image, limit=0.5
        )
    )

    if shift_within_tolerance:

        # vertical constraint = eucentric movement
        if constrain_vertical:
            microscope.eucentric_move(
                settings=settings, dy=-dy
            )  # FLAG_TEST
        else:
            # move the stage
            microscope.stable_move(
                settings=settings,
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
    """Calculates the shift between two images by cross-correlating them and finding the position of maximum correlation.

    Args:
        ref_image (FibsemImage): The reference image.
        new_image (FibsemImage): The new image to align to the reference.
        lowpass (int, optional): The low-pass filter frequency (in pixels) for the bandpass filter used to
            enhance the correlation signal. Defaults to 128.
        highpass (int, optional): The high-pass filter frequency (in pixels) for the bandpass filter used to
            enhance the correlation signal. Defaults to 6.
        sigma (int, optional): The standard deviation (in pixels) of the Gaussian filter used to create the bandpass
            mask. Defaults to 6.
        use_rect_mask (bool, optional): Whether to use a rectangular mask for the correlation. If True, the correlation
            is performed only inside a rectangle that covers most of the image, to reduce the effect of noise at the
            edges. Defaults to False.
        ref_mask (np.ndarray, optional): A mask to apply to the reference image before correlation. If not None,
            it should be a binary array with the same shape as the images. Pixels with value 0 will be ignored in the
            correlation. Defaults to None.
        xcorr_limit (int, optional): If not None, the correlation map will be circularly masked to a square
            with sides of length 2 * xcorr_limit + 1, centred on the maximum correlation peak. This can be used to
            limit the search range and improve the accuracy of the shift. Defaults to None.

    Returns:
        A tuple (x_shift, y_shift, xcorr), where x_shift and y_shift are the shifts along x and y (in meters),
        and xcorr is the cross-correlation map between the images.
    """
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
    """
    Cross-correlate two images using Fourier convolution matching.

    Args:
        img1 (np.ndarray): The reference image.
        img2 (np.ndarray): The new image to be cross-correlated with the reference.
        bandpass (np.ndarray, optional): A bandpass mask to apply to both images before cross-correlation. Defaults to None.

    Returns:
        np.ndarray: The cross-correlation map between the two images.
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
