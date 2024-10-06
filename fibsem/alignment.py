import logging
from typing import Optional, Tuple, Union

import numpy as np

from fibsem import acquire, utils, validation
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    MicroscopeSettings,
    ReferenceImages,
)

def auto_eucentric_correction(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    image_settings: ImageSettings,
    tilt_degrees: int = 25,
    xcorr_limit: int = 250,
) -> None:

    raise NotImplementedError

    # image_settings.save = False
    # image_settings.beam_type = BeamType.ELECTRON
    # calibration.auto_charge_neutralisation(
    #     microscope.connection, image_settings
    # )  # TODO: need to change this function

    # for hfw in [400e-6, 150e-6, 80e-6, 80e-6]:
    #     image_settings.hfw = hfw

    #     correct_stage_eucentric_alignment(
    #         microscope,
    #         settings,
    #         image_settings,
    #         tilt_degrees=tilt_degrees,
    #         xcorr_limit=xcorr_limit,
    #     )


def beam_shift_alignment(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    ref_image: FibsemImage,
    reduced_area: Optional[FibsemRectangle] = None,
    alignment_current: Optional[float] = None,
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
    if alignment_current is not None:
        initial_current = microscope.get("current", image_settings.beam_type)
        microscope.set("current", alignment_current, image_settings.beam_type)

    # TODO: we should just use the same settings as the reference image, no need to pass in image_settings?
    
    import time
    time.sleep(3) # threading is too fast?
    ref_image_settings = ImageSettings.fromFibsemImage(ref_image)
    ref_image_settings.beam_type = BeamType.ION
    ref_image_settings.reduced_area = reduced_area
    ref_image_settings.autocontrast = False
    ref_image_settings.save = True
    ref_image_settings.filename = image_settings.filename
    new_image = acquire.new_image(
        microscope, settings=ref_image_settings
    )
    dx, dy, _ = shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )
    image_settings.autocontrast = True
    

    # adjust beamshift (reverse direction for x)
    microscope.beam_shift(-dx, dy, image_settings.beam_type)

    # reset beam current
    if alignment_current is not None:
        microscope.set("current", initial_current, image_settings.beam_type)


def beam_shift_alignment_v2(
    microscope: FibsemMicroscope,
    ref_image: FibsemImage,
    alignment_current: Optional[float] = None,
):
    """Aligns the images by adjusting the beam shift instead of moving the stage.

    This method uses cross-correlation between the reference image and a new image to calculate the
    optimal beam shift for alignment. This approach offers increased precision, but a lower range
    compared to stage movement.

    NOTE: Only shift the ion beam, never the electron beam.

    Args:
        microscope (FibsemMicroscope): An OpenFIBSEM microscope client.
        ref_image (FibsemImage): The reference image to align to.

    Raises:
        ValueError: If `image_settings.beam_type` is not set to `BeamType.ION`.

    """

    import time
    time.sleep(3) # threading is too fast?
    image_settings = ImageSettings.fromFibsemImage(ref_image)
    image_settings.autocontrast = False
    image_settings.save = True
    image_settings.filename = f"beam_shift_alignment_{utils.current_timestamp_v2()}"

    # set alignment current
    if alignment_current is not None:
        initial_current = microscope.get("current", image_settings.beam_type)
        microscope.set("current", alignment_current, image_settings.beam_type)

    new_image = acquire.new_image(
        microscope, settings=image_settings
    )
    dx, dy, _ = shift_from_crosscorrelation(
        ref_image, new_image, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )    

    # adjust beamshift (reverse direction)
    microscope.beam_shift(-dx, -dy, image_settings.beam_type)

    # reset beam current
    if alignment_current is not None:
        microscope.set("current", initial_current, image_settings.beam_type)
    

def correct_stage_drift(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    reference_images: ReferenceImages,
    alignment: Tuple[BeamType, BeamType] = (BeamType.ELECTRON, BeamType.ELECTRON),
    rotate: bool = False,
    ref_mask_rad: int = 512,
    xcorr_limit: Union[Tuple[int, int], None] = None,
    constrain_vertical: bool = False,
    use_beam_shift: bool = False,
) -> bool:
    """Corrects the stage drift by aligning low- and high-resolution reference images
    using cross-correlation.

    Args:
        microscope (FibsemMicroscope): The microscope used for image acquisition.
        settings (MicroscopeSettings): The settings used for image acquisition.
        reference_images (ReferenceImages): A container of low- and high-resolution
            reference images.
        alignment (Tuple[BeamType, BeamType], optional): A tuple of two `BeamType`
            objects, specifying the beam types used for the alignment of low- and
            high-resolution images, respectively. Defaults to (BeamType.ELECTRON,
            BeamType.ELECTRON).
        rotate (bool, optional): Whether to rotate the reference images before
            alignment. Defaults to False.
        ref_mask_rad (int, optional): The radius of the circular mask used for reference
        xcorr_limit (Tuple[int, int] | None, optional): A tuple of two integers that
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

        ref_mask = masks.create_circle_mask(ref_image.data.shape, ref_mask_rad)

        # take new images
        # set new image settings (same as reference)
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
            use_beam_shift=use_beam_shift,
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
    use_beam_shift: bool = False,
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
            microscope.vertical_move( dx=0, dy=-dy
            )  # FLAG_TEST
        else:
            if use_beam_shift:
                # move the beam shift
                microscope.beam_shift(dx=-dx, dy=-dy, beam_type=new_beam_type)
            else:
                # move the stage
                microscope.stable_move(
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
) -> Tuple[float, float, np.ndarray]:
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
    dx = err[1] * pixelsize_x
    dy = err[0] * pixelsize_y  # this could be the issue?

    msgd = {"msg": "cross-correlation", "pixelsize": (pixelsize_x, pixelsize_y), 
        "max": (maxX, maxY), "centre": cen, "shift": (err[1], err[0]), "shift_meters": (dx, dy)}
    logging.debug(msgd)

    # logging.debug(f"cross-correlation:")
    # logging.debug(f"pixelsize: x: {pixelsize_x:.2e}, y: {pixelsize_y:.2e}")
    # logging.debug(f"maxX: {maxX}, {maxY}, centre: {cen}")
    # logging.debug(f"x: {err[1]}px, y: {err[0]}px")
    # logging.debug(f"x: {dx:.2e}m, y: {dy:.2e} meters")

    # save data
    _save_alignment_data(
        ref_image=ref_image,
        new_image=new_image,
        bandpass=bandpass,
        xcorr=xcorr,
        use_rect_mask=use_rect_mask,
        ref_mask=ref_mask,
        xcorr_limit=xcorr_limit,
        lowpass=lowpass,
        highpass=highpass,
        sigma=sigma,
        dx=dx,
        dy=dy,
        pixelsize_x=pixelsize_x,
        pixelsize_y=pixelsize_y,
    )

    # metres
    return dx, dy, xcorr


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

def _save_alignment_data(
    ref_image: FibsemImage,
    new_image: FibsemImage,
    bandpass: np.ndarray,
    xcorr: np.ndarray,
    ref_mask: np.ndarray = None,
    lowpass: float = None,
    highpass: float = None,
    sigma: float = None,
    xcorr_limit: float = None,
    use_rect_mask: bool = False,
    dx: float = None,
    dy: float = None,
    pixelsize_x: float = None,
    pixelsize_y: float = None,
    

):
    """Save alignment data to disk."""
    
    import os

    import pandas as pd
    import tifffile as tff

    from fibsem import config as cfg

    ts = utils.current_timestamp_v2()
    fname = os.path.join(cfg.DATA_CC_PATH, str(ts))

    # save fibsem images
    ref_image.save(fname + "_ref.tif")
    new_image.save(fname + "_new.tif")

    # convert to tiff , save
    tff.imwrite(fname + "_xcorr.tif", xcorr)
    tff.imwrite(fname + "_bandpass.tif", bandpass)
    if ref_mask is not None:
        tff.imwrite(fname + "_ref_mask.tif", ref_mask)

    info = {
        # "ref_image": ref_image, "new_image": new_image, "bandpass": bandpass, "xcorr": xcorr, "ref_mask": ref_mask,
        "lowpass": lowpass, "highpass": highpass, "sigma": sigma,
        "pixelsize_x": pixelsize_x, "pixelsize_y": pixelsize_y, 
        "use_rect_mask": use_rect_mask, "xcorr_limit": xcorr_limit, "ref_mask": ref_mask is not None,
        "dx": dx, "dy": dy, "fname": fname, "timestamp": ts }


    df = pd.DataFrame.from_dict(info, orient='index').T
    
    # save the dataframe to a csv file, append if the file already exists
    DATAFRAME_PATH = os.path.join(cfg.DATA_CC_PATH, "data.csv")
    if os.path.exists(DATAFRAME_PATH):
        df_tmp = pd.read_csv(DATAFRAME_PATH)
        df = pd.concat([df_tmp, df], axis=0, ignore_index=True)
    
    df.to_csv(DATAFRAME_PATH, index=False)

def _multi_step_alignment(microscope: FibsemMicroscope, image_settings: ImageSettings, 
    ref_image: FibsemImage, reduced_area: FibsemRectangle, alignment_current: float = None, steps:int = 3) -> None:
    
    # set alignment current
    if alignment_current is not None:
        initial_current = microscope.get("current", image_settings.beam_type)
        microscope.set("current", alignment_current, image_settings.beam_type)
        autocontrast = image_settings.autocontrast
        image_settings.autocontrast = False

    base_label = image_settings.filename
    for i in range(steps):
        image_settings.filename = f"{base_label}_{i:02d}"
        image_settings.beam_type = BeamType.ION
        beam_shift_alignment(microscope, image_settings, 
                                        ref_image=ref_image,
                                            reduced_area=reduced_area)
    # reset beam current
    if alignment_current is not None:
        microscope.set("current", initial_current, image_settings.beam_type)
        image_settings.autocontrast = autocontrast


def multi_step_alignment_v2(microscope: FibsemMicroscope, 
    ref_image: FibsemImage, beam_type: BeamType, 
    alignment_current: float = None, steps:int = 3) -> None:
    """Runs the beam shift alignment multiple times. Optionally sets the beam current before alignment."""
    # set alignment current
    if alignment_current is not None:
        initial_current = microscope.get("current", beam_type)
        microscope.set("current", alignment_current, beam_type)

    for i in range(steps):
        beam_shift_alignment_v2(microscope, ref_image=ref_image)
    
    # reset beam current
    if alignment_current is not None:
        microscope.set("current", initial_current, beam_type)


### From AutoLamella v1

# def realign(microscope, new_image, reference_image):
#     """Realign to reference image using beam shift.

#     Parameters
#     ----------
#     microscope : Autoscript microscope object
#     new_image : The most recent image acquired.
#         Must have the same dimensions and relative position as the reference.
#     reference_image : The reference image to align with.
#         Muast have the same dimensions and relative position as the new image
#     Returns
#     -------
#     microscope.beams.ion_beam.beam_shift.value
#         The current beam shift position (after any realignment)
#     """
#     from autoscript_core.common import ApplicationServerException

#     shift_in_meters = _calculate_beam_shift(new_image, reference_image)
#     try:
#         microscope.beams.ion_beam.beam_shift.value += shift_in_meters
#     except ApplicationServerException:
#         logging.warning(
#             "Cannot move beam shift beyond limits, "
#             "will continue with no beam shift applied."
#         )
#     return microscope.beams.ion_beam.beam_shift.value
    
# def _calculate_beam_shift(image_1, image_2):
#     """Cross correlation to find shift between two images.

#     Parameters
#     ----------
#     image_1 : AdornedImage
#         Original image to use as reference point.
#     image_2 : AdornedImage
#         Possibly shifted image to align with original.

#     Returns
#     -------
#     realspace_beam_shift
#         Beam shift in x, y format (meters), list of floats.

#     Raises
#     ------
#     ValueError
#         If images are not the same dimensions, raise a ValueError.
#     """
#     if image_1.data.shape != image_2.data.shape:
#         raise ValueError("Images must be the same shape for cross correlation.")
#     mask_image_1 = _mask_circular(image_1.data.shape)
#     mask_image_2 = _mask_rectangular(image_2.data.shape)
#     norm_image_1 = _normalize_image(image_1.data) * mask_image_1
#     norm_image_2 = _normalize_image(image_2.data) * mask_image_2
#     pixel_shift = _simple_register_translation(norm_image_2, norm_image_1)
#     # Autoscript y-axis has an inverted positive direction
#     pixel_shift[1] = -pixel_shift[1]
#     pixelsize_x = image_1.metadata.binary_result.pixel_size.x
#     realspace_beam_shift = pixel_shift * pixelsize_x
#     logging.info("pixel_shift calculated = {}".format(pixel_shift))
#     logging.info("realspace_beam_shift calculated = {}".format(realspace_beam_shift))
#     return realspace_beam_shift


# def _simple_register_translation(src_image, target_image, max_shift_mask=None):
#     """Calculate pixel shift between two input images.

#     This function runs with numpy or cupy for GPU acceleration.

#     Parameters
#     ----------
#     src_image : array
#         Reference image.
#     target_image : array
#         Image to register.  Must be same dimensionality as ``src_image``.
#     max_shift_mask : array
#         The fourier mask restricting the maximum allowable pixel shift.

#     Returns
#     -------
#     shifts : ndarray
#         Pixel shift in x, y order between target and source image.

#     References
#     ----------
#     scikit-image register_translation function in the skimage.feature module.
#     """
#     src_freq = np.fft.fftn(src_image)
#     target_freq = np.fft.fftn(target_image)
#     print('using bp mask')
#     bp_mask = _bandpass_mask(target_image.data.shape, target_image.data.shape[0] / 3, inner_radius=2, sigma=3)
#     bp_target_freq = bp_mask * target_freq

#     # Whole-pixel shift - Compute cross-correlation by an IFFT
#     shape = src_freq.shape
#     image_product = src_freq * bp_target_freq.conj()
#     cross_correlation = np.fft.ifftn(image_product)
#     # Locate maximum
#     maxima = np.unravel_index(
#         np.argmax(np.abs(cross_correlation)), cross_correlation.shape
#     )
#     midpoints = np.array([float(np.round(axis_size / 2)) for axis_size in shape])
#     shifts = np.array(maxima, dtype=np.float64)
#     shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
#     shifts = np.flip(shifts, axis=0).astype(int)  # x, y order
#     return shifts


# def _normalize_image(image, mask=None):
#     """Ensure the image mean is zero and the standard deviation is one.

#     Parameters
#     ----------
#     image : ndarray
#         The input image array.
#     mask : ndarray, optional
#         A mask image containing values between zero and one.
#         Dimensions must match the input image.

#     Returns
#     -------
#     ndarray
#         The normalized image.
#         The mean intensity is equal to zero and standard deviation equals one.
#     """
#     image = image - np.mean(image)
#     image = image / np.std(image)
#     if mask:
#         image = image * mask
#     return image


# def _mask_circular(image_shape, sigma=5.0, *, radius=None):
#     """Make a circular mask with soft edges for image normalization.

#     Parameters
#     ----------
#     image_shape : tuple
#         Shape of the original image array
#     sigma : float, optional
#         Sigma value (in pixels) for gaussian blur function, by default 5.
#     radius : int, optional
#         Radius of circle, by default None which will create a circle that fills
#         90% of the smallest image dimension.

#     Returns
#     -------
#     ndarray
#         Circular mask with soft edges in array matching the input image_shape
#     """
#     if radius is None:
#         # leave at least a 5% gap on each edge
#         radius = 0.45 * min(image_shape)
#     center = np.divide(image_shape, 2)
#     rr, cc = skimage.draw.disk(center, radius=radius, shape=image_shape)
#     mask = np.zeros(image_shape)
#     mask[rr, cc] = 1.0
#     mask = ndi.gaussian_filter(mask, sigma=sigma)
#     return mask


# def _mask_rectangular(image_shape, sigma=5.0, *, start=None, extent=None):
#     """Make a rectangular mask with soft edges for image normalization.

#     Parameters
#     ----------
#     image_shape : tuple
#         Shape of the original image array
#     sigma : float, optional
#         Sigma value (in pixels) for gaussian blur function, by default 5.
#     start : tuple, optional
#         Origin point of the rectangle, e.g., ([plane,] row, column).
#         Default start is 5% of the total image width and height.
#     extent : int, optional
#         The extent (size) of the drawn rectangle.
#         E.g., ([num_planes,] num_rows, num_cols).
#         Default is for the rectangle to cover 95% of the image width & height.

#     Returns
#     -------
#     ndarray
#         Rectangular mask with soft edges in array matching input image_shape.
#     """
#     if extent is None:
#         # leave at least a 5% gap on each edge
#         start = np.round(np.array(image_shape) * 0.05)
#         extent = np.round(np.array(image_shape) * 0.90)
#     rr, cc = skimage.draw.rectangle(start, extent=extent, shape=image_shape)
#     mask = np.zeros(image_shape)
#     mask[rr.astype(int), cc.astype(int)] = 1.0
#     mask = ndi.gaussian_filter(mask, sigma=sigma)
#     return mask


# def _bandpass_mask(image_shape, outer_radius, inner_radius=0, sigma=5):
#     """Create a fourier bandpass mask.

#     Parameters
#     ----------
#     image_shape : tuple
#         Shape of the original image array
#     outer_radius : int
#         Outer radius for bandpass filter array.
#     inner_radius : int, optional
#         Inner radius for bandpass filter array, by default 0
#     sigma : int, optional
#         Sigma value for edge blending, by default 5 pixels.

#     Returns
#     -------
#     _bandpass_mask : ndarray
#         The bandpass image mask.
#     """
#     _bandpass_mask = np.zeros(image_shape)
#     center = np.divide(image_shape, 2)
#     inner_circle_rr, inner_circle_cc = skimage.draw.disk(
#         center, inner_radius, shape=image_shape
#     )
#     outer_circle_rr, outer_circle_cc = skimage.draw.disk(
#         center, outer_radius, shape=image_shape
#     )
#     _bandpass_mask[outer_circle_rr, outer_circle_cc] = 1.0
#     _bandpass_mask[inner_circle_rr, inner_circle_cc] = 0.0
#     _bandpass_mask = ndi.gaussian_filter(_bandpass_mask, sigma)
#     _bandpass_mask = np.array(_bandpass_mask)
#     # fourier space origin should be in the corner
#     _bandpass_mask = np.roll(
#         _bandpass_mask, (np.array(image_shape) / 2).astype(int), axis=(0, 1)
#     )
#     return _bandpass_mask