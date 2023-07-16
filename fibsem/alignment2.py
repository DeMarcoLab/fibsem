import logging

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (
    AdornedImage,
    MoveSettings,
    Rectangle,
    StagePosition,
)

from fibsem import acquire, calibration, movement, utils, validation
from fibsem.imaging import masks
from fibsem.imaging import utils as image_utils
from fibsem.structures import (
    BeamType,
    ImageSettings,
    MicroscopeSettings,
    ReferenceImages,
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
    ref_image: AdornedImage,
    reduced_area: Rectangle = None,
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
    xcorr_limit: tuple = None,
    constrain_vertical: bool = False,
    beam_shift: bool = False
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

        ref_mask = masks.create_circle_mask(ref_image.data.shape, radius=512)
        # if use_ref_mask:
        #     pass
        #     # ref_mask = masks.create_lamella_mask(
        #     #     ref_image,
        #     #     settings.protocol["lamella"],
        #     #     scale=mask_scale,
        #     #     use_trench_height=True,
        #     # )  # TODO: refactor, liftout specific
        # else:
        #     ref_mask = None

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
            beam_shift=beam_shift
        )

        if ret is False:
            break  # cross correlation has failed...

    return ret


def eucentric_correct_stage_drift(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    reference_images: ReferenceImages,
    rotate: bool = False,
    xcorr_limit: tuple = None,
) -> bool:
    """Eucentric correction of the stage drift by crosscorrelating low-res and high-res reference images"""

    if xcorr_limit is None:
        xcorr_limit = (None, None, None)#, None)

    # target images
    targets = (BeamType.ELECTRON, BeamType.ELECTRON, BeamType.ION)#, BeamType.ELECTRON, BeamType.ION)
    eucentric_move = (False, False, True) 

    # lp, hp, sigma
    params = [
        (256, 12, 6),
        (128, 12, 6),
        (128, 8, 6),
        # (128, 8, 6),
    ]

    if rotate:
        # rotate references
        # align ref ib -> new eb
        # eucentric align ref eb -> new ib
        ref_order = (reference_images.low_res_ib, reference_images.high_res_ib, reference_images.high_res_eb)#, , reference_images.high_res_eb)
        ref_order = [image_utils.rotate_image(ref) for ref in ref_order]
    else:
        # not rotated
        # align ref eb -> new eb
        # eucentric align ref ib -> new ib
        ref_order = (reference_images.low_res_eb, reference_images.low_res_ib, reference_images.high_res_eb, reference_images.high_res_ib)
        # TODO: this is wrong now

    # align lowres, then highres
    for i, (ref_image, target, euc_move) in enumerate(zip(ref_order, targets, eucentric_move)):

        ref_mask = masks.create_circle_mask(ref_image.data.shape, radius=512)

        # take new images
        # set new image settings (same as reference)
        settings.image = utils.match_image_settings(
            ref_image, settings.image, beam_type=target
        )
        new_image = acquire.new_image(microscope, settings.image)

        # crosscorrelation alignment
        align_using_reference_images(
            microscope,
            settings,
            ref_image,
            new_image,
            ref_mask=ref_mask,
            xcorr_limit=xcorr_limit[i],
            constrain_vertical=euc_move,
            lp_px=params[i][0],
            hp_px=params[i][1],
            sigma=params[i][2],
        )

def align_using_reference_images(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    ref_image: AdornedImage,
    new_image: AdornedImage,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None,
    constrain_vertical: bool = False,
    beam_shift: bool = False,
    lp_px: int  = 128,  # MAGIC_NUMBER
    hp_px: int = 8,  
    sigma: int = 6,
) -> bool:
    """Align new image to reference image using crosscorrelation
    
    Args:
        microscope (SdbMicroscopeClient): microscope client
        settings (MicroscopeSettings): microscope settings
        ref_image (AdornedImage): reference image
        new_image (AdornedImage): new image
        ref_mask (np.ndarray, optional): reference mask. Defaults to None.
        xcorr_limit (int, optional): crosscorrelation limit. Defaults to None.
        constrain_vertical (bool, optional): constrain vertical movement. Defaults to False.
        beam_shift (bool, optional): use beam shift. Defaults to False.
        lp_px (int, optional): lowpass filter size. Defaults to 128.
        hp_px (int, optional): highpass filter size. Defaults to 8.
        sigma (int, optional): gaussian filter sigma. Defaults to 6.
    Returns:
        bool: True if alignment was successful, False otherwise
    """

    # get beam type
    ref_beam_type = BeamType[ref_image.metadata.acquisition.beam_type.upper()]
    new_beam_type = BeamType[new_image.metadata.acquisition.beam_type.upper()]

    logging.info(
        f"aligning {ref_beam_type.name} reference image to {new_beam_type.name}."
    )

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
            movement.move_stage_eucentric_correction(microscope, settings=settings, dy=dy) # FLAG_TEST
        else:
            # TODO: does rotating the reference need to be taken into account? i believe so
            # move the stage
            if beam_shift:
                microscope.beams.ion_beam.beam_shift.value += (-dx, dy)
            else:
                movement.move_stage_relative_with_corrected_movement(
                    microscope,
                    settings,
                    dx=dx,
                    dy=-dy,
                    beam_type=new_beam_type,
                )

    return shift_within_tolerance


def shift_from_crosscorrelation(
    ref_image: AdornedImage,
    new_image: AdornedImage,
    lowpass: int = 128,
    highpass: int = 6,
    sigma: int = 6,
    use_rect_mask: bool = False,
    ref_mask: np.ndarray = None,
    xcorr_limit: int = None,
) -> tuple[float, float, np.ndarray]:

    # get pixel_size
    pixelsize_x = new_image.metadata.binary_result.pixel_size.x
    pixelsize_y = new_image.metadata.binary_result.pixel_size.y

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
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape) # TODO: backwards
    cen = np.asarray(xcorr.shape) / 2
    err = np.array(cen - [maxX, maxY], int)

    # calculate shift in metres
    x_shift = err[1] * pixelsize_x
    y_shift = err[0] * pixelsize_y  # this could be the issue?

    ########################################

    # plot ref, new and xcorr with matplotlib with titles and midpoint
    # TODO: add the filtering to the image
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    ax1.imshow(ref_data_norm, cmap="gray")
    ax1.set_title(f"ref (lp: {lowpass}, hp: {highpass}, sigma: {sigma})")
    ax1.plot(cen[1], cen[0], "w+", ms=20)
    ax2.imshow(new_data_norm, cmap="gray")
    ax2.set_title("new")
    ax2.plot(cen[1], cen[0], "w+", ms=20)
    ax3.imshow(xcorr, cmap="turbo")
    ax3.set_title(f"xcorr (shift: {x_shift:.2e}, {y_shift:.2e})")
    ax3.plot(maxY, maxX, "w+", ms=20)
  
    from fibsem import utils as f_utils
    f_utils.log_current_figure(fig, "crosscorrelation")

    ###############

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
        bandpass (np.ndarray): bandpass filter

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