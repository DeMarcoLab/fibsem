
import logging
from datetime import datetime

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                         StagePosition)
from PIL import Image, ImageDraw
from scipy import fftpack

from fibsem import acquire, movement, utils
from fibsem.structures import (BeamSettings, MicroscopeState, Point,
                               ReferenceImages, BeamType, ImageSettings)


def correct_stage_drift(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    reference_images: ReferenceImages,
    alignment: tuple(BeamType) = (BeamType.ELECTRON, BeamType.ELECTRON),
    rotate: bool = False,
    use_ref_mask: bool = False,
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

    # rotate reference
    if rotate:
        ref_lowres = rotate_AdornedImage(ref_lowres)
        ref_highres = rotate_AdornedImage(ref_highres)

    # align lowres, then highres
    for ref_image in [ref_lowres, ref_highres]:

        if use_ref_mask:
            ref_mask = create_lamella_mask(ref_image, settings, factor = 4)
        else: 
            ref_mask = None

        # take new images
        # set new image settings (same as reference)
        image_settings = match_image_settings(
            ref_image, image_settings, beam_type=alignment[1]
        )
        new_image = acquire.new_image(microscope, image_settings)

        # crosscorrelation alignment
        ret = align_using_reference_images(
            microscope, settings, ref_lowres, new_image, ref_mask=ref_mask
        )

        if ret is False:
            break # cross correlation has failed...

    return ret

def align_using_reference_images(
    microscope: SdbMicroscopeClient,
    settings: dict,
    ref_image: AdornedImage,
    new_image: AdornedImage,
    ref_mask: np.ndarray = None
) -> bool:

    # get beam type
    ref_beam_type = BeamType(ref_image.metadata.acquisition.beam_type.upper())
    new_beam_type = BeamType(new_image.metadata.acquisition.beam_type.upper())

    logging.info(
        f"aligning {ref_beam_type.name} reference image to {new_beam_type.name}."
    )
    lp_px = int(max(new_image.data.shape) * 0.66)
    hp_px = int(max(new_image.data.shape) / 64)
    sigma = 6

    dx, dy, _ = shift_from_crosscorrelation(
        new_image, ref_image, lowpass=lp_px, highpass=hp_px, sigma=sigma, 
        use_rect_mask=True, ref_mask=ref_mask
    )

    shift_within_tolerance = check_shift_within_tolerance(
        dx=dx, dy=dy, ref_image=ref_image, limit=0.25
    )

    if shift_within_tolerance:

        # move the stage
        movement.move_stage_relative_with_corrected_movement(microscope, 
            settings, 
            dx=-dx, 
            dy=dy, 
            beam_type=new_beam_type)

    return shift_within_tolerance

def shift_from_crosscorrelation(
    img1: AdornedImage,
    img2: AdornedImage,
    lowpass: int = 128,
    highpass: int = 6,
    sigma: int = 6,
    use_rect_mask: bool = False,
    ref_mask: np.ndarray = None
) -> tuple[float, float, np.ndarray]:

    # get pixel_size
    pixelsize_x = img2.metadata.binary_result.pixel_size.x
    pixelsize_y = img2.metadata.binary_result.pixel_size.y

    # normalise both images
    img1_data_norm = normalise_image(img1)
    img2_data_norm = normalise_image(img2)

    # cross-correlate normalised images
    if use_rect_mask:
        rect_mask = _mask_rectangular(img2_data_norm.shape)
        img1_data_norm = rect_mask * img1_data_norm
        img2_data_norm = rect_mask * img2_data_norm

    if ref_mask is not None:
        img1_data_norm = ref_mask * img1_data_norm # mask the reference

    # run crosscorrelation
    xcorr = crosscorrelation(
        img1_data_norm, img2_data_norm, bp=True, lp=lowpass, hp=highpass, sigma=sigma
    )

    # calculate maximum crosscorrelation
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    cen = np.asarray(xcorr.shape) / 2
    err = np.array(cen - [maxX, maxY], int)

    # calculate shift in metres
    x_shift = err[1] * pixelsize_x
    y_shift = err[0] * pixelsize_y
    
    logging.info(f"cross-correlation:")
    logging.info(f"maxX: {maxX}, {maxY}, centre: {cen}")
    logging.info(f"x: {err[1]}px, y: {err[0]}px")
    logging.info(f"x: {x_shift:.2e}m, y: {y_shift:.2e} meters")

    # metres
    return x_shift, y_shift, xcorr

def crosscorrelation(img1: np.ndarray, img2: np.ndarray,  
    lp: int = 128, hp: int = 6, sigma: int = 6, bp: bool = False) -> np.ndarray:
    
    if img1.shape != img2.shape:
        err = f"Image 1 {img1.shape} and Image 2 {img2.shape} need to have the same shape"
        logging.error(err)
        raise ValueError(err)

    if bp: 
        bandpass = bandpass_mask(
            size=(img1.shape[1], img1.shape[0]), 
            lp=lp, hp=hp, sigma=sigma
        )
        n_pixels = img1.shape[0] * img1.shape[1]
        
        img1ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img1)))
        tmp = img1ft * np.conj(img1ft)
        img1ft = n_pixels * img1ft / np.sqrt(tmp.sum())
        
        img2ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        
        img2ft = n_pixels * img2ft / np.sqrt(tmp.sum())

        xcorr = np.real(fftpack.fftshift(fftpack.ifft2(img1ft * np.conj(img2ft))))
    else: # TODO: why are these different...
        img1ft = fftpack.fft2(img1)
        img2ft = np.conj(fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
    
    return xcorr



### UTILS
def pixel_to_realspace_coordinate(coord: list, image: AdornedImage) -> list:
    """Convert pixel image coordinate to real space coordinate.

    This conversion deliberately ignores the nominal pixel size in y,
    as this can lead to inaccuracies if the sample is not flat in y.

    Parameters
    ----------
    coord : listlike, float
        In x, y format & pixel units. Origin is at the top left.

    image : AdornedImage
        Image the coordinate came from.

        # do we have a sample image somewhere?
    Returns
    -------
    realspace_coord
        xy coordinate in real space. Origin is at the image center.
        Output is in (x, y) format.
    """
    coord = np.array(coord).astype(np.float64)
    if len(image.data.shape) > 2:
        y_shape, x_shape = image.data.shape[0:2]
    else:
        y_shape, x_shape = image.data.shape

    pixelsize_x = image.metadata.binary_result.pixel_size.x
    # deliberately don't use the y pixel size, any tilt will throw this off
    coord[1] = y_shape - coord[1]  # flip y-axis for relative coordinate system
    # reset origin to center
    coord -= np.array([x_shape / 2, y_shape / 2]).astype(np.int32)
    realspace_coord = list(np.array(coord) * pixelsize_x)  # to real space
    return realspace_coord # TODO: convert to use Point struct


def match_image_settings(
    ref_image: AdornedImage,
    image_settings: ImageSettings,
    beam_type: BeamType = BeamType.ELECTRON,
) -> ImageSettings:
    """Generate matching image settings from an image."""
    image_settings.hfw = f"{ref_image.height}x{ref_image.width}"
    image_settings.dwell_time = ref_image.metadata.scan_settings.dwell_time
    image_settings.beam_type = beam_type
    image_settings.save = True
    image_settings.label = utils.current_timestamp()

    return image_settings

def check_shift_within_tolerance(
    dx: float, dy: float, ref_image: AdornedImage, limit: float = 0.25
) -> bool:
    """Check if required shift is wihtin safety limit"""
    # check if the cross correlation movement is within the safety limit

    pixelsize_x = ref_image.metadata.binary_result.pixel_size.x
    X_THRESHOLD = limit * pixelsize_x * ref_image.width
    Y_THRESHOLD = limit * pixelsize_x * ref_image.height

    return abs(dx) < X_THRESHOLD and abs(dy) < Y_THRESHOLD


def measure_brightness(img: AdornedImage, crop_size: int = None) -> float:
    cx, cy = img.data.shape[1] //2, img.data.shape[0] // 2

    if crop_size is not None:
        img = img.data[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]
    else:
        img = img.data

    return np.mean(img), img

def rotate_AdornedImage(image: AdornedImage):
    """Rotate the AdornedImage 180 degrees."""
    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = AdornedImage(data=data, metadata=image.metadata)
    return reference


def normalise_image(img: AdornedImage) -> np.ndarray:
    """Normalise the image"""
    return (img.data - np.mean(img.data)) / np.std(img.data)

def cosine_stretch(img: AdornedImage, tilt_degrees: float):
    """Apply a cosine stretch to an image based on the relative tilt. 

    This is required when aligning images with different tilts to ensure features are the same size.

    Args:
        img (AdornedImage): _description_
        tilt_degrees (float): _description_

    Returns:
        _type_: _description_
    """
    # TODO: do smaller version for negative tilt??

    tilt = np.deg2rad(tilt_degrees)

    shape = int(img.data.shape[0] / np.cos(tilt)), int(img.data.shape[1] / np.cos(tilt))

    # cosine stretch
    print(f"tilt: {tilt}, cosine: {np.cos(tilt)}")
    print("initial ", img.data.shape, "new shape: ", shape)
    # larger

    from PIL import Image

    resized_img = np.asarray(Image.fromarray(img.data).resize(size=(shape[1], shape[0])))
    # crop centre out?
    c = Point(resized_img.shape[1]//2, resized_img.shape[0]//2)
    dy, dx = img.data.shape[0]//2, img.data.shape[1]//2
    scaled_img = resized_img[c.y-dy:c.y+dy, c.x-dx:c.x+dx]

    # TODO: smaller?

    print("rescaled shape:", scaled_img.shape)

    return AdornedImage(data=scaled_img, metadata=img.metadata)




### MASKING
def bandpass_mask(size=(128, 128), lp=32, hp=2, sigma=3):
    x = size[0]
    y = size[1]
    lowpass = circ_mask(size=(x, y), radius=lp, sigma=0)
    hpass_tmp = circ_mask(size=(x, y), radius=hp, sigma=0)
    highpass = -1 * (hpass_tmp - 1)
    tmp = lowpass * highpass
    if sigma > 0:
        bandpass = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        bandpass = tmp
    return bandpass


def circ_mask(size=(128, 128), radius=32, sigma=3):
    x = size[0]
    y = size[1]
    img = Image.new("I", size)
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        (x / 2 - radius, y / 2 - radius, x / 2 + radius, y / 2 + radius),
        fill="white",
        outline="white",
    )
    tmp = np.array(img, float) / 255
    if sigma > 0:
        mask = ndi.filters.gaussian_filter(tmp, sigma=sigma)
    else:
        mask = tmp
    return mask


# FROM AUTOLAMELLA
def _mask_rectangular(image_shape, sigma=5.0, *, start=None, extent=None):
    """Make a rectangular mask with soft edges for image normalization.

    Parameters
    ----------
    image_shape : tuple
        Shape of the original image array
    sigma : float, optional
        Sigma value (in pixels) for gaussian blur function, by default 5.
    start : tuple, optional
        Origin point of the rectangle, e.g., ([plane,] row, column).
        Default start is 5% of the total image width and height.
    extent : int, optional
        The extent (size) of the drawn rectangle.
        E.g., ([num_planes,] num_rows, num_cols).
        Default is for the rectangle to cover 95% of the image width & height.

    Returns
    -------
    ndarray
        Rectangular mask with soft edges in array matching input image_shape.
    """
    import skimage

    if extent is None:
        # leave at least a 5% gap on each edge
        start = np.round(np.array(image_shape) * 0.05)
        extent = np.round(np.array(image_shape) * 0.90)
    rr, cc = skimage.draw.rectangle(start, extent=extent, shape=image_shape)
    mask = np.zeros(image_shape)
    mask[rr.astype(int), cc.astype(int)] = 1.0
    mask = ndi.gaussian_filter(mask, sigma=sigma)
    return mask


def apply_image_mask(img: AdornedImage, mask: np.ndarray) -> np.ndarray:

    return normalise_image(img) * mask


def create_rect_mask(img: np.ndarray, pt: Point, w: int, h: int, sigma: int = 0) -> np.ndarray:
    """Create a rectangular mask at centred at the desired point.

    Args:
        img (np.ndarray): Image to be masked
        pt (Point): Mask centre point
        w (int): mask width
        h (int): mask height
        sigma (int, optional): gaussian blur to apply to mask (softness). Defaults to 0.

    Returns:
        np.ndarray: mask
    """
    mask = np.zeros_like(img)

    y_min, y_max = int(np.clip(pt.y-h/2, 0, img.shape[1])), int(np.clip(pt.y+h/2, 0, img.shape[1]))
    x_min, x_max = int(np.clip(pt.x-w/2, 0, img.shape[1])), int(np.clip(pt.x+w/2, 0, img.shape[1]))

    mask[y_min:y_max, x_min:x_max] = 1

    if sigma:
        mask = ndi.filters.gaussian_filter(mask, sigma=sigma)

    return mask 

def create_lamella_mask(img: AdornedImage, settings: dict, factor: int = 2, circ: bool = True, pt: Point = None, use_lamella_height: bool = False) -> np.ndarray:
    """Create a mask based on the size of the lamella

    Args:
        img (AdornedImage): _description_
        settings (dict): _description_
        factor (int, optional): _description_. Defaults to 2.
        circ (bool, optional): _description_. Defaults to True.
        pt (Point, optional): _description_. Defaults to None.
        use_lamella_height (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: _description_
    """

    # centre mask point
    if pt is None:
        pt = Point(img.data.shape[1]//2, img.data.shape[0]//2)

    # get real size from protocol
    lamella_width = settings["protocol"]["lamella"]["lamella_width"]
    if use_lamella_height:
        lamella_height = settings["protocol"]["lamella"]["lamella_height"]
    else:
        lamella_height = settings["protocol"]["lamella"]["protocol_stages"][0]["trench_height"]
        
    # convert to px
    pixelsize = img.metadata.binary_result.pixel_size.x
    vfw = img.height * pixelsize
    hfw = img.width * pixelsize
    
    lamella_height_px = int((lamella_height / vfw) * img.height) 
    lamella_width_px = int((lamella_width / hfw) * img.width) 

    if circ:
        mask = circ_mask(
            size=(img.data.shape[1], img.data.shape[0]), 
            radius=max(lamella_height_px, lamella_width_px) * factor , sigma=12
        )
    else:
        mask = create_rect_mask(img=img.data,  
            pt=pt,
            w=int(lamella_width_px * factor), 
            h=int(lamella_height_px * factor), sigma=3)

    return mask

## AUTO ALIGNMENTS
def beam_shift_alignment(
    microscope: SdbMicroscopeClient,
    image_settings: ImageSettings,
    ref_image: AdornedImage,
    reduced_area,
):
    """Align the images by adjusting the beam shift, instead of moving the stage
            (increased precision, lower range)

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client
        image_settings (acquire.ImageSettings): settings for taking image
        ref_image (AdornedImage): reference image to align to
        reduced_area (Rectangle): The reduced area to image with.
    """

    # # align using cross correlation
    img1 = ref_image
    img2 = acquire.new_image(
        microscope, settings=image_settings, reduced_area=reduced_area
    )
    dx, dy, _ = shift_from_crosscorrelation(
        img1, img2, lowpass=50, highpass=4, sigma=5, use_rect_mask=True
    )

    # adjust beamshift
    microscope.beams.ion_beam.beam_shift.value += (-dx, dy)

def auto_link_stage(microscope: SdbMicroscopeClient, hfw: float = 150e-6) -> None:
    """Automatically focus and link sample stage z-height.

    Notes:
        - Focusing determines the working distance (focal distance) of the beam
        - Relinking is required whenever there is a significant change in vertical distance, i.e. moving
          from the landing grid to the sample grid.
        - Linking determines the specimen coordinate system, as it is defined as the relative dimensions of the top of stage
          to the instruments.
    """

    microscope.imaging.set_active_view(1)
    original_hfw = microscope.beams.electron_beam.horizontal_field_width.value
    microscope.beams.electron_beam.horizontal_field_width.value = hfw
    acquire.autocontrast(microscope, beam_type=BeamType.ELECTRON)
    microscope.auto_functions.run_auto_focus()
    microscope.specimen.stage.link()
    # NOTE: replace with auto_focus_and_link if performance of focus is poor
    # # Restore original settings
    microscope.beams.electron_beam.horizontal_field_width.value = original_hfw



def automatic_eucentric_correction(
    microscope: SdbMicroscopeClient,
    settings: dict,
    image_settings: ImageSettings,
    eucentric_height: float = 3.9e-3,
):
    """Automatic procedure to reset to the eucentric position

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope client connection
        settings (dict): configuration dictionary
        image_settings (ImageSettings): imaging settings
        eucentric_height (float, optional): manually calibrated eucentric height. Defaults to 3.9e-3.
    """

    # autofocus in eb
    auto_link_stage(microscope)

    # move stage to z=3.9
    microscope.specimen.stage.set_default_coordinate_system()

    # turn on z-y linked movement # NB: cant do this through API
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    eucentric_position = StagePosition(z=eucentric_height)
    movement.safe_absolute_stage_movement(microscope, eucentric_position)

    # retake images to check
    acquire.take_reference_images(microscope, image_settings)


def automatic_eucentric_correction_v2(
    microscope: SdbMicroscopeClient, settings: dict, image_settings: ImageSettings
) -> None:

    # assume the feature of interest is on the image centre.
    
    # TODO: get user to manually centre?
        
    # iterative eucentric alignment

    hfw = 900e-6
    tolerance = 5.0e-6
    iteration = 0
    
    while True:

        # take low res reference image
        image_settings.hfw = hfw
        ref_eb, ref_ib = acquire.take_reference_images(microscope, image_settings)

        # calculate cross correlation...
        # x = horizontal, y = vertical

        # THESE ARE AT DIFFERENCE ANGLES??


        # align using cross correlation
        dx, dy, _ = shift_from_crosscorrelation(
            ref_eb, ref_ib, use_rect_mask=True
        )

        # stop if both are within tolernace
        if dy <= tolerance:
            break

        # move z??
        movement.move_stage_eucentric_correction(
            microscope, settings, dy=dy, beam_type=BeamType.ION
        )

        # align eb (cross correlate) back to original ref (move eb back to centre)
        image_settings.beam_type = BeamType.ELECTRON
        new_eb = acquire.new_image(microscope, image_settings, reduced_area=None)
        dx, dy, _ = shift_from_crosscorrelation(
            ref_eb, new_eb, lowpass=128, highpass=6, sigma=6, use_rect_mask=True
        )

        # move feature back to centre of eb
        movement.move_stage_relative_with_corrected_movement(
            microscope =microscope,
            settings=settings,
            dx=dx, dy=dy, beam_type=BeamType.ELECTRON)

        # repeat
        hfw = hfw / 2

        # increase count
        iteration += 1

        if iteration == 5:
            # unable to align within given iterations
            break

    # TODO: do we want to align in x too?

    return



# STATE MANAGEMENT

def get_raw_stage_position(microscope: SdbMicroscopeClient) -> StagePosition:
    """Get the current stage position in raw coordinate system, and switch back to specimen"""
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)
    stage_position = microscope.specimen.stage.current_position
    microscope.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)

    return stage_position

def get_current_microscope_state(
    microscope: SdbMicroscopeClient
) -> MicroscopeState:
    """Get the current microscope state v2 """

    current_microscope_state = MicroscopeState(
        timestamp=datetime.timestamp(datetime.now()),
        # get absolute stage coordinates (RAW)
        absolute_position=get_raw_stage_position(microscope),
        # electron beam settings
        eb_settings=BeamSettings(
            beam_type=BeamType.ELECTRON,
            working_distance=microscope.beams.electron_beam.working_distance.value,
            beam_current=microscope.beams.electron_beam.beam_current.value,
            hfw=microscope.beams.electron_beam.horizontal_field_width.value,
            resolution=microscope.beams.electron_beam.scanning.resolution.value,
            dwell_time=microscope.beams.electron_beam.scanning.dwell_time.value,
            stigmation=microscope.beams.electron_beam.stigmator.value,
        ),
        # ion beam settings
        ib_settings=BeamSettings(
            beam_type=BeamType.ION,
            working_distance=microscope.beams.ion_beam.working_distance.value,
            beam_current=microscope.beams.ion_beam.beam_current.value,
            hfw=microscope.beams.ion_beam.horizontal_field_width.value,
            resolution=microscope.beams.ion_beam.scanning.resolution.value,
            dwell_time=microscope.beams.ion_beam.scanning.dwell_time.value,
            stigmation=microscope.beams.ion_beam.stigmator.value,
        ),
    )

    return current_microscope_state


def set_microscope_state(microscope: SdbMicroscopeClient, microscope_state: MicroscopeState):
    """Reset the microscope state to the provided state"""

    logging.info(
        f"restoring microscope state..."
    )

    # move to position
    movement.safe_absolute_stage_movement(
        microscope=microscope, stage_position=microscope_state.absolute_position
    )

    # restore electron beam
    logging.info(f"restoring electron beam settings...")
    microscope.beams.electron_beam.working_distance.value = (
        microscope_state.eb_settings.working_distance
    )
    microscope.beams.electron_beam.beam_current.value = (
        microscope_state.eb_settings.beam_current
    )
    microscope.beams.electron_beam.horizontal_field_width.value = (
        microscope_state.eb_settings.hfw
    )
    microscope.beams.electron_beam.scanning.resolution.value = (
        microscope_state.eb_settings.resolution
    )
    microscope.beams.electron_beam.scanning.dwell_time.value = (
        microscope_state.eb_settings.dwell_time
    )
    microscope.beams.electron_beam.stigmator.value = (
        microscope_state.eb_settings.stigmation
    )

    # restore ion beam
    logging.info(f"restoring ion beam settings...")
    microscope.beams.ion_beam.working_distance.value = (
        microscope_state.ib_settings.working_distance
    )
    microscope.beams.ion_beam.beam_current.value = (
        microscope_state.ib_settings.beam_current
    )
    microscope.beams.ion_beam.horizontal_field_width.value = (
        microscope_state.ib_settings.hfw
    )
    microscope.beams.ion_beam.scanning.resolution.value = (
        microscope_state.ib_settings.resolution
    )
    microscope.beams.ion_beam.scanning.dwell_time.value = (
        microscope_state.ib_settings.dwell_time
    )
    microscope.beams.ion_beam.stigmator.value = microscope_state.ib_settings.stigmation

    logging.info(f"microscope state restored")
    return

