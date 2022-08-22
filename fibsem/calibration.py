
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
from fibsem.structures import (BeamSettings, MicroscopeSettings, MicroscopeState, Point,
                               ReferenceImages, BeamType, ImageSettings)


def correct_stage_drift(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
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
            ref_mask = create_lamella_mask(ref_image, settings.protocol["lamella"], factor = 4) # TODO: refactor, liftout specific
        else: 
            ref_mask = None

        # take new images
        # set new image settings (same as reference)
        settings.image = match_image_settings(
            ref_image, settings.image, beam_type=alignment[1]
        )
        new_image = acquire.new_image(microscope, settings.image)

        # crosscorrelation alignment
        ret = align_using_reference_images(
            microscope, settings, ref_lowres, new_image, ref_mask=ref_mask
        )

        if ret is False:
            break # cross correlation has failed...

    return ret

def align_using_reference_images(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    ref_image: AdornedImage,
    new_image: AdornedImage,
    ref_mask: np.ndarray = None
) -> bool:

    # get beam type
    ref_beam_type = BeamType[ref_image.metadata.acquisition.beam_type.upper()]
    new_beam_type = BeamType[new_image.metadata.acquisition.beam_type.upper()]

    logging.info(
        f"aligning {ref_beam_type.name} reference image to {new_beam_type.name}."
    )
    # lp_px = int(max(new_image.data.shape) * 0.66)
    # hp_px = int(max(new_image.data.shape) / 64)
    sigma = 6
    lp_px = int(max(new_image.data.shape) / 6)
    hp_px = int(max(new_image.data.shape) / 256)

    dx, dy, xcorr = shift_from_crosscorrelation(
        ref_image, new_image, lowpass=lp_px, highpass=hp_px, sigma=sigma, 
        use_rect_mask=True, ref_mask=ref_mask
    )

    from liftout import utils
    utils.plot_crosscorrelation(ref_image, new_image, dx, dy, xcorr)


    shift_within_tolerance = check_shift_within_tolerance(
        dx=dx, dy=dy, ref_image=ref_image, limit=0.5
    )

    if shift_within_tolerance:

        # move the stage
        movement.move_stage_relative_with_corrected_movement(microscope, 
            settings, 
            dx=dx, 
            dy=-dy, 
            beam_type=new_beam_type)

    return shift_within_tolerance

def shift_from_crosscorrelation(
    ref_image: AdornedImage,
    new_image: AdornedImage,
    lowpass: int = 128,
    highpass: int = 6,
    sigma: int = 6,
    use_rect_mask: bool = False,
    ref_mask: np.ndarray = None
) -> tuple[float, float, np.ndarray]:

    # get pixel_size
    pixelsize_x = new_image.metadata.binary_result.pixel_size.x
    pixelsize_y = new_image.metadata.binary_result.pixel_size.y

    # normalise both images
    ref_data_norm = normalise_image(ref_image)
    new_data_norm = normalise_image(new_image)

    # cross-correlate normalised images
    if use_rect_mask:
        rect_mask = _mask_rectangular(new_data_norm.shape)
        ref_data_norm = rect_mask * ref_data_norm
        new_data_norm = rect_mask * new_data_norm

    if ref_mask is not None:
        ref_data_norm = ref_mask * ref_data_norm # mask the reference

    # run crosscorrelation
    xcorr = crosscorrelation(
        ref_data_norm, new_data_norm, bp=True, lp=lowpass, hp=highpass, sigma=sigma
    )

    # calculate maximum crosscorrelation
    maxX, maxY = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    cen = np.asarray(xcorr.shape) / 2
    err = np.array(cen - [maxX, maxY], int)

    # calculate shift in metres
    x_shift = err[1] * pixelsize_x
    y_shift = err[0] * pixelsize_y # this could be the issue?
    
    logging.info(f"pixelsize: x: {pixelsize_x}, y: {pixelsize_y}")

    logging.info(f"cross-correlation:")
    logging.info(f"maxX: {maxX}, {maxY}, centre: {cen}")
    logging.info(f"x: {err[1]}px, y: {err[0]}px")
    logging.info(f"x: {x_shift:.2e}m, y: {y_shift:.2e} meters")

    # metres
    return x_shift, y_shift, xcorr

def crosscorrelation(img1: np.ndarray, img2: np.ndarray,  
    lp: int = 128, hp: int = 6, sigma: int = 6, bp: bool = False) -> np.ndarray:
    """Cross-correlate images (fourier convolution matching)

    Args:
        img1 (np.ndarray): reference_image
        img2 (np.ndarray): new image
        lp (int, optional): lowpass. Defaults to 128.
        hp (int, optional): highpass . Defaults to 6.
        sigma (int, optional): sigma (gaussian blur). Defaults to 6.
        bp (bool, optional): use a bandpass. Defaults to False.

    Returns:
        np.ndarray: crosscorrelation map
    """
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

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        # ax[0].imshow(fftpack.ifft2(img1ft).real)
        # ax[1].imshow(fftpack.ifft2(img2ft).real)
        # plt.show()

        xcorr = np.real(fftpack.fftshift(fftpack.ifft2(img1ft * np.conj(img2ft))))
    else: # TODO: why are these different...
        img1ft = fftpack.fft2(img1)
        img2ft = np.conj(fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
    
    return xcorr

# numpy version
def crosscorrelation_v2_np(img1: np.ndarray, img2: np.ndarray,  
    lp: int = 128, hp: int = 6, sigma: int = 6, bp: bool = False) -> np.ndarray:
    """Cross-correlate images (fourier convolution matching)

    Args:
        img1 (np.ndarray): reference_image
        img2 (np.ndarray): new image
        lp (int, optional): lowpass. Defaults to 128.
        hp (int, optional): highpass . Defaults to 6.
        sigma (int, optional): sigma (gaussian blur). Defaults to 6.
        bp (bool, optional): use a bandpass. Defaults to False.

    Returns:
        np.ndarray: crosscorrelation map
    """
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
        
        img1ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img1)))
        tmp = img1ft * np.conj(img1ft)
        img1ft = n_pixels * img1ft / np.sqrt(tmp.sum())
        
        img2ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        
        img2ft = n_pixels * img2ft / np.sqrt(tmp.sum())

        xcorr = np.real(np.fft.fftshift(np.fft.ifft2(img1ft * np.conj(img2ft))))
    else: # TODO: why are these different...
        img1ft = np.fft.fft2(img1)
        img2ft = np.conj(np.fft.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(np.fft.fftshift(np.fft.ifft2(img1ft * img2ft)))
    
    return xcorr



def match_image_settings(
    ref_image: AdornedImage,
    image_settings: ImageSettings,
    beam_type: BeamType = BeamType.ELECTRON,
) -> ImageSettings:
    """Generate matching image settings from an image."""
    image_settings.resolution = f"{ref_image.width}x{ref_image.height}"
    image_settings.dwell_time = ref_image.metadata.scan_settings.dwell_time
    image_settings.hfw = ref_image.width * ref_image.metadata.binary_result.pixel_size.x
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


def measure_brightness(img: AdornedImage) -> float:
    
    return np.mean(img.data)

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

# new masks below
def create_distance_map_px(w: int, h: int) -> np.ndarray:
    x = np.arange(0, w)
    y = np.arange(0, h)

    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(((w / 2) - X) ** 2 + ((h / 2) - Y) ** 2)

    return distance

def create_circle_mask(shape: tuple = (128, 128), radius: int = 32, sigma: int = 3) -> np.ndarray:
    """_summary_

    Args:
        shape (tuple, optional): _description_. Defaults to (128, 128).
        radius (int, optional): _description_. Defaults to 32.
        sigma (int, optional): _description_. Defaults to 3.

    Returns:
        np.ndarray: _description_
    """
    distance = create_distance_map_px(w = shape[1], h=shape[0])
    mask = distance <= radius

    if sigma:
        mask = ndi.filters.gaussian_filter(mask, sigma=sigma)

    return mask


def create_bandpass_mask(shape: tuple = (256, 256), lp: int = 32, hp: int = 2, sigma: int = 3) -> np.ndarray:
    """_summary_

    Args:
        shape (tuple, optional): _description_. Defaults to (256, 256).
        lp (int, optional): _description_. Defaults to 32.
        hp (int, optional): _description_. Defaults to 2.
        sigma (int, optional): _description_. Defaults to 3.

    Returns:
        np.ndarray: _description_
    """
    distance = create_distance_map_px(w = shape[1], h=shape[0])
    
    lowpass = distance <= lp
    highpass = distance >= hp

    mask = lowpass * highpass
    
    if sigma:
        mask = ndi.filters.gaussian_filter(mask, sigma=sigma)

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


def create_rect_mask(img: np.ndarray, w: int, h: int, sigma: int = 0, pt: Point= None) -> np.ndarray:
    """Create a rectangular mask at centred at the desired point.

    Args:
        img (np.ndarray): Image to be masked
        w (int): mask width
        h (int): mask height
        sigma (int, optional): gaussian blur to apply to mask (softness). Defaults to 0.
        pt (Point): Mask centre point. Defaults to None

    Returns:
        np.ndarray: mask
    """
    # centre mask point
    if pt is None:
        pt = Point(img.data.shape[1]//2, img.data.shape[0]//2)

    mask = np.zeros_like(img)

    y_min, y_max = int(np.clip(pt.y-h/2, 0, img.shape[1])), int(np.clip(pt.y+h/2, 0, img.shape[1]))
    x_min, x_max = int(np.clip(pt.x-w/2, 0, img.shape[1])), int(np.clip(pt.x+w/2, 0, img.shape[1]))

    mask[y_min:y_max, x_min:x_max] = 1

    if sigma:
        mask = ndi.filters.gaussian_filter(mask, sigma=sigma)

    return mask 

def create_lamella_mask(img: AdornedImage, protocol: dict, scale: int = 2, circ: bool = False, pt: Point = None, use_trench_height: bool = False) -> np.ndarray:
    """Create a mask based on the size of the lamella

    Args:
        img (AdornedImage): reference image
        settings (dict): protocol dictionary
        scale (int, optional): mask size will be multipled by this scale . Defaults to 2.
        circ (bool, optional): use a circular mask. Defaults to False.
        pt (Point, optional): original point for the mask. Defaults to None.
        use_trench_height (bool, optional): use the trench height to calculate the mask size instead of lamella height). Defaults to False.

    Returns:
        np.ndarray: mask
    """

    # get the size of the lamella in pixels
    lamella_height_px, lamella_width_px = get_lamella_size_in_pixels(img, protocol, use_trench_height)

    if circ:
        mask = circ_mask(
            size=(img.data.shape[1], img.data.shape[0]), 
            radius=max(lamella_height_px, lamella_width_px) * scale , sigma=12
        )
    else:
        mask = create_rect_mask(img=img.data,  
            pt=pt,
            w=int(lamella_width_px * scale), 
            h=int(lamella_height_px * scale), sigma=3)

    return mask




def get_lamella_size_in_pixels(img: AdornedImage, protocol: dict, use_trench_height: bool = False) -> tuple[int]:
    """Get the relative size of the lamella in pixels based on the hfw of the image.

    Args:
        img (AdornedImage): reference image
        settings (dict): protocol dictionary
        use_lamella_height (bool, optional): get the height of the lamella (True), or Trench. Defaults to False.

    Returns:
        tuple[int]: _description_
    """
    # get real size from protocol
    lamella_width = protocol["lamella_width"]
    lamella_height = protocol["lamella_height"]
        
    total_height = lamella_height
    if use_trench_height:
        trench_height = protocol["protocol_stages"][0]["trench_height"]
        total_height += 2 * trench_height

    # convert to px
    pixelsize = img.metadata.binary_result.pixel_size.x
    vfw = img.height * pixelsize
    hfw = img.width * pixelsize
    
    lamella_height_px = int((total_height / vfw) * img.height) 
    lamella_width_px = int((lamella_width / hfw) * img.width) 

    return (lamella_height_px, lamella_width_px)




## AUTO ALIGNMENTS


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
        ),
        # ion beam settings
        ib_settings=BeamSettings(
            beam_type=BeamType.ION,
            working_distance=microscope.beams.ion_beam.working_distance.value,
            beam_current=microscope.beams.ion_beam.beam_current.value,
            hfw=microscope.beams.ion_beam.horizontal_field_width.value,
            resolution=microscope.beams.ion_beam.scanning.resolution.value,
            dwell_time=microscope.beams.ion_beam.scanning.dwell_time.value,
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
    # microscope.beams.electron_beam.stigmator.value = (
    #     microscope_state.eb_settings.stigmation
    # )

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
    # microscope.beams.ion_beam.stigmator.value = microscope_state.ib_settings.stigmation

    logging.info(f"microscope state restored")
    return

