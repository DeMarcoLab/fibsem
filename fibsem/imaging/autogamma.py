import logging
import numpy as np

from skimage import exposure
from fibsem.structures import FibsemImage

def auto_gamma(
    image: FibsemImage,
    min_gamma: float = 0.15,
    max_gamma: float = 1.8,
    scale_factor: float = 0.01,
    gamma_threshold: int = 45,
    method: str = "autogamma",
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

    if method == "autogamma":
        std = np.std(image.data)  # unused variable?
        mean = np.mean(image.data)
        diff = mean - 255 / 2.0
        gam = np.clip(
            min_gamma, 1 + diff * scale_factor, max_gamma
        )
        if abs(diff) < gamma_threshold:
            gam = 1.0
        if image.metadata is not None:
            logging.debug(
                f"AUTO_GAMMA | {image.metadata.image_settings.beam_type} | {diff:.3f} | {gam:.3f}"
            )
        image_data = exposure.adjust_gamma(image.data, gam)

        image = FibsemImage(data=image_data, metadata=image.metadata)
    
    if method == "autoclahe":
        image = apply_clahe(image)

    return image



def apply_clahe(
    image: FibsemImage,
    which_package: str = "skimage",
    clip_limit_cv2: float = 15,
    tile_grid_size: int = 8,
    clip_limit_skimage: float = 0.02,
    kernel_size = None
) -> FibsemImage:
    """
    Applies Contrast Limited Adaptive Histogram Equalisation correction to the input `FibsemImage`.
    image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these
    blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region
    (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied.
    If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and
    distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts
    in tile borders, bilinear interpolation is applied.

    In OpenCV tileGridSize (tile_grid_size) is by default 8x8, clipLimit (clip_limit_cv2) is by default 40

    In skimage kernel_size (int or array_like), is optional. It defines the shape of contextual regions used in the
    algorithm. By default, kernel_size is 1/8 of image height by 1/8 of its width.
    clip_limit float, optional. Clipping limit, normalized between 0 and 1 (higher values give more contrast).

    Args:
        image (FibsemImage): The input `FibsemImage` to apply gamma correction to.

        type (str) Either "skimage" or "OpenCV" to apply the filter from the corresponding library

        clip_limit_cv2 (float): used if which_package=="OpenCV". Defaults to 15. (by default 40 in OpenCV)
        tile_grid_size (int): used if which_package=='OpenCV'. Defaults to 8x8 pixels.

        clip_limit_skimage (float): used if which_package=="skimage". Defaults to 0.01. Clipping limit, normalised between 0 and 1 (higher values give more contrast).
        tile_grid_size (int): used if which_package=='skimage'. if None, defaults to kernel_size is 1/8 of image height by 1/8 of its width.

    Returns:
        A new `FibsemImage` object containing the clahe-enhanced image data, with the same metadata
        as the input image.

    Notes:
        - This function applies gamma correction to the input image using either the `cv2.createCLAHE` or `skimage.exposure.equalize_adapthist` function
        - The `FibsemImage` object in the returned list contains the clahe-enhanced image data as a
          numpy array, as well as other image metadata.
    """

    """
        OpenCV requires 8-bit images for CLAHE, skimage requires either 8-bit images or arrays with values between [0,1]
        Here, we convert the raw data into an 8-bit image to proceed
    """

    temp = image.data
    temp = temp / temp.max()
    temp = (temp * 2**8).astype(np.uint8)

    if which_package=='OpenCV':
        import cv2
        tile_grid_size = int(tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit=clip_limit_cv2,
                                tileGridSize=(tile_grid_size,tile_grid_size))
        image_data = clahe.apply(temp)

    else: # default filter
        # nbin = 256 default, for 8-bit images
        image_data = exposure.equalize_adapthist(temp,
                                                 kernel_size=kernel_size,
                                                 clip_limit=clip_limit_skimage, nbins=256)
        import skimage
        image_data = skimage.img_as_ubyte(image_data)

    return FibsemImage(data=image_data, metadata=image.metadata)
