import numpy as np
import scipy.ndimage as ndi
from fibsem import conversions
from fibsem.imaging import utils as image_utils
from fibsem.structures import Point, FibsemImage
from PIL import Image, ImageDraw


### MASKING
def create_circle_mask(
    shape: tuple = (128, 128), radius: int = 32, sigma: int = 3
) -> np.ndarray:
    """_summary_

    Args:
        shape (tuple, optional): _description_. Defaults to (128, 128).
        radius (int, optional): _description_. Defaults to 32.
        sigma (int, optional): _description_. Defaults to 3.

    Returns:
        np.ndarray: _description_
    """
    distance = image_utils.create_distance_map_px(w=shape[1], h=shape[0])
    mask = distance <= radius

    if sigma:
        mask = ndi.filters.gaussian_filter(mask, sigma=sigma)

    return mask


def create_bandpass_mask(
    shape: tuple = (256, 256), lp: int = 32, hp: int = 2, sigma: int = 3
) -> np.ndarray:
    """_summary_

    Args:
        shape (tuple, optional): _description_. Defaults to (256, 256).
        lp (int, optional): _description_. Defaults to 32.
        hp (int, optional): _description_. Defaults to 2.
        sigma (int, optional): _description_. Defaults to 3.

    Returns:
        np.ndarray: _description_
    """

    distance = image_utils.create_distance_map_px(w=shape[1], h=shape[0])

    lowpass = distance <= lp
    highpass = distance >= hp

    mask = (lowpass * highpass).astype(np.float32)
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


def create_rect_mask(
    img: np.ndarray, w: int, h: int, sigma: int = 0, pt: Point = None
) -> np.ndarray:
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
        pt = Point(img.data.shape[1] // 2, img.data.shape[0] // 2)

    mask = np.zeros_like(img)

    y_min, y_max = int(np.clip(pt.y - h / 2, 0, img.shape[1])), int(
        np.clip(pt.y + h / 2, 0, img.shape[1])
    )
    x_min, x_max = int(np.clip(pt.x - w / 2, 0, img.shape[1])), int(
        np.clip(pt.x + w / 2, 0, img.shape[1])
    )

    mask[y_min:y_max, x_min:x_max] = 1

    if sigma:
        mask = ndi.filters.gaussian_filter(mask, sigma=sigma)

    return mask


def create_lamella_mask(
    img: FibsemImage,
    protocol: dict,
    scale: int = 2,
    circ: bool = False,
    pt: Point = None,
    use_trench_height: bool = False,
) -> np.ndarray:
    """Create a mask based on the size of the lamella

    Args:
        img (FibsemImage): reference image
        settings (dict): protocol dictionary
        scale (int, optional): mask size will be multipled by this scale . Defaults to 2.
        circ (bool, optional): use a circular mask. Defaults to False.
        pt (Point, optional): original point for the mask. Defaults to None.
        use_trench_height (bool, optional): use the trench height to calculate the mask size instead of lamella height). Defaults to False.

    Returns:
        np.ndarray: mask
    """

    # get the size of the lamella in pixels
    lamella_height_px, lamella_width_px = conversions.get_lamella_size_in_pixels(
        img, protocol, use_trench_height
    )

    if circ:
        mask = create_circle_mask(
            size=(img.data.shape[1], img.data.shape[0]),
            radius=max(lamella_height_px, lamella_width_px) * scale,
            sigma=12,
        )
    else:
        mask = create_rect_mask(
            img=img.data,
            pt=pt,
            w=int(lamella_width_px * scale),
            h=int(lamella_height_px * scale),
            sigma=3,
        )

    return mask


def apply_circular_mask(img: np.ndarray, radius: int, sigma: int = 0) -> np.ndarray:
    circ_mask = create_circle_mask(img.shape, radius=radius, sigma=sigma)

    if img.ndim == 3:
        circ_mask = np.moveaxis(np.array([circ_mask, circ_mask, circ_mask]), 0, 2)

    return img * circ_mask


def create_area_mask(arr, left=False, right=False, upper=False, lower=False):
    cy, cx = np.asarray(arr.shape) // 2

    # left
    left_mask = np.zeros_like(arr)
    left_mask[:, :cx] = 1

    # right
    right_mask = np.zeros_like(arr)
    right_mask[:, cx:] = 1

    # top
    top_mask = np.zeros_like(arr)
    top_mask[:cy, :] = 1

    # bottom
    bot_mask = np.zeros_like(arr)
    bot_mask[cy:, :] = 1

    if left is True:
        h_mask = left_mask
    elif right is True:
        h_mask = right_mask
    else:
        h_mask = np.ones_like(arr)

    if upper is True:
        v_mask = top_mask
    elif lower is True:
        v_mask = bot_mask
    else:
        v_mask = np.ones_like(arr)

    mask = np.logical_and((v_mask), (h_mask))

    return mask


def create_vertical_mask(arr, width=128):
    mask = np.zeros_like(arr)
    mid = arr.shape[1] // 2
    mask[:, mid - width : mid + width] = 1
    return mask



def create_mask(arr: np.ndarray, mask_info: dict) -> np.ndarray:

    # mask_info = {
    #     "type": "circle",
    #     "radius": 500,
    #     "sigma": 0,
    #     "invert": False
    # }

    if mask_info["type"] == "circle":
        mask = create_circle_mask(arr.shape, radius=mask_info["radius"], sigma=mask_info["sigma"])

    if mask_info["type"] == "rect":
        mask = create_rect_mask(arr, pt=mask_info["pt"], w=mask_info["w"], h=mask_info["h"], sigma=mask_info["sigma"])

    if mask_info["invert"]:
        mask = np.logical_not(mask)

    return mask