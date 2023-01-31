import numpy as np
from fibsem.structures import Point, FibsemImage
from PIL import Image


def create_distance_map_px(w: int, h: int) -> np.ndarray:
    x = np.arange(0, w)
    y = np.arange(0, h)

    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(((w / 2) - X) ** 2 + ((h / 2) - Y) ** 2)

    return distance


def measure_brightness(img: FibsemImage) -> float:

    return np.mean(img.data)


def rotate_image(image: FibsemImage):
    """Rotate the AdornedImage 180 degrees."""
    data = np.rot90(np.rot90(np.copy(image.data)))
    reference = FibsemImage(data=data, metadata=image.metadata)
    return reference


def normalise_image(img: FibsemImage) -> np.ndarray:
    """Normalise the image"""
    return (img.data - np.mean(img.data)) / np.std(img.data)


def cosine_stretch(img: FibsemImage, tilt_degrees: float):
    """Apply a cosine stretch to an image based on the relative tilt.

    This is required when aligning images with different tilts to ensure features are the same size.

    Args:
        img (AdornedImage): _description_
        tilt_degrees (float): _description_

    Returns:
        _type_: _description_
    """
    # note: do smaller version for negative tilt??

    tilt = np.deg2rad(tilt_degrees)

    shape = int(img.data.shape[0] / np.cos(tilt)), int(img.data.shape[1] / np.cos(tilt))

    # cosine stretch
    # larger
    resized_img = np.asarray(
        Image.fromarray(img.data).resize(size=(shape[1], shape[0]))
    )

    # crop centre out?
    c = Point(resized_img.shape[1] // 2, resized_img.shape[0] // 2)
    dy, dx = img.data.shape[0] // 2, img.data.shape[1] // 2
    scaled_img = resized_img[c.y - dy : c.y + dy, c.x - dx : c.x + dx]

    return FibsemImage(data=scaled_img, metadata=img.metadata)


def apply_image_mask(img: FibsemImage, mask: np.ndarray) -> np.ndarray:

    return normalise_image(img) * mask
