import numpy as np
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem.structures import Point


def image_to_microscope_image_coordinates(
    coord: Point, image: np.ndarray, pixelsize: float
) -> Point:

    # convert from image pixel coord (0, 0) top left to microscope image (0, 0) mid

    # shape
    cy, cx = np.asarray(image.shape) // 2

    # distance from centre?
    dy = -(coord.y - cy)  # neg = down
    dx = coord.x - cx  # neg = left

    point_m = convert_point_from_pixel_to_metres(Point(dx, dy), pixelsize)

    return point_m


def get_lamella_size_in_pixels(
    img: AdornedImage, protocol: dict, use_trench_height: bool = False
) -> tuple[int]:
    """Get the relative size of the lamella in pixels based on the hfw of the image.

    Args:
        img (AdornedImage): reference image
        protocol (dict): protocol dictionary
        use_trench_height (bool, optional): get the height of the trench (True), or Lamella. Defaults to False.

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

    # convert to m
    pixelsize = img.metadata.binary_result.pixel_size.x
    vfw = convert_pixels_to_metres(img.height, pixelsize)
    hfw = convert_pixels_to_metres(img.width, pixelsize)

    # lamella size in px (% of image)
    lamella_height_px = int((total_height / vfw) * img.height)
    lamella_width_px = int((lamella_width / hfw) * img.width)

    return (lamella_height_px, lamella_width_px)


def convert_metres_to_pixels(distance: float, pixelsize: float) -> int:
    """Convert distance in metres to pixels"""
    return int(distance / pixelsize)


def convert_pixels_to_metres(pixels: int, pixelsize: float) -> float:
    """Convert pixels to distance in metres"""
    return float(pixels * pixelsize)


def distance_between_points(p1: Point, p2: Point) -> Point:
    """Calculate the distance between two points in each coordinate"""

    return Point(x=(p2.x - p1.x), y=(p2.y - p1.y))


def convert_point_from_pixel_to_metres(point: Point, pixelsize: float) -> Point:

    point_m = Point(
        x=convert_pixels_to_metres(point.x, pixelsize),
        y=convert_pixels_to_metres(point.y, pixelsize),
    )

    return point_m


def convert_point_from_metres_to_pixel(point: Point, pixelsize: float) -> Point:

    point_px = Point(
        x=convert_metres_to_pixels(point.x, pixelsize),
        y=convert_metres_to_pixels(point.y, pixelsize),
    )
    return point_px
