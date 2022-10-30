import numpy as np
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem.structures import Point

### UTILS
def pixel_to_realspace_coordinate(
    coord: list, image: AdornedImage, pixelsize: int = None
) -> list:
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

    if pixelsize is None:
        pixelsize = image.metadata.binary_result.pixel_size.x
        # deliberately don't use the y pixel size, any tilt will throw this off
    coord[1] = y_shape - coord[1]  # flip y-axis for relative coordinate system
    # reset origin to center
    coord -= np.array([x_shape / 2, y_shape / 2]).astype(np.int32)
    realspace_coord = list(np.array(coord) * pixelsize)  # to real space
    return realspace_coord  # TODO: convert to use Point struct


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
