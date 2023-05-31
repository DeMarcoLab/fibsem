import numpy as np
from fibsem.structures import Point, FibsemImage


def image_to_microscope_image_coordinates(
    coord: Point, image: np.ndarray, pixelsize: float
) -> Point:
    """
    Convert an image pixel coordinate to a microscope image coordinate.

    The microscope image coordinate system is centered on the image with positive Y-axis pointing upwards.

    Args:
        coord (Point): A Point object representing the pixel coordinates in the original image.
        image (np.ndarray): A numpy array representing the image.
        pixelsize (float): The pixel size in meters.

    Returns:
        Point: A Point object representing the corresponding microscope image coordinates in meters.
    """
    # convert from image pixel coord (0, 0) top left to microscope image (0, 0) mid

    # shape
    cy, cx = np.asarray(image.shape) // 2

    # distance from centre?
    dy = float(-(coord.y - cy))  # neg = down
    dx = float(coord.x - cx)  # neg = left

    point_m = convert_point_from_pixel_to_metres(Point(dx, dy), pixelsize)

    return point_m


def get_lamella_size_in_pixels(
    img: FibsemImage, protocol: dict, use_trench_height: bool = False
) -> tuple[int]:
    """Get the relative size of the lamella in pixels based on the hfw of the image.

    Args:
        img (FibsemImage): A reference image.
        protocol (dict): A dictionary containing the protocol information.
        use_trench_height (bool, optional): If True, returns the height of the trench instead of the lamella. Default is False.

    Returns:
        tuple[int]: A tuple containing the height and width of the lamella in pixels.
    """
    # get real size from protocol
    lamella_width = protocol["lamella_width"]
    lamella_height = protocol["lamella_height"]

    total_height = lamella_height
    if use_trench_height:
        trench_height = protocol["stages"][0]["trench_height"]
        total_height += 2 * trench_height

    # convert to m
    pixelsize = img.metadata.pixel_size.x
    width, height = img.metadata.image_settings.resolution
    vfw = convert_pixels_to_metres(height, pixelsize)
    hfw = convert_pixels_to_metres(width, pixelsize)

    # lamella size in px (% of image)
    lamella_height_px = int((total_height / vfw) * height)
    lamella_width_px = int((lamella_width / hfw) * width)

    return (lamella_height_px, lamella_width_px)


def convert_metres_to_pixels(distance: float, pixelsize: float) -> int:
    """
    Convert a distance in metres to pixels based on a given pixel size.

    Args:
        distance (float): The distance to convert, in metres.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        int: The distance converted to pixels, as an integer.
    """
    return int(distance / pixelsize)


def convert_pixels_to_metres(pixels: int, pixelsize: float) -> float:
    """
    Convert a distance in pixels to metres based on a given pixel size.

    Args:
        pixels (int): The number of pixels to convert.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        float: The distance converted to metres.
    """
    return float(pixels * pixelsize)


def distance_between_points(p1: Point, p2: Point) -> Point:
    """
    Calculate the Euclidean distance between two points, returning a Point object representing the result.

    Args:
        p1 (Point): The first point.
        p2 (Point): The second point.

    Returns:
        Point: A Point object representing the distance between the two points.
    """

    return Point(x=(p2.x - p1.x), y=(p2.y - p1.y))


def convert_point_from_pixel_to_metres(point: Point, pixelsize: float) -> Point:
    """
    Convert a Point object from pixel coordinates to metre coordinates, based on a given pixel size.

    Args:
        point (Point): The Point object to convert.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        Point: The converted Point object, with its x and y values in metre coordinates.
    """
    point_m = Point(
        x=convert_pixels_to_metres(point.x, pixelsize),
        y=convert_pixels_to_metres(point.y, pixelsize),
    )

    return point_m


def convert_point_from_metres_to_pixel(point: Point, pixelsize: float) -> Point:
    """
    Convert a Point object from metre coordinates to pixel coordinates, based on a given pixel size.

    Args:
        point (Point): The Point object to convert.
        pixelsize (float): The size of a pixel in metres.

    Returns:
        Point: The converted Point object, with its x and y values in pixel coordinates.
    """
    point_px = Point(
        x=convert_metres_to_pixels(point.x, pixelsize),
        y=convert_metres_to_pixels(point.y, pixelsize),
    )
    return point_px
