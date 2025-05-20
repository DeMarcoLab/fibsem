import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from skimage.transform import resize
from PIL import Image
import numpy as np

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import (
    BasePattern,
)
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemImage,
    FibsemImageMetadata,
    FibsemRectangleSettings,
    FibsemBitmapSettings,
    ImageSettings,
    Point,
)

COLOURS = [
    "yellow",
    "cyan",
    "magenta",
    "lime",
    "orange",
    "hotpink",
    "green",
    "blue",
    "red",
    "purple",
]


PROPERTIES = {
    "line_width": 1,
    "opacity": 0.3,
    "crosshair_size": 20,
    "crosshair_colour": "yellow",
    "rotation_point": "center",
}





def _rect_pattern_to_image_pixels(
    pattern: FibsemRectangleSettings, pixel_size: float, image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Convert rectangle pattern to image pixel coordinates.
    Args:
        pattern: FibsemRectangleSettings: Rectangle pattern to convert.
        pixel_size: float: Pixel size of the image.
        image_shape: Tuple[int, int]: Shape of the image.
    Returns:
        Tuple[int, int, int, int]: Parameters (center_x, center_y, width, height) in image pixel coordinates.
    """
    # get pattern parameters
    width = pattern.width
    height = pattern.height
    mx, my = pattern.centre_x, pattern.centre_y

    # position in metres from image centre
    pmx, pmy = mx / pixel_size, my / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] // 2, image_shape[1] // 2
    px = cx + pmx
    py = cy - pmy

    # convert parameters to pixels
    width = width / pixel_size
    height = height / pixel_size

    return px, py, width, height


# TODO: circle patches, line patches
def _draw_rectangle_pattern(
    image: FibsemImage,
    pattern: BasePattern,
    colour: str = "yellow",
    name: str = "Rectangle",
) -> List[PatchCollection]:
    """Draw a rectangle pattern on an image.
    Args:
        image: FibsemImage: Image to draw pattern on.
        pattern: BasePattern: Rectangle pattern to draw.
        colour: str: Colour of rectangle patches.
        name: str: Name of the rectangle patches.
    Returns:
        List[PatchCollection]: List of patch collections to draw.
    """
    # common image properties
    pixel_size = image.metadata.pixel_size.x  # assume isotropic
    image_shape = image.data.shape

    patch_collections = []
    p: FibsemRectangleSettings
    for i, p in enumerate(pattern.define(), 1):
        if not isinstance(p, FibsemRectangleSettings):
            logging.debug(f"Pattern {p} is not a rectangle, skipping")
            continue
        # convert from microscope image (real-space) to image pixel-space
        px, py, width, height = _rect_pattern_to_image_pixels(
            p, pixel_size, image_shape
        )

        patch_collection = PatchCollection(
            [
                mpatches.Rectangle(
                    (px - width / 2, py - height / 2),  # bottom left corner
                    width=width,
                    height=height,
                    angle=math.degrees(p.rotation),
                    rotation_point=PROPERTIES["rotation_point"],
                    linewidth=PROPERTIES["line_width"],
                    edgecolor=colour,
                    facecolor=colour,
                    alpha=PROPERTIES["opacity"],
                )
            ],
            match_original=True,
        )
        if i == 1:
            patch_collection.set_label(f"{name}")
        patch_collections.append(patch_collection)

    return patch_collections


def _draw_bitmap_pattern(
    image: FibsemImage,
    pattern: BasePattern,
    colour: str = "yellow",
    name: str = "Bitmap",
) -> List[PatchCollection]:
    """Draw a rectangle pattern on an image.
    Args:
        image: FibsemImage: Image to draw pattern on.
        pattern: BitmapPattern: Rectangle pattern to draw.
        colour: str: Colour of bitmap patches (blanked regions are inverted).
        name: str: Name of the bitmap patches.
    Returns:
        List[PatchCollection]: List of patch collections to draw.
    """
    # common image properties
    pixel_size = image.metadata.pixel_size.x  # assume isotropic
    image_shape = image.data.shape

    colour_array = mcolors.to_rgb(colour)
    inverted_colour_array = tuple(1.0 - _ for _ in colour_array)

    patch_collections = []
    p: FibsemBitmapSettings
    for i, p in enumerate(pattern.define(), 1):
        if not isinstance(p, FibsemBitmapSettings):
            logging.debug(f"Pattern {p} is not a bitmap, skipping")
            continue
        # convert from microscope image (real-space) to image pixel-space
        px, py, width, height = _rect_pattern_to_image_pixels(
            p, pixel_size, image_shape
        )
        if isinstance(p.bitmap, np.ndarray):
            array = p.bitmap.copy()  # Don't modify the pattern array!
            dwell_time_index = 0
            blanking_flag = 1
        elif p.path is not None:
            with Image.open(p.path, formats=("BMP",)) as im:
                array = np.asarray(im)
            dwell_time_index = 2
            blanking_flag = 0
        else:
            raise ValueError(
                "Unable to draw bitmap pattern from FibsemBitmapSettings as bitmap and path are both None"
            )

        dwell_time_array = array[:, :, dwell_time_index]
        blanking_array = array[:, :, 1] == blanking_flag  # blanking index is 1 for both
        del array

        if np.issubdtype(dwell_time_array.dtype, np.integer):
            # Scale to dtype range
            dwell_minmax = (
                np.iinfo(dwell_time_array.dtype).min,
                np.iinfo(dwell_time_array.dtype).max,
            )
        else:
            # Assume float scaled between 0 and 1
            dwell_minmax = (0, 1)

        # Ensure no rectangles will be subpixel (these are not displayed)
        target_shape = list(dwell_time_array.shape)
        resize_array = False
        if height < dwell_time_array.shape[0]:
            resize_array = True
            target_shape[0] = round(height)
        if width < dwell_time_array.shape[1]:
            resize_array = True
            target_shape[1] = round(width)

        if resize_array:
            dwell_time_array = resize(
                dwell_time_array,
                output_shape=target_shape,
                preserve_range=True,
                order=1,  # bi-linear interpolation
            )
            blanking_array = resize(
                blanking_array, output_shape=target_shape, preserve_range=True, order=0
            )

        dwell_time_array = dwell_time_array.astype(np.float64)
        # Cast dwell time multiplier to range 0-1
        dwell_time_array -= dwell_minmax[0]
        dwell_time_array /= dwell_minmax[1] - dwell_minmax[0]

        rectangle_height = (
            1
            if round(height) == dwell_time_array.shape[0]
            else height / dwell_time_array.shape[0]
        )
        rectangle_width = (
            1
            if round(width) == dwell_time_array.shape[1]
            else width / dwell_time_array.shape[1]
        )

        bitmap_rects = []
        for j in range(dwell_time_array.shape[0]):
            for k in range(dwell_time_array.shape[1]):
                # Draw a small rectangle for each (resized) bitmap pixel
                alpha_multiplier = 1 if blanking_array[j, k] else dwell_time_array[j, k]
                bitmap_rects.append(
                    mpatches.Rectangle(
                        (
                            px - (width / 2) + k,
                            py - (height / 2) + j,
                        ),  # bottom left corner
                        width=rectangle_width,
                        height=rectangle_height,
                        angle=math.degrees(p.rotation),
                        rotation_point=PROPERTIES["rotation_point"],
                        linewidth=0,
                        edgecolor="none",
                        facecolor=inverted_colour_array
                        if blanking_array[j, k]
                        else colour,
                        alpha=PROPERTIES["opacity"] * alpha_multiplier,
                    )
                )

        # Draw the edges
        bitmap_rects.append(
            mpatches.Rectangle(
                (
                    px - width / 2,
                    py - height / 2,
                ),  # bottom left corner
                width=width,
                height=height,
                angle=math.degrees(p.rotation),
                rotation_point=PROPERTIES["rotation_point"],
                linewidth=PROPERTIES["line_width"],
                edgecolor=colour_array,
                facecolor="none",
                alpha=PROPERTIES["opacity"],
            )
        )

        # Store all the rectangles as a patch collection
        patch_collection = PatchCollection(bitmap_rects, match_original=True)

        if i == 1:
            patch_collection.set_label(f"{name}")
        patch_collections.append(patch_collection)

    return patch_collections

def get_drawing_function(name: str) -> Callable:
    
    if name in ["Circle", "Bitmap", "Line", "SerialSection"]:
        return None
    return _draw_rectangle_pattern


def draw_milling_patterns(
    image: FibsemImage,
    milling_stages: List[FibsemMillingStage],
    crosshair: bool = True,
    scalebar: bool = True,
    title: str = "Milling Patterns",
) -> plt.Figure:
    """
    Draw milling patterns on an image.
    Args:
        image: FibsemImage: Image to draw patterns on.
        milling_stages: List[FibsemMillingStage]: Milling stages to draw.
        crosshair: bool: Draw crosshair at centre of image.
        scalebar: bool: Draw scalebar on image.
    Returns:
        plt.Figure: Figure with patterns drawn.
    """
    fig: plt.Figure
    ax: plt.Axes    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image.data, cmap="gray")

    patch_collections: List[PatchCollection] = []
    for i, stage in enumerate(milling_stages):
        colour = COLOURS[i % len(COLOURS)]
        p = stage.pattern

        drawing_func = get_drawing_function(p.name)
        if drawing_func is None:
            logging.debug(f"Drawing Pattern {p.name} not currently supported, skipping")
            continue

        patch_collections.extend(
            drawing_func(image, p, colour=colour, name=stage.name)
        )

    for pc in patch_collections:
        ax.add_collection(pc)
    ax.legend()

    # draw crosshair at centre of image
    if crosshair:
        cy, cx = image.data.shape[0] // 2, image.data.shape[1] // 2
        ax.plot(cx, cy, "y+", markersize=PROPERTIES["crosshair_size"])

    # draw scalebar
    if scalebar:
        try:
            # optional dependency, best effort
            from matplotlib_scalebar.scalebar import ScaleBar
            scalebar = ScaleBar(
                dx=image.metadata.pixel_size.x,
                color="black",
                box_color="white",
                box_alpha=0.5,
                location="lower right",
            )

            plt.gca().add_artist(scalebar)
        except ImportError:
            logging.debug("Scalebar not available, skipping")

    # set title
    ax.set_title(title)

    return fig, ax

# Plotting utilities for drawing pattern as numpy arrays

@dataclass
class DrawnPattern:
    pattern: np.ndarray
    position: Point
    is_exclusion: bool

def _create_annulus_shape(width, height, inner_radius, outer_radius):
    # Create a grid of coordinates
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2)
    # Generate the donut shape
    donut = np.logical_and(distance <= outer_radius, distance >= inner_radius).astype(int)
    return donut

def draw_annulus_shape(pattern_settings: FibsemCircleSettings, image: FibsemImage) -> DrawnPattern:
    """Convert an annulus pattern to a np array. Note: annulus can only be plotted as image
    Args:
        pattern_settings: FibsemCircleSettings: Annulus pattern settings.
        image: FibsemImage: Image to draw pattern on.
    Returns:
        np.ndarray: Annulus shape in image.
        Point: Position of the annulus in the image.
    """
    
    # image parameters (centre, pixel size)
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y

    # pattern parameters
    radius = pattern_settings.radius
    thickness = pattern_settings.thickness
    center_x = pattern_settings.centre_x
    center_y = pattern_settings.centre_y

    radius_px = radius / pixelsize_x # isotropic
    shape = int(2 * radius_px)
    inner_radius_ratio = 0 # full circle
    if not np.isclose(thickness, 0):
        inner_radius_ratio = (radius - thickness)/radius
   
    annulus_shape = _create_annulus_shape(width=shape, height=shape, 
                                          inner_radius=inner_radius_ratio, 
                                          outer_radius=1)

    # get pattern centre in image coordinates
    pattern_centre_x = int(icx + center_x / pixelsize_x)
    pattern_centre_y = int(icy - center_y / pixelsize_y)

    pos = Point(x=pattern_centre_x, y=pattern_centre_y)

    return DrawnPattern(pattern=annulus_shape, position=pos, is_exclusion=pattern_settings.is_exclusion)

def draw_rectangle_shape(pattern_settings: FibsemRectangleSettings, image: FibsemImage) -> DrawnPattern:
    """Convert a rectangle pattern to a np array.
    Args:
        pattern_settings: FibsemRectangleSettings: Rectangle pattern settings.
        image: FibsemImage: Image to draw pattern on.
    Returns:
        np.ndarray: Rectangle shape in image.
        Point: Position of the rectangle in the image.
    """
    # TODO: support rotation for drawing rectangles

    # image parameters (centre, pixel size)
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y

    # pattern parameters
    width = pattern_settings.width
    height = pattern_settings.height
    centre_x = pattern_settings.centre_x
    centre_y = pattern_settings.centre_y
    rotation = pattern_settings.rotation

    # pattern to pixel coords
    w = int(width / pixelsize_x)
    h = int(height / pixelsize_y)
    cx = int(icx + (centre_x / pixelsize_y))
    cy = int(icy - (centre_y / pixelsize_y))

    #
    shape = np.ones((h, w))
    shape[:h, :w] = 1

    # get pattern centre in image coordinates
    pos = Point(x=cx, y=cy)

    return DrawnPattern(pattern=shape, position=pos, is_exclusion=pattern_settings.is_exclusion)

def draw_pattern_shape(ps, image):
    if isinstance(ps, FibsemCircleSettings):
        return draw_annulus_shape(ps, image)
    elif isinstance(ps, FibsemRectangleSettings):
        return draw_rectangle_shape(ps, image)
    else:
        raise ValueError(f"Unsupported shape type {type(ps)}")

def draw_pattern_in_image(image: np.ndarray, 
                          drawn_pattern: DrawnPattern) -> np.ndarray:

    pattern = drawn_pattern.pattern
    pos = drawn_pattern.position
    
    # place the annulus shape in the image
    w = pattern.shape[1] // 2
    h = pattern.shape[0] // 2

    # fill the annulus shape in the image
    xmin, xmax = pos.x - w, pos.x + w
    ymin, ymax = pos.y - h, pos.y + h
    zero_image = np.zeros_like(image)
    zero_image[ymin:ymax, xmin:xmax] = pattern[:2*h, :2*w].astype(bool)

    # if the pattern is an exclusion, set the image to zero
    if drawn_pattern.is_exclusion:
        image[zero_image == 1] = 0
    else:
        # add the annulus shape to the image, clip to 1
        image = np.clip(image+zero_image, 0, 1)

    return image

def compose_pattern_image(image: np.ndarray, drawn_patterns: List[DrawnPattern]) -> np.ndarray:
    """Create an image with annulus shapes."""
    # create an empty image
    pattern_image = np.zeros_like(image)

    # sort drawn_patterns so that exclusions are drawn last
    drawn_patterns = sorted(drawn_patterns, key=lambda x: x.is_exclusion)

    # add each pattern shape to the image
    for dp in drawn_patterns:
        pattern_image = draw_pattern_in_image(pattern_image, dp)

    return pattern_image