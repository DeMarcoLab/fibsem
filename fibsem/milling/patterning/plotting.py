import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fibsem.utils import format_value
from fibsem.milling.base import FibsemMillingStage
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemRectangleSettings,
    Point,
)
from .utils import (
    create_pattern_mask,
    get_pattern_bounding_box,
    get_patterns_bounding_box,
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

OVERLAP_PROPERTIES = {
    "line_width": 0.5,
    "edge_color": "red",
    "face_color": "red",
    "alpha": 0.6,
    "line_style": "--",
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

def _circle_pattern_to_image_pixels(
    pattern: FibsemCircleSettings, pixel_size: float, image_shape: Tuple[int, int]
) -> Tuple[int, int, float, float, float, float]:
    """Convert circle pattern to image pixel coordinates.
    Args:
        pattern: FibsemCircleSettings: Circle pattern to convert.
        pixel_size: float: Pixel size of the image.
        image_shape: Tuple[int, int]: Shape of the image.
    Returns:
        Tuple[int, int, float, float, float, float]: Parameters (center_x, center_y, radius, inner_radius, start_angle, end_angle) in image pixel coordinates.
    """
    # get pattern parameters
    radius = pattern.radius
    thickness = pattern.thickness
    mx, my = pattern.centre_x, pattern.centre_y
    start_angle = pattern.start_angle
    end_angle = pattern.end_angle

    # position in metres from image centre
    pmx, pmy = mx / pixel_size, my / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] // 2, image_shape[1] // 2
    px = cx + pmx
    py = cy - pmy

    # convert parameters to pixels
    radius_px = radius / pixel_size
    inner_radius_px = max(0, (radius - thickness) / pixel_size) if thickness > 0 else 0

    return px, py, radius_px, inner_radius_px, start_angle, end_angle

def _line_pattern_to_image_pixels(
    pattern: FibsemLineSettings, pixel_size: float, image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Convert line pattern to image pixel coordinates.
    Args:
        pattern: FibsemLineSettings: Line pattern to convert.
        pixel_size: float: Pixel size of the image.
        image_shape: Tuple[int, int]: Shape of the image.
    Returns:
        Tuple[int, int, int, int]: Parameters (start_x, start_y, end_x, end_y) in image pixel coordinates.
    """
    # get pattern parameters
    start_x, start_y = pattern.start_x, pattern.start_y
    end_x, end_y = pattern.end_x, pattern.end_y

    # position in metres from image centre
    start_px, start_py = start_x / pixel_size, start_y / pixel_size
    end_px, end_py = end_x / pixel_size, end_y / pixel_size

    # convert to image coordinates
    cy, cx = image_shape[0] // 2, image_shape[1] // 2
    start_pixel_x = cx + start_px
    start_pixel_y = cy - start_py
    end_pixel_x = cx + end_px
    end_pixel_y = cy - end_py

    return start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y

def _create_rectangle_patch(shape: FibsemRectangleSettings, image: FibsemImage, colour: str) -> mpatches.Rectangle:
    """Create a rectangle patch from a shape."""
    pixel_size = image.metadata.pixel_size.x
    image_shape = image.data.shape
    px, py, width, height = _rect_pattern_to_image_pixels(shape, pixel_size, image_shape)
    
    return mpatches.Rectangle(
        (px - width / 2, py - height / 2),
        width=width,
        height=height,
        angle=math.degrees(shape.rotation),
        rotation_point=PROPERTIES["rotation_point"],
        linewidth=PROPERTIES["line_width"],
        edgecolor=colour,
        facecolor=colour,
        alpha=PROPERTIES["opacity"],
    )

def _create_circle_patch(shape: FibsemCircleSettings, image: FibsemImage, colour: str) -> mpatches.Patch:
    """Create a circle patch from a shape."""
    pixel_size = image.metadata.pixel_size.x
    image_shape = image.data.shape
    px, py, radius_px, inner_radius_px, start_angle, end_angle = _circle_pattern_to_image_pixels(
        shape, pixel_size, image_shape
    )
    
    if inner_radius_px > 0:
        # annulus/ring pattern
        return mpatches.Annulus(
            (px, py),
            r=inner_radius_px,
            width=radius_px - inner_radius_px,
            angle=math.degrees(shape.rotation),
            linewidth=PROPERTIES["line_width"],
            edgecolor=colour,
            facecolor=colour,
            alpha=PROPERTIES["opacity"],
        )
    elif start_angle != 0 or end_angle != 360:
        # arc/wedge pattern
        return mpatches.Wedge(
            (px, py),
            r=radius_px,
            theta1=start_angle,
            theta2=end_angle,
            linewidth=PROPERTIES["line_width"],
            edgecolor=colour,
            facecolor=colour,
            alpha=PROPERTIES["opacity"],
        )
    else:
        # full circle pattern
        return mpatches.Circle(
            (px, py),
            radius=radius_px,
            linewidth=PROPERTIES["line_width"],
            edgecolor=colour,
            facecolor=colour,
            alpha=PROPERTIES["opacity"],
        )

def _create_line_patch(shape: FibsemLineSettings, image: FibsemImage, colour: str) -> mpatches.FancyArrowPatch:
    """Create a line patch from a shape."""
    pixel_size = image.metadata.pixel_size.x
    image_shape = image.data.shape
    start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y = _line_pattern_to_image_pixels(
        shape, pixel_size, image_shape
    )
    
    return mpatches.FancyArrowPatch(
        (start_pixel_x, start_pixel_y),
        (end_pixel_x, end_pixel_y),
        linewidth=PROPERTIES["line_width"] * 2,
        edgecolor=colour,
        facecolor=colour,
        alpha=PROPERTIES["opacity"] + 0.2,
        arrowstyle='-',
    )



def _detect_pattern_overlaps(milling_stages: List[FibsemMillingStage], image: FibsemImage) -> List[mpatches.Patch]:
    """Detect overlapping regions between patterns and create patches to highlight them.
    
    Args:
        milling_stages: List of milling stages to check for overlaps.
        image: FibsemImage for coordinate conversion.
    
    Returns:
        List of patches representing overlap regions.
    """
    if len(milling_stages) < 2:
        return []
    
    overlap_patches = []
    
    # Create masks for each pattern
    pattern_masks = []
    for stage in milling_stages:
        stage_mask = create_pattern_mask(stage, image, include_exclusions=False)
        pattern_masks.append(stage_mask)
    
    # Find overlaps between patterns
    for i in range(len(pattern_masks)):
        for j in range(i + 1, len(pattern_masks)):
            overlap = pattern_masks[i] & pattern_masks[j]
            if np.any(overlap):
                # Create contour patches for overlap regions
                try:
                    import skimage.measure
                    contours = skimage.measure.find_contours(overlap.astype(float), 0.5)
                    
                    for contour in contours:
                        if len(contour) > 3:  # Only create patches for significant overlaps
                            # Swap x,y coordinates for matplotlib (contour gives row,col)
                            contour_xy = contour[:, [1, 0]]
                            patch = mpatches.Polygon(
                                contour_xy,
                                closed=True,
                                linewidth=OVERLAP_PROPERTIES["line_width"],
                                edgecolor=OVERLAP_PROPERTIES["edge_color"],
                                facecolor=OVERLAP_PROPERTIES["face_color"],
                                alpha=OVERLAP_PROPERTIES["alpha"],
                                linestyle=OVERLAP_PROPERTIES["line_style"],
                            )
                            overlap_patches.append(patch)
                except ImportError:
                    # Fallback: create simple rectangle patches for overlap bounding boxes
                    overlap_coords = np.where(overlap)
                    if len(overlap_coords[0]) > 0:
                        y_min, y_max = overlap_coords[0].min(), overlap_coords[0].max()
                        x_min, x_max = overlap_coords[1].min(), overlap_coords[1].max()
                        
                        patch = mpatches.Rectangle(
                            (x_min, y_min),
                            x_max - x_min,
                            y_max - y_min,
                            linewidth=OVERLAP_PROPERTIES["line_width"],
                            edgecolor=OVERLAP_PROPERTIES["edge_color"],
                            facecolor=OVERLAP_PROPERTIES["face_color"],
                            alpha=OVERLAP_PROPERTIES["alpha"],
                            linestyle=OVERLAP_PROPERTIES["line_style"],
                        )
                        overlap_patches.append(patch)
    
    return overlap_patches

def draw_milling_patterns(
    image: FibsemImage,
    milling_stages: List[FibsemMillingStage],
    crosshair: bool = True,
    scalebar: bool = True,
    title: str = "Milling Patterns",
    show_current: bool = False,
    show_preset: bool = False,
    show_depth: bool = False,
    highlight_overlaps: bool = False,
) -> plt.Figure:
    """
    Draw milling patterns on an image. Supports patterns composed of multiple shape types.
    Args:
        image: FibsemImage: Image to draw patterns on.
        milling_stages: List[FibsemMillingStage]: Milling stages to draw.
        crosshair: bool: Draw crosshair at centre of image.
        scalebar: bool: Draw scalebar on image.
        title: str: Title for the plot.
        show_current: bool: Show milling current in legend.
        show_preset: bool: Show preset name in legend.
        show_depth: bool: Show pattern depth in microns in legend.
        highlight_overlaps: bool: Highlight overlapping pattern regions.
    Returns:
        plt.Figure: Figure with patterns drawn.
    """
    fig: plt.Figure
    ax: plt.Axes    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image.data, cmap="gray")

    patches = []
    for i, stage in enumerate(milling_stages):
        colour = COLOURS[i % len(COLOURS)]
        pattern = stage.pattern

        extra_parts = []
        if show_current:
            extra_parts.append(format_value(stage.milling.milling_current, "A"))
        if show_preset:
            extra_parts.append(stage.milling.preset)
        if show_depth:
            # Get depth from pattern
            depth_m = getattr(pattern, 'depth', None)
            if depth_m is not None:
                depth_um = depth_m * 1e6  # Convert from meters to microns
                extra_parts.append(f"{depth_um:.1f}μm")
        
        extra = ", ".join(extra_parts)

        # Get all shapes from the pattern
        try:
            shapes = pattern.define()
        except Exception as e:
            logging.debug(f"Failed to define pattern {pattern.name}: {e}")
            continue

        # Process each shape individually
        stage_patches = []
        for j, shape in enumerate(shapes):
            try:
                # Get the appropriate drawing function based on shape type
                if isinstance(shape, FibsemRectangleSettings):
                    patch = _create_rectangle_patch(shape, image, colour)
                elif isinstance(shape, FibsemCircleSettings):
                    patch = _create_circle_patch(shape, image, colour)
                elif isinstance(shape, FibsemLineSettings):
                    patch = _create_line_patch(shape, image, colour)
                else:
                    logging.debug(f"Unsupported shape type {type(shape)}, skipping")
                    continue
                
                # Set label only for the first shape of each stage
                if j == 0:
                    lbl = f"{stage.name}"
                    if extra:
                        lbl += f" ({extra})"
                    patch.set_label(lbl)
                
                stage_patches.append(patch)
                
            except Exception as e:
                logging.debug(f"Failed to create patch for shape {type(shape)}: {e}")
                continue
        
        patches.extend(stage_patches)

    for patch in patches:
        ax.add_patch(patch)
    
    # Detect and highlight overlaps if requested
    if highlight_overlaps:
        overlap_patches = _detect_pattern_overlaps(milling_stages, image)
        for overlap_patch in overlap_patches:
            ax.add_patch(overlap_patch)
        
        # Add overlap indication to legend
        if overlap_patches:
            overlap_patches[0].set_label("Overlaps")
    
    ax.legend()

    # set axis limits
    ax.set_xlim(0, image.data.shape[1])
    ax.set_ylim(image.data.shape[0], 0)

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
    """Convert a rectangle pattern to a np array with rotation support.
    Args:
        pattern_settings: FibsemRectangleSettings: Rectangle pattern settings.
        image: FibsemImage: Image to draw pattern on.
    Returns:
        DrawnPattern: Rectangle shape in image with rotation applied.
    """
    from scipy.ndimage import rotate

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
    cx = int(icx + (centre_x / pixelsize_x))  # Fix: use pixelsize_x for x coordinate
    cy = int(icy - (centre_y / pixelsize_y))

    # Create base rectangle shape
    shape = np.ones((h, w), dtype=float)

    # Apply rotation if specified
    if not np.isclose(rotation, 0):
        # Convert radians to degrees for scipy
        rotation_degrees = np.degrees(rotation)
        
        # Rotate the shape, reshape=True to accommodate the rotated rectangle
        shape = rotate(shape, rotation_degrees, reshape=True, order=1, prefilter=False)
        
        # Convert back to binary (rotation may introduce intermediate values)
        shape = (shape > 0.5).astype(float)

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


def simple_example(stages: List[FibsemMillingStage], image: FibsemImage) -> plt.Figure:
    """Simple demonstration of masks and bounding boxes."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Show masks
    ax1.imshow(image.data, cmap='gray', alpha=0.7)
    ax1.set_title("Pattern Masks")
    
    for i, stage in enumerate(stages):
        # Create mask
        mask = create_pattern_mask(stage, image)
        
        # Show mask as colored overlay
        masked = np.ma.masked_where(~mask, mask)
        ax1.imshow(masked, alpha=0.6, cmap='gray', vmin=0, vmax=10)
        
        print(f"{stage.name}: {np.sum(mask)} pixels")
    
    # Plot 2: Show bounding boxes
    ax2.imshow(image.data, cmap='gray')
    ax2.set_title("Bounding Boxes")
    
    for i, stage in enumerate(stages):
        # Get bounding box
        x_min, y_min, x_max, y_max = get_pattern_bounding_box(stage, image)
        
        if (x_min, y_min, x_max, y_max) != (0, 0, 0, 0):
            # Draw bounding box using COLOURS from plotting.py
            bbox = mpatches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=COLOURS[i % len(COLOURS)], facecolor='none'
            )
            ax2.add_patch(bbox)
            ax2.text(x_min, y_min-5, stage.name, color=COLOURS[i % len(COLOURS)], fontsize=9)
    
    # Add combined bounding box
    x_min, y_min, x_max, y_max = get_patterns_bounding_box(stages, image)
    if (x_min, y_min, x_max, y_max) != (0, 0, 0, 0):
        combined_bbox = mpatches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=3, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax2.add_patch(combined_bbox)
        ax2.text(x_min, y_max+10, 'Combined', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig