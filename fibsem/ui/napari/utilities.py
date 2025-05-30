import logging
from typing import List, Optional, Tuple, Union

import napari
import numpy as np

from fibsem import constants
from fibsem.structures import (
    FibsemImage,
    Point,
)
from fibsem.ui.napari.properties import (
    IMAGING_CROSSHAIR_LAYER_PROPERTIES,
    IMAGING_SCALEBAR_LAYER_PROPERTIES,
    STAGE_POSITION_SHAPE_LAYER_PROPERTIES,
)

CROSSHAIR_LAYER_NAME = IMAGING_CROSSHAIR_LAYER_PROPERTIES["name"]
SCALEBAR_LAYER_NAME = IMAGING_SCALEBAR_LAYER_PROPERTIES["name"]

def draw_crosshair_in_napari(
    viewer: napari.Viewer,
    sem_shape: Tuple[int, int],
    fib_shape: Optional[Tuple[int, int]] = None,
    is_checked: bool = False,
    size_ratio: float = 0.05,
) -> None:
    """Draw a crosshair in the napari viewer at the centre of each image. 
    The crosshair is drawn using the shapes layer in napari.
    Args:
        viewer: napari viewer object
        sem_shape: shape of the SEM image
        fib_shape: shape of the FIB image (Optional)
        is_checked: boolean value to check if the crosshair is displayed
        size_ratio: size of the crosshair (percentage of the image size)
    """

    layers_in_napari = []
    for layer in viewer.layers:
        layers_in_napari.append(layer.name)

    if not is_checked:
        if CROSSHAIR_LAYER_NAME in layers_in_napari:
            viewer.layers[CROSSHAIR_LAYER_NAME].opacity = 0
            return

    # get the centre points of the images
    crosshair_centres = [[sem_shape[0] // 2, sem_shape[1] // 2]]
    if fib_shape is not None:
        fib_centre = [fib_shape[0] // 2, sem_shape[1] + fib_shape[1] // 2]
        crosshair_centres.append(fib_centre)

    # compute the size of the crosshair
    size_px = size_ratio * sem_shape[1]

    crosshairs = []
    for cy, cx in crosshair_centres:
        horizontal_line = [[cy, cx - size_px], [cy, cx + size_px]]
        vertical_line = [[cy - size_px, cx], [cy + size_px, cx]]
        crosshairs.extend([horizontal_line, vertical_line])

    # create a crosshair using shapes layer, or update the existing one
    if CROSSHAIR_LAYER_NAME not in layers_in_napari:
        viewer.add_shapes(
            data=crosshairs,
            shape_type=IMAGING_CROSSHAIR_LAYER_PROPERTIES["shape_type"],
            edge_width=IMAGING_CROSSHAIR_LAYER_PROPERTIES["edge_width"],
            edge_color=IMAGING_CROSSHAIR_LAYER_PROPERTIES["edge_color"],
            face_color=IMAGING_CROSSHAIR_LAYER_PROPERTIES["face_color"],
            opacity=IMAGING_CROSSHAIR_LAYER_PROPERTIES["opacity"],
            blending=IMAGING_CROSSHAIR_LAYER_PROPERTIES["blending"],
            name=CROSSHAIR_LAYER_NAME,
        )
    else:
        viewer.layers[CROSSHAIR_LAYER_NAME].data = crosshairs
        viewer.layers[CROSSHAIR_LAYER_NAME].opacity = 0.8

def _scale_length_value(hfw: float) -> float:
    scale_length_value = hfw * constants.METRE_TO_MICRON * 0.2

    if scale_length_value > 0 and scale_length_value < 100:
        scale_length_value = round(scale_length_value / 5) * 5
    if scale_length_value > 100 and scale_length_value < 500:
        scale_length_value = round(scale_length_value / 25) * 25
    if scale_length_value > 500 and scale_length_value < 1000:
        scale_length_value = round(scale_length_value / 50) * 50

    scale_ratio = scale_length_value / (hfw * constants.METRE_TO_MICRON)

    return scale_ratio, scale_length_value

def draw_scalebar_in_napari(
    viewer: napari.Viewer,
    eb_image: FibsemImage,
    ib_image: FibsemImage,
    is_checked: bool = False,
    width: float = 0.1,
) -> np.ndarray:
    """Draw a scalebar in napari viewer for each image independently."""
    layers_in_napari = []
    for layer in viewer.layers:
        layers_in_napari.append(layer.name)

    if eb_image.metadata is None or ib_image.metadata is None:
        logging.debug("No metadata available for scalebar")
        return

    # if not showing the scalebar, hide the layer and return
    if not is_checked:
        if SCALEBAR_LAYER_NAME in layers_in_napari:
            viewer.layers[SCALEBAR_LAYER_NAME].opacity = 0
        return

    sem_shape = eb_image.data.shape
    fib_shape = ib_image.data.shape
    h, w = 0.9, 0.15
    location_points = [
        [int(sem_shape[0] * h), int(sem_shape[1] * w)],
        [int(fib_shape[0] * h), int(sem_shape[1] + fib_shape[1] * w)],
    ]

    # making the scale bar line
    scale_bar_shape = []
    scale_bar_txt = []
    h1 = 25
    h2 = 50

    for i, pt in enumerate(location_points):
        if i == 0:
            scale_ratio, scale_value = _scale_length_value(
                eb_image.metadata.image_settings.hfw
            )
            length = scale_ratio * sem_shape[1]
        else:
            scale_ratio, scale_value = _scale_length_value(
                ib_image.metadata.image_settings.hfw
            )
            length = scale_ratio * fib_shape[1]

        hwidth = 0.5 * length
        main_line = [
            [pt[0] + h1, int(pt[1] - hwidth)],
            [pt[0] + h1, int(pt[1] + hwidth)],
        ]
        left_line = [
            [pt[0] + h2, int(pt[1] - hwidth)],
            [pt[0], int(pt[1] - hwidth)],
        ]
        right_line = [
            [pt[0] + h2, int(pt[1] + hwidth)],
            [pt[0], int(pt[1] + hwidth)],
        ]
        scale_bar_shape.extend([main_line, left_line, right_line])
        scale_bar_txt.extend([f"{scale_value} um", "", ""])

    scale_bar_txt = {
        "string": scale_bar_txt,
        "color": IMAGING_SCALEBAR_LAYER_PROPERTIES["text"]["color"],
        "translation": IMAGING_SCALEBAR_LAYER_PROPERTIES["text"]["translation"],
    }

    if SCALEBAR_LAYER_NAME not in layers_in_napari:
        viewer.add_shapes(
            data=scale_bar_shape,
            shape_type=IMAGING_SCALEBAR_LAYER_PROPERTIES["shape_type"],
            edge_width=IMAGING_SCALEBAR_LAYER_PROPERTIES["edge_width"],
            edge_color=IMAGING_SCALEBAR_LAYER_PROPERTIES["edge_color"],
            name=SCALEBAR_LAYER_NAME,
            text=scale_bar_txt,
            opacity=1,
        )
    else:
        viewer.layers[SCALEBAR_LAYER_NAME].data = scale_bar_shape
        viewer.layers[SCALEBAR_LAYER_NAME].opacity = 1
        viewer.layers[SCALEBAR_LAYER_NAME].text = scale_bar_txt


def is_inside_image_bounds(coords: Tuple[float, float], shape: Tuple[int, int]) -> bool:
    """Check if the coordinates are inside the image bounds.
    Args:
        coords (Tuple[float, float]): y, x coordinates
        shape (Tuple[int, int]): image shape (y, x)
    Returns:
        bool: True if inside image bounds, False otherwise."""
    ycoord, xcoord = coords

    if (ycoord > 0 and ycoord < shape[0]) and (
        xcoord > 0 and xcoord < shape[1]
    ):
        return True
    
    return False

def draw_positions_in_napari(
    viewer: napari.Viewer,
    points: List[Point],
    size_px: int = 100,
    show_names: bool = False,
    layer_name: Optional[str] = None,
) -> None:
    """Draw a list of positions in napari viewer as shapes
    The crosshair is drawn using the shapes layer in napari.
    Args:
        viewer: napari viewer object
        points: list of Point objects in image coordinates
    """

    positions = []
    txt = []
    for pt in points:
        cy, cx = pt.y, pt.x # convert to image coordinates
        horizontal_line = [[cy, cx - size_px], [cy, cx + size_px]]
        vertical_line = [[cy - size_px, cx], [cy + size_px, cx]]
        positions.extend([horizontal_line, vertical_line])
        txt.extend((pt.name, ""))

    text = None
    if show_names:
        text = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["text"]
        text["string"] = txt

    if layer_name is None:
        layer_name = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["name"]

    color = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["edge_color"]
    if "saved" in layer_name:
        color = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["saved_color"]
    
    if layer_name not in viewer.layers:
        viewer.add_shapes(
            data=positions,
            shape_type=STAGE_POSITION_SHAPE_LAYER_PROPERTIES["shape_type"],
            edge_width=STAGE_POSITION_SHAPE_LAYER_PROPERTIES["edge_width"],
            edge_color=color,
            face_color=color,
            opacity=STAGE_POSITION_SHAPE_LAYER_PROPERTIES["opacity"],
            blending=STAGE_POSITION_SHAPE_LAYER_PROPERTIES["blending"],
            name=layer_name,
            text=text,
        )
    else:
        viewer.layers[layer_name].data = positions
        viewer.layers[layer_name].text = text
        viewer.layers[layer_name].opacity = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["opacity"]
        viewer.layers[layer_name].edge_color = color
        viewer.layers[layer_name].face_color = color
        viewer.layers[layer_name].edge_width = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["edge_width"]
        viewer.layers[layer_name].blending = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["blending"]
        viewer.layers[layer_name].shape_type = STAGE_POSITION_SHAPE_LAYER_PROPERTIES["shape_type"]

    return layer_name 


def is_position_inside_layer(position: Tuple[float, float], target_layer) -> bool:
    """Check if the position of the event is inside the bounds of the target layer.
    Args:
        event: napari event object containing the position
        target_layer: the layer to check against
    Returns:
        bool: True if the position is inside the layer bounds, False otherwise.
    """
    coords = target_layer.world_to_data(position)

    extent_min = target_layer.extent.data[0]  # (z, y, x)
    extent_max = target_layer.extent.data[1]

    # if they are 4d, remove the first dimension
    if len(coords) == 4:
        logging.warning(f"4D coordinates detected: {coords}, removing first dimension")
        coords = coords[1:]
        extent_min = extent_min[1:]
        extent_max = extent_max[1:]

    # convert the above logs into a json msg
    msgd = {
        "target_layer": target_layer.name,
        "event_position": position,
        "coords": coords,
        "extent_min": extent_min,
        "extent_max": extent_max,
    }
    logging.debug(msgd)

    for i, coord in enumerate(coords):
        if coord < extent_min[i] or coord > extent_max[i]:
            logging.debug(
                f"Coordinate {coord} is out of bounds ({extent_min[i]}, {extent_max[i]})"
            )
            return False

    return True