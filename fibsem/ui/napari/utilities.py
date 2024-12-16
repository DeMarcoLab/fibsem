import logging
from typing import List, Optional, Tuple, Union

import napari
import numpy as np

from fibsem import constants
from fibsem.structures import (
    FibsemImage,
)
from fibsem.ui.napari.properties import (
    IMAGING_CROSSHAIR_LAYER_PROPERTIES,
    IMAGING_SCALEBAR_LAYER_PROPERTIES,
)

CROSSHAIR_LAYER_NAME = IMAGING_CROSSHAIR_LAYER_PROPERTIES["name"]
SCALEBAR_LAYER_NAME = IMAGING_SCALEBAR_LAYER_PROPERTIES["name"]
SCALEBAR_VALUE_PROPERTIES = IMAGING_SCALEBAR_LAYER_PROPERTIES["value"]
SCALEBAR_VALUE_LAYER_NAME = SCALEBAR_VALUE_PROPERTIES["name"]


def draw_crosshair_in_napari(
    viewer: napari.Viewer,
    sem_shape: Tuple[int, int],
    fib_shape: Optional[Tuple[int, int]] = None,
    is_checked: bool = False,
    size_ratio: float = 0.15,
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

    crosshairs = []
    for cy, cx in crosshair_centres:
        size_px = size_ratio * cy
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

    if is_checked:
        sem_shape = eb_image.data.shape
        fib_shape = ib_image.data.shape
        h, w = 0.9, 0.15
        location_points = [
            [int(sem_shape[0] * h), int(sem_shape[1] * w)],
            [int(fib_shape[0] * h), int(sem_shape[1] + fib_shape[1] * w)],
        ]

        if is_checked:
            # making the scale bar line
            scale_bar_shape = []

            for i, pt in enumerate(location_points):
                if i == 0:
                    scale_ratio, eb_scale = _scale_length_value(
                        eb_image.metadata.image_settings.hfw
                    )
                    length = scale_ratio * eb_image.data.shape[1]
                else:
                    scale_ratio, ib_scale = _scale_length_value(
                        ib_image.metadata.image_settings.hfw
                    )
                    length = scale_ratio * ib_image.data.shape[1]

                hwidth = 0.5 * length
                main_line = [
                    [pt[0] + 25, int(pt[1] - hwidth)],
                    [pt[0] + 25, int(pt[1] + hwidth)],
                ]
                left_line = [
                    [pt[0] + 50, int(pt[1] - hwidth)],
                    [pt[0], int(pt[1] - hwidth)],
                ]
                right_line = [
                    [pt[0] + 50, int(pt[1] + hwidth)],
                    [pt[0], int(pt[1] + hwidth)],
                ]
                scale_bar_shape.extend([main_line, left_line, right_line])

            if SCALEBAR_LAYER_NAME not in layers_in_napari:
                viewer.add_shapes(
                    data=scale_bar_shape,
                    shape_type=IMAGING_SCALEBAR_LAYER_PROPERTIES["shape_type"],
                    edge_width=IMAGING_SCALEBAR_LAYER_PROPERTIES["edge_width"],
                    edge_color=IMAGING_SCALEBAR_LAYER_PROPERTIES["edge_color"],
                    name=SCALEBAR_LAYER_NAME,
                )
            else:
                viewer.layers[SCALEBAR_LAYER_NAME].data = scale_bar_shape
                viewer.layers[SCALEBAR_LAYER_NAME].opacity = 1

            ## making the scale bar value

            text = {
                "string": [f"{eb_scale} um", f"{ib_scale} um"],
                "color": SCALEBAR_VALUE_PROPERTIES["text"]["color"],
            }

            if SCALEBAR_LAYER_NAME not in layers_in_napari:
                viewer.add_points(
                    data=location_points,
                    text=text,
                    name=SCALEBAR_VALUE_LAYER_NAME,
                    size=SCALEBAR_VALUE_PROPERTIES["size"],
                    edge_color=SCALEBAR_VALUE_PROPERTIES["edge_color"],
                    face_color=SCALEBAR_VALUE_PROPERTIES["face_color"],
                )
            else:
                viewer.layers[SCALEBAR_VALUE_LAYER_NAME].data = location_points
                viewer.layers[SCALEBAR_VALUE_LAYER_NAME].text = text
                viewer.layers[SCALEBAR_VALUE_LAYER_NAME].opacity = 1

    else:
        if SCALEBAR_LAYER_NAME in layers_in_napari:
            viewer.layers[SCALEBAR_LAYER_NAME].opacity = 0
            viewer.layers[SCALEBAR_VALUE_LAYER_NAME].opacity = 0
