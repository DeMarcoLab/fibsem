import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union


import napari
import numpy as np
from napari.layers import Layer as NapariLayer
from napari.layers.shapes.shapes import Shapes as NapariShapes


from fibsem import constants
from fibsem.structures import (
    FibsemImage,
)


CROSSHAIR_LAYER_NAME = "crosshair"
SCALEBAR_LAYER_NAME = "scalebar"
SCALEBAR_VALUE_LAYER_NAME = "scalebar_value"

def draw_crosshair_in_napari(viewer: napari.Viewer, 
                    eb_image: FibsemImage, 
                    ib_image: FibsemImage, 
                    is_checked: bool = False, 
                    width: float = 0.15) -> None:
    """Draw a crosshair in the napari viewer at the centre of each image"""

    layers_in_napari = []
    for layer in viewer.layers:
        layers_in_napari.append(layer.name)

    if eb_image.metadata is None or ib_image.metadata is None:
        logging.debug("No metadata available for crosshair")
        return

    if not is_checked:
        if CROSSHAIR_LAYER_NAME in layers_in_napari :
            viewer.layers[CROSSHAIR_LAYER_NAME].opacity = 0
            return
    
    # get the centre points of the images
    sem_shape = eb_image.data.shape
    fib_shape = ib_image.data.shape
    centre_points = [[sem_shape[0]//2, sem_shape[1]//2],
                     [fib_shape[0]//2, sem_shape[1] + fib_shape[1]//2]] # TODO: use layer translation instead of hardcoding

    crosshairs = []

    for cy, cx in centre_points:
        
        size_px = width * cy
        horizontal_line = [[cy, cx - size_px], [cy,cx + size_px]]
        vertical_line = [[cy - size_px, cx], [cy + size_px, cx]]

        crosshairs.extend([horizontal_line, vertical_line])

    # create a crosshair using shapes layer, or update the existing one
    if CROSSHAIR_LAYER_NAME not in layers_in_napari:

        viewer.add_shapes(data=crosshairs, 
                            shape_type='line', 
                            edge_width=5, 
                            edge_color='yellow', 
                            face_color='yellow', 
                            opacity=0.8, blending='translucent', 
                            name=CROSSHAIR_LAYER_NAME)
    else:
        viewer.layers[CROSSHAIR_LAYER_NAME].data = crosshairs
        viewer.layers[CROSSHAIR_LAYER_NAME].opacity = 0.8


def _scale_length_value(hfw: float) -> float:

    scale_length_value = hfw*constants.METRE_TO_MICRON*0.2

    if scale_length_value > 0 and scale_length_value < 100:
        scale_length_value = round(scale_length_value/5)*5
    if scale_length_value > 100 and scale_length_value < 500:
        scale_length_value = round(scale_length_value/25)*25
    if scale_length_value > 500 and scale_length_value < 1000:
        scale_length_value = round(scale_length_value/50)*50

    scale_ratio = scale_length_value/(hfw*constants.METRE_TO_MICRON)

    return scale_ratio,scale_length_value


def draw_scalebar_in_napari(viewer: napari.Viewer, eb_image: FibsemImage, ib_image: FibsemImage, is_checked: bool = False, width: float = 0.1) -> np.ndarray:
    """Draw a scalebar in napari viewer for each image independently."""
    layers_in_napari = []
    for layer in viewer.layers:
        layers_in_napari.append(layer.name)

    if eb_image.metadata is None or ib_image.metadata is None:
        logging.debug("No metadata available for scalebar")
        return

    if is_checked:

        location_points = [ [int(eb_image.data.shape[0]*0.9), int(eb_image.data.shape[1]*0.15)],
                           [int(ib_image.data.shape[0]*0.9), int(eb_image.data.shape[1] + ib_image.data.shape[1]*0.15)]]

        if is_checked:
            
            # making the scale bar line
            scale_bar_shape = []

            for i,point in enumerate(location_points):                

                if i == 0:
                    scale_ratio,eb_scale = _scale_length_value(eb_image.metadata.image_settings.hfw)
                    length = scale_ratio*eb_image.data.shape[1]
                else:
                    scale_ratio,ib_scale = _scale_length_value(ib_image.metadata.image_settings.hfw)
                    length = scale_ratio*ib_image.data.shape[1]

                main_line = [[point[0]+25, int(point[1]-0.5*length)], [point[0]+25, int(point[1]+0.5*length)]]
        
                left_line = [[point[0]+50, int(point[1]-0.5*length)], [point[0], int(point[1]-0.5*length)]]
                right_line = [[point[0]+50, int(point[1]+0.5*length)], [point[0], int(point[1]+0.5*length)]]

                scale_bar_shape.append(main_line)
                scale_bar_shape.append(left_line)
                scale_bar_shape.append(right_line)

            if SCALEBAR_LAYER_NAME not in layers_in_napari:
                
                viewer.add_shapes(
                    data=scale_bar_shape,
                    shape_type='line',
                    edge_width=5,
                    edge_color='yellow',
                    name=SCALEBAR_LAYER_NAME
                )
            else:

                viewer.layers[SCALEBAR_LAYER_NAME].data = scale_bar_shape
                viewer.layers[SCALEBAR_LAYER_NAME].opacity = 1

            ## making the scale bar value

            

            text = {
                "string": [f"{eb_scale} um",f"{ib_scale} um"],
                "color":"white"
            }

            if SCALEBAR_LAYER_NAME not in layers_in_napari:

                viewer.add_points(
                    data=location_points,
                    text=text,
                    size=20,
                    name=SCALEBAR_VALUE_LAYER_NAME,
                    edge_color='transparent',
                    face_color='transparent',

                )
            else:
                viewer.layers[SCALEBAR_VALUE_LAYER_NAME].data = location_points
                viewer.layers[SCALEBAR_VALUE_LAYER_NAME].text = text
                viewer.layers[SCALEBAR_VALUE_LAYER_NAME].opacity = 1

    else:

        if SCALEBAR_LAYER_NAME in layers_in_napari:
            viewer.layers[SCALEBAR_LAYER_NAME].opacity = 0
            viewer.layers[SCALEBAR_VALUE_LAYER_NAME].opacity = 0