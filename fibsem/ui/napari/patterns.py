import logging
import time
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import napari
import numpy as np
from napari.layers import Layer as NapariLayer
from napari.layers.shapes.shapes import Shapes as NapariShapes
from PIL import Image

from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemPatternSettings,
    FibsemRectangleSettings,
    Point,
)

# colour wheel
COLOURS = ["yellow", "cyan", "magenta", "lime", "orange", "hotpink", "green", "blue", "red", "purple"]

def convert_pattern_to_napari_circle(
    pattern_settings: FibsemCircleSettings, image: FibsemImage
):
    
    if not isinstance(pattern_settings, FibsemCircleSettings):
        raise ValueError("Pattern is not a circle")
    
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y


    # pattern to pixel coords
    r = int(pattern_settings.radius / pixelsize_x)
    cx = int(icx + (pattern_settings.centre_x / pixelsize_y))
    cy = int(icy - (pattern_settings.centre_y / pixelsize_y))

    # create corner coords
    xmin, ymin = cx - r, cy - r
    xmax, ymax = cx + r, cy + r

    # create circle
    shape = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]  # ??
    return shape


def convert_pattern_to_napari_line(
    pattern_settings: FibsemLineSettings, image: FibsemImage
) -> np.ndarray:
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y
    
    # extract pattern information from settings
    if not isinstance(pattern_settings, FibsemLineSettings):
        raise ValueError("Pattern is not a line")

    start_x = pattern_settings.start_x
    start_y = pattern_settings.start_y
    end_x = pattern_settings.end_x
    end_y = pattern_settings.end_y

    # pattern to pixel coords
    px0 = int(icx + (start_x / pixelsize_x))
    py0 = int(icy - (start_y / pixelsize_y))
    px1 = int(icx + (end_x / pixelsize_x))
    py1 = int(icy - (end_y / pixelsize_y))

    # napari shape format [[y_start, x_start], [y_end, x_end]])
    shape = [[py0, px0], [py1, px1]]
    return shape

def convert_pattern_to_napari_rect(
    pattern_settings: FibsemRectangleSettings, image: FibsemImage
) -> np.ndarray:
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y
    # extract pattern information from settings
    if isinstance(pattern_settings, FibsemLineSettings):
        pattern_width = pattern_settings.end_x - pattern_settings.start_x
        pattern_height = max(pattern_settings.end_y - pattern_settings.start_y, 0.5e-6)
        pattern_rotation = np.arctan2(
            pattern_height, pattern_width
        )  # TODO: line rotation doesnt work correctly, fix
        pattern_centre_x = (pattern_settings.end_x + pattern_settings.start_x) / 2
        pattern_centre_y = (pattern_settings.end_y + pattern_settings.start_y) / 2

    
    elif isinstance(pattern_settings, FibsemCircleSettings): #only used for out of bounds check
        pattern_width = 2*pattern_settings.radius
        pattern_height = 2*pattern_settings.radius
        pattern_centre_x = pattern_settings.centre_x
        pattern_centre_y = pattern_settings.centre_y
        pattern_rotation = 0

    else:
        pattern_width = pattern_settings.width
        pattern_height = pattern_settings.height
        pattern_centre_x = pattern_settings.centre_x
        pattern_centre_y = pattern_settings.centre_y
        pattern_rotation = pattern_settings.rotation
    # pattern to pixel coords
    w = int(pattern_width / pixelsize_x)
    h = int(pattern_height / pixelsize_y)
    cx = int(icx + (pattern_centre_x / pixelsize_y))
    cy = int(icy - (pattern_centre_y / pixelsize_y))
    r = -pattern_rotation  #
    xmin, xmax = -w / 2, w / 2
    ymin, ymax = -h / 2, h / 2
    px0 = cx + (xmin * np.cos(r) - ymin * np.sin(r))
    py0 = cy + (xmin * np.sin(r) + ymin * np.cos(r))
    px1 = cx + (xmax * np.cos(r) - ymin * np.sin(r))
    py1 = cy + (xmax * np.sin(r) + ymin * np.cos(r))
    px2 = cx + (xmax * np.cos(r) - ymax * np.sin(r))
    py2 = cy + (xmax * np.sin(r) + ymax * np.cos(r))
    px3 = cx + (xmin * np.cos(r) - ymax * np.sin(r))
    py3 = cy + (xmin * np.sin(r) + ymax * np.cos(r))
    # napari shape format
    shape = [[py0, px0], [py1, px1], [py2, px2], [py3, px3]]
    return shape

def create_crosshair_shape(centre_point: Point, image: FibsemImage) -> np.ndarray:

    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2

    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y

    pattern_centre_x = centre_point.x
    pattern_centre_y = centre_point.y

    cx = int(icx + (pattern_centre_x / pixelsize_y))
    cy = int(icy - (pattern_centre_y / pixelsize_y))

    r_angles = [0,np.deg2rad(90)] #
    w = 40
    h = 1
    crosshair_shapes = []

    for r in r_angles:
        xmin, xmax = -w / 2, w / 2
        ymin, ymax = -h / 2, h / 2
        px0 = cx + (xmin * np.cos(r) - ymin * np.sin(r))
        py0 = cy + (xmin * np.sin(r) + ymin * np.cos(r))
        px1 = cx + (xmax * np.cos(r) - ymin * np.sin(r))
        py1 = cy + (xmax * np.sin(r) + ymin * np.cos(r))
        px2 = cx + (xmax * np.cos(r) - ymax * np.sin(r))
        py2 = cy + (xmax * np.sin(r) + ymax * np.cos(r))
        px3 = cx + (xmin * np.cos(r) - ymax * np.sin(r))
        py3 = cy + (xmin * np.sin(r) + ymax * np.cos(r))
        # napari shape format
        shape = [[py0, px0], [py1, px1], [py2, px2], [py3, px3]]
        crosshair_shapes.append(shape)

    return crosshair_shapes


def convert_bitmap_pattern_to_napari_image(
        pattern_settings: FibsemBitmapSettings, image: FibsemImage
) -> np.ndarray:
    # image centre
    icy, icx = image.data.shape[0] // 2, image.data.shape[1] // 2
    # pixel size
    pixelsize_x, pixelsize_y = image.metadata.pixel_size.x, image.metadata.pixel_size.y

    resize_x = int(pattern_settings.width / pixelsize_x)
    resize_y = int(pattern_settings.height / pixelsize_y)

    
    image_bmp = Image.open(pattern_settings.path)
    image_resized = image_bmp.resize((resize_x, resize_y))
    image_rotated = image_resized.rotate(-pattern_settings.rotation, expand=True)
    img_array = np.array(image_rotated)

    pattern_centre_x = int(icx - pattern_settings.width/pixelsize_x/2) + image.data.shape[1] 
    pattern_centre_y = int(icy - pattern_settings.height/pixelsize_y/2)

    pattern_point_x = int(pattern_centre_x + pattern_settings.centre_x / pixelsize_x)
    pattern_point_y = int(pattern_centre_y - pattern_settings.centre_y / pixelsize_y)

    translate_position = (pattern_point_y,pattern_point_x)

    
    return img_array, translate_position
   

        
from fibsem.milling.patterning.plotting import compose_pattern_image

IGNORE_SHAPES_LAYERS = ["ruler_line", "crosshair", "scalebar", "scalebar_value", "label", "alignment_area"] # ignore these layers when removing all shapes
IMAGE_PATTERN_LAYERS = ["annulus-layer", "bmp_Image"]

def remove_all_napari_shapes_layers(viewer: napari.Viewer, layer_type: NapariLayer = NapariShapes, ignore: List[str] = []):

    # remove all shapes layers
    layers_to_remove = []
    layers_to_ignore = IGNORE_SHAPES_LAYERS + ignore
    for layer in viewer.layers:

        if layer.name in layers_to_ignore:
            continue
        if isinstance(layer, layer_type) or any([layer_name in layer.name for layer_name in IMAGE_PATTERN_LAYERS]):
            layers_to_remove.append(layer)
    for layer in layers_to_remove:
        viewer.layers.remove(layer)  # Not removing the second layer?


NAPARI_DRAWING_FUNCTIONS = {
    FibsemCircleSettings: convert_pattern_to_napari_circle,
    FibsemLineSettings: convert_pattern_to_napari_line,
    FibsemRectangleSettings: convert_pattern_to_napari_rect,
    FibsemBitmapSettings: convert_bitmap_pattern_to_napari_image,
}


def draw_milling_patterns_in_napari(
    viewer: napari.Viewer,
    ib_image: FibsemImage,
    translation: Tuple[int, int],
    milling_stages: List[FibsemMillingStage],
    draw_crosshair: bool = True,
):

    # convert fibsem patterns to napari shapes

    t_1 = time.time()
    active_layers = []
    for i, stage in enumerate(milling_stages):
        shape_patterns = []
        shape_types = []
        t0 = time.time()

        patterns: List[FibsemPatternSettings] = stage.pattern.define()
        point: Point = stage.pattern.point
        name: str = stage.name
        is_line_pattern: bool = False
        
        # amnulus shapes
        is_annulus: bool = False
        annulus_layer = f"{name}-annulus-layer"

        drawn_patterns = []




        for pattern_settings in patterns:
            # if isinstance(pattern_settings, FibsemBitmapSettings):
            #     if pattern_settings.path == None or pattern_settings.path == '':
            #         continue

            #     bmp_Image, translate_position = convert_bitmap_pattern_to_napari_image(pattern_settings=pattern_settings, image=ib_image)
            #     if "bmp_Image" in viewer.layers:
            #         viewer.layers.remove(viewer.layers["bmp_Image"])
            #     viewer.add_image(bmp_Image,translate=translate_position,name="bmp_Image")
            #     shape_patterns = []
            #     active_layers.append("bmp_Image")
            #     continue

            if isinstance(pattern_settings, FibsemCircleSettings):
                if pattern_settings.thickness != 0:
                    # # annulus is special case becaue napari can't draw annulus as shape, so we draw them as an image
                    # is_annulus = True
                    # drawn_patterns.append(draw_pattern_shape(pattern_settings, ib_image))
                    continue # TODO: re-enable annulus drawing
                else:
                    shape = convert_pattern_to_napari_circle(pattern_settings=pattern_settings, image=ib_image)
                    shape_types.append("ellipse")
                    active_layers.append(name)

            elif isinstance(pattern_settings, FibsemLineSettings):
                shape = convert_pattern_to_napari_line(pattern_settings=pattern_settings, image=ib_image)
                shape_types.append("line")
                active_layers.append(name)
                is_line_pattern = True
            
            else:
                shape = convert_pattern_to_napari_rect(
                    pattern_settings=pattern_settings, image=ib_image
                )
                shape_types.append("rectangle")
                active_layers.append(name)

            shape_patterns.append(shape)
        
        t1 = time.time()

        if len(shape_patterns) > 0:
            
            if draw_crosshair:
                crosshair_shapes = create_crosshair_shape(centre_point=point, image=ib_image)
                crosshair_shape_types = ["rectangle","rectangle"]
                shape_patterns += crosshair_shapes
                shape_types += crosshair_shape_types

            # _name = f"Stage {i+1:02d}"
            if name in viewer.layers:
                viewer.layers[name].data = []
                viewer.layers[name].data = shape_patterns
                viewer.layers[name].shape_type = shape_types
                viewer.layers[name].edge_color = COLOURS[i % len(COLOURS)]
                viewer.layers[name].face_color=COLOURS[i % len(COLOURS)]
            else:
                viewer.add_shapes(
                    data=shape_patterns,
                    name=name,
                    shape_type=shape_types,
                    edge_width=0.5,
                    edge_color=COLOURS[i % len(COLOURS)],
                    face_color=COLOURS[i % len(COLOURS)],
                    opacity=0.5,
                    blending="translucent",
                    translate=translation,
                )

            # TODO: properties dict for all parameters

            if is_line_pattern:
                viewer.layers[name].edge_width = 3

        # if is_annulus:
        #     # draw all the annulus for the stage on the same image
        #     if annulus_layer in viewer.layers:
        #         viewer.layers.remove(viewer.layers[annulus_layer])

        #     annulus_image = compose_pattern_image(ib_image.data, drawn_patterns)

        #     
        #     label_layer = viewer.add_labels(data=annulus_image, 
        #                         translate=translation, 
        #                         name=annulus_layer,
        #                         blending="additive",
        #                         opacity=0.6)
            
        #     cmap = {0: "black", 1: COLOURS[i % len(COLOURS)]}
        #     if hasattr(label_layer, "colormap"): # attribute changed in napari 0.5.0+
        #         label_layer.colormap = cmap
        #     else:
        #         label_layer.color = cmap
        #     active_layers.append(annulus_layer)

        t2 = time.time()
        # remove all un-updated layers (assume they have been deleted)        
        remove_all_napari_shapes_layers(viewer=viewer, layer_type=NapariShapes, ignore=active_layers)
        # TODO: annulus layers are not removed correctly
        t3 = time.time()
        logging.debug(f"_DRAW_SHAPES: CONVERT: {t1-t0}, ADD/UPDATE: {t2-t1}, REMOVE: {t3-t2}")
    t_2 = time.time()
    logging.debug(f"_DRAW_SHAPES: total time: {t_2-t_1}")

    return active_layers # list of milling pattern layers


def _draw_milling_stages_on_image(image: FibsemImage, milling_stages: List[FibsemMillingStage], show: bool = True):

    viewer = napari.Viewer()
    viewer.add_image(image.data, name='test_image')
    draw_milling_patterns_in_napari(viewer=viewer,ib_image=image,eb_image=None,milling_stages=milling_stages)
    screenshot = viewer.screenshot()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screenshot)
    viewer.close()

    for i,stage in enumerate(milling_stages):
    
        plt.plot(0,0,'-',color=COLOURS[i % len(COLOURS)],label=stage.name)

    ax.axis('off')
    ax.legend()
    if show:
        plt.show()
    
    return fig

def convert_point_to_napari(resolution: list, pixel_size: float, centre: Point):
    icy, icx = resolution[1] // 2, resolution[0] // 2

    cx = int(icx + (centre.x / pixel_size))
    cy = int(icy - (centre.y / pixel_size))

    return Point(cx, cy)

def validate_pattern_placement(
    resolution: Tuple[int, int], shape: List[List[float]]
):
    """Validate that the pattern is within the image resolution"""
    x_lim = resolution[0]
    y_lim = resolution[1]

    for coordinate in shape:
        x_coord = coordinate[1]
        y_coord = coordinate[0]

        if x_coord < 0 or x_coord > x_lim:
            return False
        if y_coord < 0 or y_coord > y_lim:
            return False

    return True