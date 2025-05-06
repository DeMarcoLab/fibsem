import numpy as np

# IMAGING


ALIGNMENT_LAYER_PROPERTIES = {
    "name": "alignment_area",
    "shape_type": "rectangle",
    "edge_color": "lime",
    "edge_width": 20,
    "face_color": "transparent",
    "opacity": 0.5,
    "metadata": {"type": "alignment"},
}

IMAGE_TEXT_LAYER_PROPERTIES = {
    "name": "label",
    "text": {
        "string": ["ELECTRON BEAM", "ION BEAM"],
        "color": "white"
    },
    "size": 20,
    "edge_width": 7,
    "edge_width_is_relative": False,
    "edge_color": "transparent",
    "face_color": "transparent",
}

IMAGING_CROSSHAIR_LAYER_PROPERTIES = {
    "name": "crosshair",
    "shape_type": "line",
    "edge_width": 5,
    "edge_color": "yellow",
    "face_color": "yellow",
    "opacity": 0.8,
    "blending": "translucent",
}

IMAGING_SCALEBAR_LAYER_PROPERTIES = {
    "name": "scalebar",
    "shape_type": "line",
    "edge_width": 5,
    "edge_color": "yellow",
    "face_color": "yellow",
    "opacity": 0.8,
    "blending": "translucent",
    "text": {   
        "color":"white",
        "translation": np.array([-20, 0]),
        "opacity": 1,
        "sze": 20,
    },
}

IMAGING_RULER_LAYER_PROPERTIES = {
    "name": "ruler",
    "size": 20,
    "face_color": "lime",
    "edge_color": "white",
    "text": {
        "color": "white",
        "translation": np.array([-20, 0]),
        "size": 10,
    },
    "line-layer": {
         "name": "ruler_line",
         "shape_type": "line",
         "edge_width": 5,
         "edge_color": "lime",
    },
}

RULER_LAYER_NAME = IMAGING_RULER_LAYER_PROPERTIES["name"]
RULER_LINE_LAYER_NAME = IMAGING_RULER_LAYER_PROPERTIES["line-layer"]["name"]

# MILLING


## MINIMAP

OVERVIEW_IMAGE_LAYER_PROPERTIES = {
    "name": "overview-image",
    "colormap": "gray",
    "blending": "additive",
    "median_filter_size": 3,
}

GRIDBAR_IMAGE_LAYER_PROPERTIES = {
    "name": "gridbar-image",
    "spacing": 100,
    "width": 20,
}

CORRELATION_IMAGE_LAYER_PROPERTIES = {
    "name": "correlation-image",
    "colormap": "green",
    "blending": "translucent",
    "opacity": 0.2,
    "colours": ["green", "cyan", "magenta", "red", "yellow"],
}

STAGE_POSITION_SHAPE_LAYER_PROPERTIES = {
    "name": "stage-positions",
    "shape_type": "line",
    "edge_width": 5,
    "edge_color": "yellow",
    "face_color": "yellow",
    "opacity": 0.8,
    "blending": "translucent",
    "text": {
        "string": [],
        "color": "white",
        "size": 15,
        "translation": np.array([-50, 0]), # text shown 50px above the point
    },
    "saved_color": "lime",
}