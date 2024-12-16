


# IMAGING
RULER_LAYER_NAME = "ruler"
RULER_LINE_LAYER_NAME = "ruler_line"
RULER_VALUE_LAYER_NAME = "ruler_value"

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
    "value": {
        "name": "scalebar_value",
        "text": {   
            "color":"white"
        },
        "size": 20,
        "edge_width": 7,
        "edge_width_is_relative": False,
        "edge_color": "transparent",
        "face_color": "transparent",
    }
}

# MILLING