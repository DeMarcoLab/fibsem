
import os

# PATH = "/home/patrick/development/data/CORRELATION/3dct/3D_correlation_test_dataset"
PATH = "/home/patrick/github/3DCT/3D_correlation_test_dataset"

if not os.path.exists(PATH):
    PATH = __file__


# TODO: add preferences.yaml file to store the default settings
# TODO: add more user preferences, e.g. colors, sizes, etc.
USER_PREFERENCES = {
    "default_path": PATH,
    "show_corresponding_points": True,
    "show_thick_dims": True,
    "use_z_gauss_optim": True,
    "use_mip": False,
}

FILE_FILTERS = (
    "TIFF files (*.tif *.tiff);;"+
    "OME-TIFF files (*.ome.tiff *.ome.tif);;"
)
        

TEXT_PROPERTIES = {
    "string": "idx",  # dataframe column
    "size": 10,
    "color": "white",
    "anchor": "upper_right",
}

COORDINATE_LAYER_PROPERTIES = {
    "text": TEXT_PROPERTIES,
    "name": "Coordinates",
    "ndim": 3,
    "size": 20,
    "projection_mode": "all",
    "symbol": "ring",
    "blending": "additive",
    "opacity": 0.9,
    "coordinates": {
        "FIB": {"color": "lime", "translation": 0},
        "Surface": {"color": "red", "translation": 0},
        "FM": {"color": "cyan", "translation": None},
        "POI": {"color": "magenta", "translation": None},
    },
}

LINE_LAYER_PROPERTIES = {
    "name": "Corresponding Points",
    "shape_type": "line",
    "edge_color": "white",
    "edge_width": 2,
}

RESULTS_LAYER_PROPERTIES = {
    "name": "Results",
    "ndim": 2,
    "size": 10,
    "symbol": "disc",
    "border_color": "magenta",
    "face_color": "magenta",
    "blending": "additive",
    "opacity": 0.9,
}

ERROR_TEXT_PROPERTIES = {
    "string": "abs_err",  # dataframe column
    "size": 5,
    "color": "red",
    "anchor": "upper_right",
}
REPROJECTION_LAYER_PROPERTIES = {
    "name": "Reprojected Data",
    "ndim": 2,
    "size": 5,
    "symbol": "disc",
    "face_color": "red",
    "border_color": "red",
    "opacity": 0.5,
    "text": ERROR_TEXT_PROPERTIES,
}

DATAFRAME_PROPERTIES = {
    "columns": ["x", "y", "z", "type", "color", "idx", "translation"]
}

CORRELATION_PROPERTIES = {"min_fiducial_points": 4, "min_poi_points": 1}

INSTRUCTIONS = """
To add points:
\tShift + Click:\tAdd FIB/FM Point (4 required)
\tCtrl + Click:\tAdd POI Point (1 required)

To edit points:
\tSelect the Coordinates Layer
\tEnter Selection Mode (Press 'S')
\tClick and Drag to move points
\tPress 'Delete' to remove points
\tPress 'Z' to enter Pan/Zoom Mode
"""

DRAG_DROP_INSTRUCTIONS = """
Enable Correlation Mode to drag, drop and rescale images

Press Ctrl + Left Click to add a point of interest once correlated.
"""

# add dynamic instructions for different steps

INSTRUCTIONS2 = """

\n3D Correlation Tool
\n

\nTo add points:
\nShift + Click: Add FIB/FM Point
\nCtrl + Click: Add POI Point

To edit points:
\nSelect the Coordinates Layer
\nEnter Selection Mode (Press 'S')
\nClick and Drag to select points
\nPress 'Delete' to remove points

\nWhile the Coordinates Layer is selected:
\n Press 'S' to enter Selection Mode
\n Press 'Z' to enter Pan/Zoom Mode

To Run Correlation:
\n1. Load FIB and FM Images
\n2. Select Fiducial Coordinates (at least 4)
\n3. Select Points of Interest (at least 1)
\n4. Click 'Run Correlation'
\n The results will be displayed on the FIB Image

Data is automatically saved in the project directory.
"""