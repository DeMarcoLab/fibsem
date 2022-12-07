


# sputtering rates, from microscope application files
MILLING_SPUTTER_RATE = {
    20e-12: 6.85e-3,  # 30kv
    0.2e-9: 6.578e-2,  # 30kv
    0.74e-9: 3.349e-1,  # 30kv
    0.89e-9: 3.920e-1,  # 20kv
    2.0e-9: 9.549e-1,  # 30kv
    2.4e-9: 1.309,  # 20kv
    6.2e-9: 2.907,  # 20kv
    7.6e-9: 3.041,  # 30kv
    28.0e-9: 1.18e1   # 30 kv
}

import os
import fibsem
BASE_PATH = os.path.dirname(fibsem.__file__)
CONFIG_PATH = os.path.join(BASE_PATH, "config")



from fibsem.patterning import MillingPattern
PATTERN_PROTOCOL_MAP = {
    MillingPattern.Trench: "lamella",
    MillingPattern.JCut: "jcut",
    MillingPattern.Sever: "sever",
    MillingPattern.Weld: "weld",
    MillingPattern.Cut: "cut",
    MillingPattern.Sharpen: "sharpen",
    MillingPattern.Thin: "thin_lamella",
    MillingPattern.Polish: "polish_lamella",
    MillingPattern.Flatten: "flatten_landing",
    MillingPattern.Fiducial: "fiducial",
}

# MILLING UI

NON_CHANGEABLE_MILLING_PARAMETERS = [
    "milling_current",
    "hfw",
    "jcut_angle",
    "rotation_angle",
    "tilt_angle",
    "tilt_offset",
    "resolution",
    "dwell_time",
    "reduced_area",
    "scan_direction",
    "cleaning_cross_section",
]
NON_SCALED_MILLING_PARAMETERS = [
    "size_ratio",
    "rotation",
    "tip_angle",
    "needle_angle",
    "percentage_roi_height",
    "percentage_from_lamella_surface",
]