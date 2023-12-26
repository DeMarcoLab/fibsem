import os

CLASS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "segmentation_config.yaml")

import yaml
with open(CLASS_CONFIG_PATH) as f:
    CLASS_CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    
CLASS_COLORS = CLASS_CONFIG["CLASS_COLORS"]
CLASS_LABELS = CLASS_CONFIG["CLASS_LABELS"]

import matplotlib.colors as mcolors
def convert_color_names_to_rgb(color_names: list[str]):
    if isinstance(color_names, dict):
        color_names = color_names.values()
    rgb_colors = [mcolors.to_rgb(color) for color in color_names]
    # Convert to 0-255 scale
    rgb_colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_colors]
    return rgb_colors

# map color names to rgb values
CLASS_COLORS_RGB = convert_color_names_to_rgb(CLASS_COLORS.values())


def get_colormap():
    return CLASS_COLORS_RGB

def get_class_labels():
    return CLASS_LABELS

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")