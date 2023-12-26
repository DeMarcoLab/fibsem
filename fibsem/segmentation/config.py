CLASS_COLORS = {
    0: "black",
    1: "red",
    2: "green",
    3: "cyan",
    4: "yellow",
    5: "magenta",
    6: "blue",
    7: "white",
    8: "orange",
    9: "purple",
    10: "pink",
    11: "brown",
    12: "gray",
    13: "olive",
    14: "maroon",
    15: "navy",
    16: "teal",
    17: "lime",
    18: "aqua",
    19: "silver",
    20: "gold",
    21: "indigo",
    22: "violet",
    23: "turquoise",
    24: "tan",
    25: "orchid",
    26: "salmon",
    27: "khaki",
    28: "coral",
    29: "crimson",
    30: "plum",
    31: "lavender",
    32: "darkgreen",
    33: "darkblue",
    34: "darkred",
    35: "darkgray",
    36: "darkorange",
}


import matplotlib.colors as mcolors
def convert_color_names_to_rgb(color_names):
    rgb_colors = [mcolors.to_rgb(color) for color in color_names]
    # Convert to 0-255 scale
    rgb_colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_colors]
    return rgb_colors

# map color names to rgb values
CLASS_COLORS_RGB = convert_color_names_to_rgb(CLASS_COLORS.values())

# TODO: enable configuration
def get_colormap():
    return CLASS_COLORS_RGB

import os
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")