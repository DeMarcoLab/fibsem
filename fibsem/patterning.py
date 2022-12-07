import logging
import math
from enum import Enum

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client._dynamic_object_proxies import (
    RectanglePattern,
    CleaningCrossSectionPattern,
)

from fibsem import milling, validation
from fibsem.structures import Point, MicroscopeSettings, MillingSettings


class MillingPattern(Enum):
    Trench = 1
    JCut = 2
    Sever = 3
    Weld = 4
    Cut = 5
    Sharpen = 6
    Thin = 7
    Polish = 8
    Flatten = 9
    Fiducial = 10


############################## PATTERNS ##############################

# ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14
def mill_horseshoe_pattern(
    microscope: SdbMicroscopeClient, protocol: dict, point: Point = Point()
) -> list[CleaningCrossSectionPattern]:
    """Calculate the trench milling patterns"""

    lamella_width = protocol["lamella_width"]
    lamella_height = protocol["lamella_height"]
    trench_height = protocol["trench_height"]
    upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
    offset = protocol["offset"]
    milling_depth = protocol["milling_depth"]

    centre_upper_y = point.y + (lamella_height / 2 + upper_trench_height / 2 + offset)
    centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

    lower_pattern = microscope.patterning.create_cleaning_cross_section(
        point.x, centre_lower_y, lamella_width, trench_height, milling_depth,
    )
    lower_pattern.scan_direction = "BottomToTop"

    upper_pattern = microscope.patterning.create_cleaning_cross_section(
        point.x, centre_upper_y, lamella_width, upper_trench_height, milling_depth,
    )
    upper_pattern.scan_direction = "TopToBottom"

    # lhs
    side_pattern = microscope.patterning.create_cleaning_cross_section(
        center_x=point.x
        + (lamella_width / 2 - protocol["side_width"] / 2)
        - protocol["side_offset"],
        center_y=point.y,
        width=protocol["side_width"],
        height=lamella_height + offset,
        depth=milling_depth,
    )
    side_pattern.scan_direction = "TopToBottom"

    return [lower_pattern, upper_pattern, side_pattern]


def spot_weld_pattern(
    microscope: SdbMicroscopeClient, protocol: dict, point: Point = Point()
) -> list[RectanglePattern]:
    # ref: spotweld terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    n_patterns = protocol["number"]
    mill_settings = MillingSettings.__from_dict__(protocol)
    mill_settings.centre_x = point.x
    mill_settings.centre_y = (
        point.y - (n_patterns - 1) * (protocol["offset"] + protocol["height"]) / 2
    )

    patterns = []
    for i in range(n_patterns):
        pattern = milling._draw_rectangle_pattern_v2(microscope, mill_settings)
        patterns.append(pattern)
        mill_settings.centre_y += protocol["offset"] + protocol["height"]

    return patterns


def jcut_milling_patterns(
    microscope: SdbMicroscopeClient, protocol: dict, point: Point = Point()
) -> list[RectanglePattern]:
    """Create JCut milling patterns

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        protocol (dict): milling protocol
        point (Point, optional): origin point for patterns. Defaults to Point().

    Returns:
        list[RectanglePattern]: jcut milling patterns
    """
    # TODO: rename jcut to undercut

    # jcut_lhs_height = protocol["lhs_height"]
    jcut_rhs_height = protocol["rhs_height"]
    jcut_lamella_height = protocol["lamella_height"]
    jcut_width = protocol["width"]
    jcut_trench_thickness = protocol["trench_width"]
    # jcut_lhs_trench_thickness = protocol["lhs_trench_width"]
    # jcut_lhs_offset = protocol["lhs_offset"]
    jcut_milling_depth = protocol["depth"]
    jcut_h_offset = protocol["h_offset"]

    jcut_half_width = jcut_width - jcut_trench_thickness / 2
    jcut_half_height = jcut_lamella_height / 2

    use_cleaning_cross_section = protocol["cleaning_cross_section"]

    # top_jcut
    jcut_top_centre_x = point.x + jcut_width / 2 - jcut_h_offset
    jcut_top_centre_y = point.y + jcut_lamella_height
    jcut_top_width = jcut_width
    jcut_top_height = jcut_trench_thickness
    jcut_top_depth = jcut_milling_depth
    
    if use_cleaning_cross_section:
        jcut_top = microscope.patterning.create_cleaning_cross_section(
            center_x=jcut_top_centre_x,
            center_y=jcut_top_centre_y,
            width=jcut_top_width,
            height=jcut_top_height,
            depth=jcut_top_depth,
        )
    else:
        jcut_top = microscope.patterning.create_rectangle(
            center_x=jcut_top_centre_x,
            center_y=jcut_top_centre_y,
            width=jcut_top_width,
            height=jcut_top_height,
            depth=jcut_top_depth,
        )

    jcut_top.scan_direction = "TopToBottom"

    # lhs_jcut
    # jcut_lhs = microscope.patterning.create_rectangle(
    #     center_x=point.x - jcut_half_width - jcut_lhs_offset,
    #     center_y=point.y + jcut_half_height - (jcut_lhs_height / 2 - jcut_half_height),
    #     width=jcut_lhs_trench_thickness,
    #     height=jcut_lhs_height,
    #     depth=jcut_milling_depth,
    # )  # depth

    # rhs jcut
    jcut_rhs_centre_x = point.x + jcut_half_width - jcut_h_offset
    jcut_rhs_centre_y = point.y + jcut_half_height - (jcut_rhs_height / 2 - jcut_half_height) + jcut_trench_thickness / 2
    jcut_rhs_width = jcut_trench_thickness
    jcut_rhs_height = jcut_rhs_height
    jcut_rhs_depth = jcut_milling_depth

    if use_cleaning_cross_section:
        jcut_rhs = microscope.patterning.create_cleaning_cross_section(
            center_x=jcut_rhs_centre_x,
            center_y=jcut_rhs_centre_y,
            width=jcut_rhs_width,
            height=jcut_rhs_height,
            depth=jcut_rhs_depth,
        )
    else:
        jcut_rhs = microscope.patterning.create_rectangle(
            center_x=jcut_rhs_centre_x,
            center_y=jcut_rhs_centre_y,
            width=jcut_rhs_width,
            height=jcut_rhs_height,
            depth=jcut_rhs_depth,
        )

    jcut_rhs.scan_direction = "TopToBottom"

    # use parallel mode for jcut
    # microscope.patterning.mode = "Parallel"

    return [jcut_top, jcut_rhs]


def calculate_sharpen_needle_pattern(
    protocol: dict, point: Point = Point()
) -> list[dict]:

    x_0, y_0 = point.x, point.y
    height = protocol["height"]
    width = protocol["width"]
    depth = protocol["depth"]
    bias = protocol["bias"]
    hfw = protocol["hfw"]
    tip_angle = protocol["tip_angle"]  # 2NA of the needle   2*alpha
    needle_angle = protocol["needle_angle"]  # needle tilt on the screen 45 deg +/-

    alpha = tip_angle / 2  # half of NA of the needletip
    beta = np.rad2deg(
        np.arctan(width / height)
    )  # box's width and length, beta is the diagonal angle
    D = np.sqrt(width ** 2 + height ** 2) / 2  # half of box diagonal
    rotation_1 = -(needle_angle + alpha)
    rotation_2 = -(needle_angle - alpha) - 180

    dx_1 = (width / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    dy_1 = (width / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddx_1 = (height / 2) * math.sin(np.deg2rad(needle_angle + alpha))
    ddy_1 = (height / 2) * math.cos(np.deg2rad(needle_angle + alpha))
    x_1 = x_0 - dx_1 + ddx_1  # centre of the bottom box
    y_1 = y_0 - dy_1 - ddy_1  # centre of the bottom box

    dx_2 = D * math.cos(np.deg2rad(needle_angle - alpha))
    dy_2 = D * math.sin(np.deg2rad(needle_angle - alpha))
    ddx_2 = (height / 2) * math.sin(np.deg2rad(needle_angle - alpha))
    ddy_2 = (height / 2) * math.cos(np.deg2rad(needle_angle - alpha))
    x_2 = x_0 - dx_2 - ddx_2  # centre of the top box
    y_2 = y_0 - dy_2 + ddy_2  # centre of the top box

    # bottom cut pattern
    cut_coord_bottom = {
        "center_x": x_1,
        "center_y": y_1,
        "width": width,
        "height": height,  # - bias,
        "depth": depth,
        "rotation": rotation_1,
        "hfw": hfw,
    }

    # top cut pattern
    cut_coord_top = {
        "center_x": x_2,
        "center_y": y_2,
        "width": width,
        "height": height,  # - bias,
        "depth": depth,
        "rotation": rotation_2,
        "hfw": hfw,
    }

    return cut_coord_bottom, cut_coord_top


def create_sharpen_needle_patterns(
    microscope: SdbMicroscopeClient, protocol: dict, point: Point = Point()
) -> list[RectanglePattern]:

    # calculate the sharpening patterns
    cut_coord_bottom, cut_coord_top = calculate_sharpen_needle_pattern(protocol, point)

    # draw the patterns
    sharpen_patterns = []

    for cut_coord in [cut_coord_bottom, cut_coord_top]:
        center_x = cut_coord["center_x"]
        center_y = cut_coord["center_y"]
        width = cut_coord["width"]
        height = cut_coord["height"]
        depth = cut_coord["depth"]
        rotation_degrees = cut_coord["rotation"]

        # create patterns
        pattern = microscope.patterning.create_rectangle(
            center_x, center_y, width, height, depth
        )
        pattern.rotation = -np.deg2rad(rotation_degrees)
        sharpen_patterns.append(pattern)
        logging.info(f"create sharpen needle pattern")
        logging.info(
            f"x: {center_x:.2e}, y: {center_y:.2e}, w: {width:.2e}, h: {height:.2e}"
        )
        logging.info(f"d: {depth:.2e}, r: {rotation_degrees} deg")

    return sharpen_patterns


def create_milling_patterns(
    microscope: SdbMicroscopeClient,
    milling_protocol: dict,
    milling_pattern_type: MillingPattern,
    point: Point = Point(0.0, 0.0),
) -> list:
    """Redraw the milling patterns with updated milling protocol"""

    if point is None:
        point = Point(0.0, 0.0)

    if milling_pattern_type == MillingPattern.Trench:
        patterns = mill_horseshoe_pattern(
            microscope=microscope, protocol=milling_protocol, point=point
        )

    if milling_pattern_type == MillingPattern.JCut:
        patterns = jcut_milling_patterns(
            microscope=microscope, protocol=milling_protocol, point=point
        )

    if milling_pattern_type in [
        MillingPattern.Sever,
        MillingPattern.Weld,
        MillingPattern.Cut,
        MillingPattern.Flatten,
    ]:
        mill_settings = MillingSettings.__from_dict__(milling_protocol)
        mill_settings.centre_x = point.x
        mill_settings.centre_y = point.y
        patterns = milling._draw_rectangle_pattern_v2(microscope, mill_settings)

    if milling_pattern_type == MillingPattern.Sharpen:
        patterns = create_sharpen_needle_patterns(
            microscope, protocol=milling_protocol, point=point
        )

    if milling_pattern_type == MillingPattern.Thin:
        patterns = milling._draw_trench_patterns(
            microscope=microscope, protocol=milling_protocol, point=point
        )

    if milling_pattern_type == MillingPattern.Polish:
        patterns = milling._draw_trench_patterns(
            microscope=microscope, protocol=milling_protocol, point=point
        )

    if milling_pattern_type == MillingPattern.Fiducial:
        mill_settings = MillingSettings.__from_dict__(milling_protocol)
        mill_settings.centre_x = point.x
        mill_settings.centre_y = point.y
        patterns = milling._draw_fiducial_patterns(microscope, mill_settings)

    # convert patterns is list
    if not isinstance(patterns, list):
        patterns = [patterns]

    return patterns


# UTILS
def get_milling_protocol_stage_settings(
    settings: MicroscopeSettings, milling_pattern: MillingPattern
):
    from fibsem import config

    stage_name = config.PATTERN_PROTOCOL_MAP[milling_pattern]
    milling_protocol_stages = milling.read_protocol_dictionary(
        settings.protocol, stage_name
    )

    # validate protocol
    if not isinstance(milling_protocol_stages, list):
        milling_protocol_stages = [milling_protocol_stages]

    for i, stage_protocol in enumerate(milling_protocol_stages):

        milling_protocol_stages[i] = validation._validate_milling_protocol(
            stage_protocol
        )

    return milling_protocol_stages
