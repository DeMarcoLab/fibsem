from fibsem import constants
import logging
import numpy as np
from fibsem.structures import (
    BeamType,
    FibsemPattern,
    FibsemImage,
    FibsemRectangle,
    FibsemPatternSettings,
    FibsemMillingSettings,
    Point,
    ImageSettings,
    MicroscopeSettings
)
from typing import Union
from fibsem.microscope import FibsemMicroscope


########################### SETUP


def setup_milling(
    microscope: FibsemMicroscope,
    mill_settings: FibsemMillingSettings = None,
):
    """Setup Microscope for Ion Beam Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        patterning_mode (str, optional): Ion beam milling patterning mode. Defaults to "Serial".
        hfw (float, optional): horizontal field width for milling. Defaults to 100e-6.
    """
    microscope.setup_milling(mill_settings = mill_settings)

def run_milling_drift_corrected(
    microscope: FibsemMicroscope, 
    milling_current: float,  
    image_settings: ImageSettings, 
    ref_image: FibsemImage, 
    reduced_area: FibsemRectangle = None,
) -> None:
    """Run Ion Beam Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_current (float, optional): ion beam milling current. Defaults to None.
        asynch (bool, optional): flag to run milling asynchronously. Defaults to False.
    """
    microscope.run_milling_drift_corrected(milling_current, image_settings, ref_image, reduced_area)

def run_milling(
    microscope: FibsemMicroscope,
    milling_current: float,
    milling_voltage: float,
    asynch: bool = False,
) -> None:
    """Run Ion Beam Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_current (float, optional): ion beam milling current. Defaults to None.
        asynch (bool, optional): flag to run milling asynchronously. Defaults to False.
    """
    microscope.run_milling(milling_current, milling_voltage, asynch)


def milling_time_estimate(microscope: FibsemMicroscope, microscope_patterns) -> float:
    """Get the milling status.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance

    """

    total_time = microscope._milling_estimate(microscope_patterns)
        
    return total_time

def finish_milling(
    microscope: FibsemMicroscope, imaging_current: float = 20e-12, imaging_voltage: float = 30e3
) -> None:
    """Clear milling patterns, and restore to the imaging current.

    Args:
        microscope (FIbsemMicroscope): Fibsem microscope instance
        imaging_current (float, optional): Imaging Current. Defaults to 20e-12.

    """
    # restore imaging current
    logging.info(f"Changing to Imaging Current: {imaging_current:.2e}")
    microscope.finish_milling(imaging_current=imaging_current, imaging_voltage=imaging_voltage)
    logging.info("Finished Ion Beam Milling.")

def draw_patterns(microscope: FibsemMicroscope, patterns: list[FibsemPatternSettings]) -> None:
    """Draw a milling pattern from settings
    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
    """
    microscope_patterns = []
    for pattern in patterns:
        microscope_patterns.append(draw_pattern(microscope, pattern))
    return microscope_patterns

        
def draw_pattern(microscope: FibsemMicroscope, pattern: FibsemPatternSettings):
    """Draw a milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        pattern_settings (FibsemPatternSettings): pattern settings
        mill_settings (FibsemMillingSettings): milling settings
    """
    if pattern.pattern is FibsemPattern.Rectangle:
        microscope_pattern = microscope.draw_rectangle(pattern)

    elif pattern.pattern is FibsemPattern.Line:
        microscope_pattern = microscope.draw_line(pattern)

    elif pattern.pattern is FibsemPattern.Circle:
        microscope_pattern = microscope.draw_circle(pattern)

    elif pattern.pattern is FibsemPattern.Bitmap:
        microscope_pattern = microscope.draw_bitmap_pattern(pattern, pattern.path)

    elif pattern.pattern is FibsemPattern.Annulus:
        microscope_pattern = microscope.draw_annulus(pattern)
        
    return microscope_pattern


def draw_rectangle(
    microscope: FibsemMicroscope, pattern_settings: FibsemPatternSettings
):
    """Draw a rectangular milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        pattern_settings (FibsemPatternSettings): pattern settings
        mill_settings (FibsemMillingSettings): milling settings
    """
    pattern = microscope.draw_rectangle(pattern_settings)
    return pattern


def draw_line(microscope: FibsemMicroscope, pattern_settings: FibsemPatternSettings):
    """Draw a line milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        mill_settings (MillingSettings): milling pattern settings
    """
    pattern = microscope.draw_line(pattern_settings)
    return pattern

def draw_circle(microscope: FibsemMicroscope, pattern_settings: FibsemPatternSettings):
    """Draw a circular milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        mill_settings (MillingSettings): milling pattern settings
    """
    pattern = microscope.draw_circle(pattern_settings)
    return pattern

def convert_to_bitmap_format(path):
    from PIL import Image
    import os 
    img=Image.open(path)
    a=img.convert("RGB", palette=Image.ADAPTIVE, colors=8)
    new_path = os.path.join(os.path.dirname(path), "24bit_img.tif")
    a.save(new_path)
    return new_path

def draw_bitmap(microscope: FibsemMicroscope, pattern_settings: FibsemPatternSettings, path: str):
    """Draw a butmap milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem
        mill_settings (MillingSettings): milling pattern settings
    """
    path = convert_to_bitmap_format(path)
    pattern = microscope.draw_bitmap_pattern(pattern_settings, path)
    return pattern

def extract_trench_parameters(protocol: dict, point: Point = Point()):
    
    lamella_width = protocol["lamella_width"]
    lamella_height = protocol["lamella_height"]
    trench_height = protocol["trench_height"]
    upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
    offset = protocol["offset"]
    milling_depth = protocol["depth"]


    centre_upper_y = point.y + (lamella_height / 2 + upper_trench_height / 2 + offset)
    centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

    # mill settings
    lower_pattern_settings = FibsemPatternSettings(
        width=lamella_width,
        height=trench_height,
        depth=milling_depth,
        centre_x=point.x,
        centre_y=centre_lower_y,
        cleaning_cross_section=True,
        scan_direction="BottomToTop",
    )

    upper_pattern_settings = FibsemPatternSettings(
        width=lamella_width,
        height=upper_trench_height,
        depth=milling_depth,
        centre_x=point.x,
        centre_y=centre_upper_y,
        cleaning_cross_section=True,
        scan_direction="TopToBottom",
    )

    return  lower_pattern_settings, upper_pattern_settings

def draw_trench(microscope: FibsemMicroscope, protocol: dict, point: Point = Point()):
    """Calculate the trench milling patterns"""


    lower_pattern_settings, upper_pattern_settings = extract_trench_parameters(protocol, point)

    # draw patterns
    lower_pattern = draw_rectangle(microscope, lower_pattern_settings)
    upper_pattern = draw_rectangle(microscope, upper_pattern_settings)

    return [lower_pattern, upper_pattern]


def draw_stress_relief(
    microscope: FibsemMicroscope,
    microexpansion_protocol: dict,
    lamella_protocol: dict,
    centre_point: Point = Point(0, 0),
    scan_direction: list[str] = ["LeftToRight", "RightToLeft"], 
):
    """
    Draw the microexpansion joints for stress relief of lamella.

    Args:
        microscope (FibsemMicroscope): OpenFIBSEM microscope instance
        microexpansion_protocol (dict): Contains a dictionary of the necessary values for drawing the joints.
        lamella_protocol (dict): Lamella protocol

    Returns:
        patterns: list
    """
    width = microexpansion_protocol["width"]
    height = microexpansion_protocol["height"]
    depth = lamella_protocol["depth"]
    if scan_direction is None:
        scan_direction = ["LeftToRight", "RightToLeft"]

    left_pattern_settings = FibsemPatternSettings(
        width=width,
        height=height,
        depth=depth,
        centre_x=centre_point.x
        - lamella_protocol["lamella_width"] / 2
        - microexpansion_protocol["distance"],
        centre_y=centre_point.y,
        cleaning_cross_section=True,
        scan_direction=scan_direction[0],
    )

    right_pattern_settings = FibsemPatternSettings(
        width=width,
        height=height,
        depth=depth,
        centre_x=centre_point.x
        + lamella_protocol["lamella_width"] / 2
        + microexpansion_protocol["distance"],
        centre_y=centre_point.y,
        cleaning_cross_section=True,
        scan_direction=scan_direction[1],
    )

    left_pattern = draw_rectangle(
        microscope=microscope, pattern_settings=left_pattern_settings
    )
    right_pattern = draw_rectangle(
        microscope=microscope, pattern_settings=right_pattern_settings
    )

    return [left_pattern, right_pattern]


def draw_fiducial(
    microscope: FibsemMicroscope,
    pattern_settings: FibsemPatternSettings,
):
    """draw the fiducial milling patterns

    Args:
        microscope (FibsemMicroscope): OpenFIBSEM microscope instance
        mill_settings (dict): fiducial milling settings
        point (Point): centre x, y coordinate
    Returns
    -------
        patterns : list
            List of rectangular patterns used to create the fiducial marker.
    """
    pattern_settings.rotation += np.deg2rad(45)
    pattern_1 = draw_rectangle(microscope, pattern_settings)
    pattern_settings.rotation = pattern_settings.rotation + np.deg2rad(90)
    pattern_2 = draw_rectangle(microscope, pattern_settings)

    return [pattern_1, pattern_2]

def draw_circle_pattern(
    microscope: FibsemMicroscope, pattern_settings: FibsemPatternSettings
):
    pattern = microscope.draw_circle(pattern_settings)
    
    return pattern

def milling_protocol(
    microscope: FibsemMicroscope,
    mill_settings: FibsemMillingSettings,
    patterns: list = [FibsemPatternSettings],
    drift_correction: bool = False,
    image_settings: ImageSettings = None,
    ref_image: FibsemImage = None,
    reduced_area: FibsemRectangle = None,
    asynch: bool = False,
):
    # setup milling
    setup_milling(microscope, mill_settings)

    # draw patterns
    for pattern in patterns:
        draw_pattern(microscope, pattern)

    # run milling
    if drift_correction:
        run_milling_drift_corrected(microscope, mill_settings.milling_current, image_settings, ref_image, reduced_area)
    else:
        run_milling(microscope, mill_settings.milling_current, mill_settings.milling_voltage, asynch)

    # finish milling
    finish_milling(microscope)

from fibsem.patterning import FibsemMillingStage

def mill_stages(microscope: FibsemMicroscope, settings: MicroscopeSettings, stages: list[FibsemMillingStage], asynch: bool=False):
    for stage in stages:
        mill_stage(microscope=microscope, settings=settings, stage=stage, asynch=asynch)

def mill_stage(microscope: FibsemMicroscope, settings: MicroscopeSettings, stage: FibsemMillingStage, asynch: bool=False):

    # set up milling
    setup_milling(microscope, stage.milling)

    # draw patterns
    for pattern in stage.pattern.patterns:
        draw_pattern(microscope, pattern)

    run_milling(microscope, stage.milling.milling_current, asynch)

    # finish milling
    finish_milling(microscope)


############################# UTILS #############################

# TODO: refactor this to be less liftout specific
def read_protocol_dictionary(protocol: dict, stage_name: str) -> list[dict]:
    """Read the milling protocol settings dictionary into a structured format

    Args:
        protocol (dict): protocol dictionary
        stage_name (str): milling stage name

    Returns:
        list[dict]: milling protocol stages
    """
    # multi-stage
    if "protocol_stages" in protocol[stage_name]:
        protocol_stages = []
        for stage_settings in protocol[stage_name]["protocol_stages"]:
            tmp_settings = protocol[stage_name].copy()
            tmp_settings.update(stage_settings)
            protocol_stages.append(tmp_settings)
    # single-stage
    else:
        protocol_stages = [protocol[stage_name]]

    return protocol_stages


# def calculate_milling_time(patterns: list, milling_current: float) -> float:

#     # TODO: replace with estimate_milling_time_in_seconds
#     # TODO: interpolate between levels?
#     from fibsem import config

#     # volume (width * height * depth) / total_volume_sputter_rate

#     # calculate sputter rate
#     if milling_current in config.MILLING_SPUTTER_RATE:
#         total_volume_sputter_rate = config.MILLING_SPUTTER_RATE[milling_current]
#     else:
#         total_volume_sputter_rate = 3.920e-1

#     # calculate total milling volume
#     volume = 0
#     for stage in patterns:
#         for pattern in stage:
#             width = pattern.width * constants.METRE_TO_MICRON
#             height = pattern.height * constants.METRE_TO_MICRON
#             depth = pattern.depth * constants.METRE_TO_MICRON
#             volume += width * height * depth

#     # estimate time
#     milling_time_seconds = (
#         volume / total_volume_sputter_rate
#     )  # um3 * 1/ (um3 / s) = seconds

#     logging.info(f"WHDV: {width:.2f}um, {height:.2f}um, {depth:.2f}um, {volume:.2f}um3")
#     logging.info(f"Volume: {volume:.2e}, Rate: {total_volume_sputter_rate:.2e} um3/s")
#     logging.info(f"Milling Estimated Time: {milling_time_seconds / 60:.2f}m")

#     return milling_time_seconds


# def estimate_milling_time_in_seconds(
#     milling_stage_patterns: list[
#         list[Union[CleaningCrossSectionPattern, RectanglePattern]]
#     ]
# ) -> float:
#     """Calculate the estimated milling time for all milling patterns in a milling stage.

#     Args:
#         milling_stage_patterns (list[list[Union[CleaningCrossSectionPattern, RectanglePattern]]]): milling patterns for each stage

#     Returns:
#         float: estimated milling time (seconds)
#     """
#     total_time_seconds = 0
#     for patterns in milling_stage_patterns:
#         for pattern in patterns:
#             total_time_seconds += pattern.time

#     return total_time_seconds


### PATTERNING

# TODO: circle, bitmap, line, stream


# def _draw_bitmap_pattern(
#     microscope: SdbMicroscopeClient,
#     mill_settings: MillingSettings,
#     bitmap_pattern: BitmapPatternDefinition,
# ) -> BitmapPattern:

#     pattern = microscope.patterning.create_bitmap(
#         center_x=mill_settings.centre_x,
#         center_y=mill_settings.centre_y,
#         width=mill_settings.width,
#         height=mill_settings.height,
#         depth=mill_settings.depth,
#         bitmap_pattern_definition=bitmap_pattern,
#     )

#     return pattern


# def _draw_circle_pattern(
#     microscope: SdbMicroscopeClient, mill_settings: MillingSettings
# ):

#     return NotImplemented

#     pattern = microscope.patterning.create_circle(
#         center_x=mill_settings.centre_x,
#         center_y=mill_settings.centre_y,
#         outer_diameter=mill_settings.outer_diameter,
#         inner_diameter=mill_settings.inner_diameter,
#         depth=mill_settings.depth,
#     )

#     return pattern


# def _draw_line_pattern(microscope: SdbMicroscopeClient, mill_settings: MillingSettings):

#     return NotImplemented

#     pattern = microscope.patterning.create_line(
#         start_x=mill_settings.start_x,
#         start_y=mill_settings.start_y,
#         end_x=mill_settings.end_x,
#         end_y=mill_settings.end_y,
#         depth=mill_settings.depth,
#     )

#     return pattern


# def _draw_stream_pattern(
#     microscope: SdbMicroscopeClient,
#     mill_settings: MillingSettings,
#     stream_pattern: StreamPatternDefinition,
# ) -> StreamPattern:

#     # 2d array
#     # shape[0] is list of coordinates
#     # shape[1] is 4d: x, y, dwell_time_in_sec, blank

#     # TODO: investigate the following
#     # assume x, y in image coordinates
#     # what does blank mean / do?
#     # how to convert to / from this for lower level control

#     return NotImplemented

#     pattern = microscope.patterning.create_stream(
#         center_x=mill_settings.centre_x,
#         center_y=mill_settings.centre_y,
#         stream_pattern_definition=stream_pattern,
#     )

#     return pattern


# def _draw_rectangle_pattern(
#     microscope: SdbMicroscopeClient, protocol: dict, x: float = 0.0, y: float = 0.0
# ):

#     logging.warning(f"Depreceated: please use _draw_rectangle_pattern_v2")

#     if protocol["cleaning_cross_section"]:
#         pattern = microscope.patterning.create_cleaning_cross_section(
#             center_x=x,
#             center_y=y,
#             width=protocol["width"],
#             height=protocol["height"],
#             depth=protocol["depth"],
#         )
#     else:
#         pattern = microscope.patterning.create_rectangle(
#             center_x=x,
#             center_y=y,
#             width=protocol["width"],
#             height=protocol["height"],
#             depth=protocol["depth"],
#         )

#     # need to make each protocol setting have these....which means validation
#     pattern.rotation = np.deg2rad(protocol["rotation"])
#     pattern.scan_direction = protocol["scan_direction"]

#     return pattern

# # DEPRECATED #
# def _draw_rectangle_pattern_v2(
#     microscope: SdbMicroscopeClient, mill_settings: MillingSettings
# ) -> Union[CleaningCrossSectionPattern, RectanglePattern]:
#     """Draw a rectangular milling pattern from settings

#     Args:
#         microscope (SdbMicroscopeClient): AutoScript microscope instance
#         mill_settings (MillingSettings): milling pattern settings

#     Returns:
#         Union[CleaningCrossSectionPattern, RectanglePattern]: milling pattern
#     """
#     if mill_settings.cleaning_cross_section:
#         pattern = microscope.patterning.create_cleaning_cross_section(
#             center_x=mill_settings.centre_x,
#             center_y=mill_settings.centre_y,
#             width=mill_settings.width,
#             height=mill_settings.height,
#             depth=mill_settings.depth,
#         )
#     else:
#         pattern = microscope.patterning.create_rectangle(
#             center_x=mill_settings.centre_x,
#             center_y=mill_settings.centre_y,
#             width=mill_settings.width,
#             height=mill_settings.height,
#             depth=mill_settings.depth,
#         )

#     pattern.rotation = mill_settings.rotation
#     pattern.scan_direction = mill_settings.scan_direction

#     return pattern


# def _draw_trench_patterns(
#     microscope: SdbMicroscopeClient, protocol: dict, point: Point = Point()
# ):
#     """Calculate the trench milling patterns"""

#     lamella_width = protocol["lamella_width"]
#     lamella_height = protocol["lamella_height"]
#     trench_height = protocol["trench_height"]
#     upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
#     offset = protocol["offset"]
#     milling_depth = protocol["milling_depth"]

#     centre_upper_y = point.y + (lamella_height / 2 + upper_trench_height / 2 + offset)
#     centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

#     # mill settings
#     lower_settings = MillingSettings(
#         width=lamella_width,
#         height=trench_height,
#         depth=milling_depth,
#         centre_x=point.x,
#         centre_y=centre_lower_y,
#         scan_direction="BottomToTop",
#         cleaning_cross_section=True,
#     )

#     upper_settings = MillingSettings(
#         width=lamella_width,
#         height=upper_trench_height,
#         depth=milling_depth,
#         centre_x=point.x,
#         centre_y=centre_upper_y,
#         scan_direction="TopToBottom",
#         cleaning_cross_section=True,
#     )

#     # draw patterns
#     lower_pattern = _draw_rectangle_pattern_v2(microscope, lower_settings)
#     upper_pattern = _draw_rectangle_pattern_v2(microscope, upper_settings)

#     return [lower_pattern, upper_pattern]


# def _draw_fiducial_patterns(
#     microscope: SdbMicroscopeClient,
#     mill_settings: MillingSettings,
# ):
#     """draw the fiducial milling patterns

#     Args:
#         microscope (SdbMicroscopeClient): AutoScript microscope connection
#         mill_settings (dict): fiducial milling settings
#         point (Point): centre x, y coordinate
#     Returns
#     -------
#         patterns : list
#             List of rectangular patterns used to create the fiducial marker.
#     """

#     pattern_1 = _draw_rectangle_pattern_v2(microscope, mill_settings)
#     pattern_2 = _draw_rectangle_pattern_v2(microscope, mill_settings)
#     pattern_2.rotation = mill_settings.rotation + np.deg2rad(90)

#     return [pattern_1, pattern_2]
