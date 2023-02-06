from fibsem import constants
import logging
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem.structures import BeamType, MillingSettings, Point
from autoscript_sdb_microscope_client.structures import (
    BitmapPatternDefinition,
    StreamPatternDefinition,
)
from autoscript_sdb_microscope_client._dynamic_object_proxies import (
    CleaningCrossSectionPattern,
    RectanglePattern,
    BitmapPattern,
    StreamPattern,
    CirclePattern,
    LinePattern,
)
from typing import Union

########################### SETUP


def setup_milling(
    microscope: SdbMicroscopeClient,
    application_file: str = "autolamella",
    patterning_mode: str = "Serial",
    hfw: float = 150e-6,
):
    """Setup Microscope for Ion Beam Milling.

    Args:
        microscope (SdbMicroscopeClient): AutoScript microscope instance
        application_file (str, optional): Application file for ion beam milling. Defaults to "autolamella".
        patterning_mode (str, optional): Ion beam milling patterning mode. Defaults to "Serial".
        hfw (float, optional): horizontal field width for milling. Defaults to 100e-6.
    """

    microscope.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
    microscope.imaging.set_active_device(BeamType.ION.value)
    microscope.patterning.set_default_beam_type(BeamType.ION.value)  # ion beam default
    microscope.patterning.set_default_application_file(application_file)
    microscope.patterning.mode = patterning_mode
    microscope.patterning.clear_patterns()  # clear any existing patterns
    microscope.beams.ion_beam.horizontal_field_width.value = hfw
    logging.info(f"setup ion beam milling")
    logging.info(
        f"application file:  {application_file}, pattern mode: {patterning_mode}, hfw: {hfw}"
    )


def run_milling(
    microscope: SdbMicroscopeClient,
    milling_current: float,
    asynch: bool = False,
) -> None:
    """Run Ion Beam Milling.

    Args:
        microscope (SdbMicroscopeClient): AutoScript microscope instance
        milling_current (float, optional): ion beam milling current. Defaults to None.
        asynch (bool, optional): flag to run milling asynchronously. Defaults to False.
    """
    # change to milling current
    microscope.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
    if microscope.beams.ion_beam.beam_current.value != milling_current:
        logging.info(f"changing to milling current: {milling_current:.2e}")
        microscope.beams.ion_beam.beam_current.value = milling_current

    # run milling (asynchronously)
    logging.info(f"running ion beam milling now... asynchronous={asynch}")
    if asynch:
        microscope.patterning.start()
    else:
        microscope.patterning.run()
        microscope.patterning.clear_patterns()


def finish_milling(
    microscope: SdbMicroscopeClient, imaging_current: float = 20e-12
) -> None:
    """Clear milling patterns, and restore to the imaging current.

    Args:
        microscope (SdbMicroscopeClient): AutoScript microscope instance
        imaging_current (float, optional): Imaging Current. Defaults to 20e-12.

    """
    # restore imaging current
    logging.info(f"changing to imaging current: {imaging_current:.2e}")
    microscope.patterning.clear_patterns()
    microscope.beams.ion_beam.beam_current.value = imaging_current
    microscope.patterning.mode = "Serial"
    logging.info("finished ion beam milling.")


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


def calculate_milling_time(patterns: list, milling_current: float) -> float:

    # TODO: replace with estimate_milling_time_in_seconds
    # TODO: interpolate between levels?
    from fibsem import config

    # volume (width * height * depth) / total_volume_sputter_rate

    # calculate sputter rate
    if milling_current in config.MILLING_SPUTTER_RATE:
        total_volume_sputter_rate = config.MILLING_SPUTTER_RATE[milling_current]
    else:
        total_volume_sputter_rate = 3.920e-1

    # calculate total milling volume
    volume = 0
    for stage in patterns:
        for pattern in stage:
            width = pattern.width * constants.METRE_TO_MICRON
            height = pattern.height * constants.METRE_TO_MICRON
            depth = pattern.depth * constants.METRE_TO_MICRON
            volume += width * height * depth

    # estimate time
    milling_time_seconds = (
        volume / total_volume_sputter_rate
    )  # um3 * 1/ (um3 / s) = seconds

    logging.info(f"WHDV: {width:.2f}um, {height:.2f}um, {depth:.2f}um, {volume:.2f}um3")
    logging.info(f"Volume: {volume:.2e}, Rate: {total_volume_sputter_rate:.2e} um3/s")
    logging.info(f"Milling Estimated Time: {milling_time_seconds / 60:.2f}m")

    return milling_time_seconds


def estimate_milling_time_in_seconds(
    milling_stage_patterns: list[
        list[Union[CleaningCrossSectionPattern, RectanglePattern]]
    ]
) -> float:
    """Calculate the estimated milling time for all milling patterns in a milling stage.

    Args:
        milling_stage_patterns (list[list[Union[CleaningCrossSectionPattern, RectanglePattern]]]): milling patterns for each stage

    Returns:
        float: estimated milling time (seconds)
    """
    total_time_seconds = 0
    for patterns in milling_stage_patterns:
        for pattern in patterns:
            total_time_seconds += pattern.time

    return total_time_seconds


### PATTERNING

# TODO: circle, bitmap, line, stream


def _draw_bitmap_pattern(
    microscope: SdbMicroscopeClient,
    mill_settings: MillingSettings,
    bitmap_pattern: BitmapPatternDefinition,
) -> BitmapPattern:

    pattern = microscope.patterning.create_bitmap(
        center_x=mill_settings.centre_x,
        center_y=mill_settings.centre_y,
        width=mill_settings.width,
        height=mill_settings.height,
        depth=mill_settings.depth,
        bitmap_pattern_definition=bitmap_pattern,
    )

    return pattern


def _draw_circle_pattern(
    microscope: SdbMicroscopeClient, mill_settings: MillingSettings
):

    return NotImplemented

    pattern = microscope.patterning.create_circle(
        center_x=mill_settings.centre_x,
        center_y=mill_settings.centre_y,
        outer_diameter=mill_settings.outer_diameter,
        inner_diameter=mill_settings.inner_diameter,
        depth=mill_settings.depth,
    )

    return pattern


def _draw_line_pattern(microscope: SdbMicroscopeClient, mill_settings: MillingSettings):

    return NotImplemented

    pattern = microscope.patterning.create_line(
        start_x=mill_settings.start_x,
        start_y=mill_settings.start_y,
        end_x=mill_settings.end_x,
        end_y=mill_settings.end_y,
        depth=mill_settings.depth,
    )

    return pattern


def _draw_stream_pattern(
    microscope: SdbMicroscopeClient,
    mill_settings: MillingSettings,
    stream_pattern: StreamPatternDefinition,
) -> StreamPattern:

    # 2d array
    # shape[0] is list of coordinates
    # shape[1] is 4d: x, y, dwell_time_in_sec, blank

    # TODO: investigate the following
    # assume x, y in image coordinates
    # what does blank mean / do?
    # how to convert to / from this for lower level control

    return NotImplemented

    pattern = microscope.patterning.create_stream(
        center_x=mill_settings.centre_x,
        center_y=mill_settings.centre_y,
        stream_pattern_definition=stream_pattern,
    )

    return pattern


def _draw_rectangle_pattern(
    microscope: SdbMicroscopeClient, protocol: dict, x: float = 0.0, y: float = 0.0
):

    logging.warning(f"Depreceated: please use _draw_rectangle_pattern_v2")

    if protocol["cleaning_cross_section"]:
        pattern = microscope.patterning.create_cleaning_cross_section(
            center_x=x,
            center_y=y,
            width=protocol["width"],
            height=protocol["height"],
            depth=protocol["depth"],
        )
    else:
        pattern = microscope.patterning.create_rectangle(
            center_x=x,
            center_y=y,
            width=protocol["width"],
            height=protocol["height"],
            depth=protocol["depth"],
        )

    # need to make each protocol setting have these....which means validation
    pattern.rotation = np.deg2rad(protocol["rotation"])
    pattern.scan_direction = protocol["scan_direction"]

    return pattern


def _draw_rectangle_pattern_v2(
    microscope: SdbMicroscopeClient, mill_settings: MillingSettings
) -> Union[CleaningCrossSectionPattern, RectanglePattern]:
    """Draw a rectangular milling pattern from settings

    Args:
        microscope (SdbMicroscopeClient): AutoScript microscope instance
        mill_settings (MillingSettings): milling pattern settings

    Returns:
        Union[CleaningCrossSectionPattern, RectanglePattern]: milling pattern
    """
    if mill_settings.cleaning_cross_section:
        pattern = microscope.patterning.create_cleaning_cross_section(
            center_x=mill_settings.centre_x,
            center_y=mill_settings.centre_y,
            width=mill_settings.width,
            height=mill_settings.height,
            depth=mill_settings.depth,
        )
    else:
        pattern = microscope.patterning.create_rectangle(
            center_x=mill_settings.centre_x,
            center_y=mill_settings.centre_y,
            width=mill_settings.width,
            height=mill_settings.height,
            depth=mill_settings.depth,
        )

    pattern.rotation = mill_settings.rotation
    pattern.scan_direction = mill_settings.scan_direction

    return pattern


def _draw_trench_patterns(
    microscope: SdbMicroscopeClient, protocol: dict, point: Point = Point()
):
    """Calculate the trench milling patterns"""

    lamella_width = protocol["lamella_width"]
    lamella_height = protocol["lamella_height"]
    trench_height = protocol["trench_height"]
    upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
    offset = protocol["offset"]
    milling_depth = protocol["milling_depth"]

    centre_upper_y = point.y + (lamella_height / 2 + upper_trench_height / 2 + offset)
    centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

    # mill settings
    lower_settings = MillingSettings(
        width=lamella_width,
        height=trench_height,
        depth=milling_depth,
        centre_x=point.x,
        centre_y=centre_lower_y,
        scan_direction="BottomToTop",
        cleaning_cross_section=True,
    )

    upper_settings = MillingSettings(
        width=lamella_width,
        height=upper_trench_height,
        depth=milling_depth,
        centre_x=point.x,
        centre_y=centre_upper_y,
        scan_direction="TopToBottom",
        cleaning_cross_section=True,
    )

    # draw patterns
    lower_pattern = _draw_rectangle_pattern_v2(microscope, lower_settings)
    upper_pattern = _draw_rectangle_pattern_v2(microscope, upper_settings)

    return [lower_pattern, upper_pattern]


def _draw_fiducial_patterns(
    microscope: SdbMicroscopeClient,
    mill_settings: MillingSettings,
):
    """draw the fiducial milling patterns

    Args:
        microscope (SdbMicroscopeClient): AutoScript microscope connection
        mill_settings (dict): fiducial milling settings
        point (Point): centre x, y coordinate
    Returns
    -------
        patterns : list
            List of rectangular patterns used to create the fiducial marker.
    """

    pattern_1 = _draw_rectangle_pattern_v2(microscope, mill_settings)
    pattern_2 = _draw_rectangle_pattern_v2(microscope, mill_settings)
    pattern_2.rotation = mill_settings.rotation + np.deg2rad(90)

    return [pattern_1, pattern_2]
