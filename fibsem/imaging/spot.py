import logging
import time
from typing import List

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point

def run_spot_burn(microscope: FibsemMicroscope,
                  coordinates: List[Point],
                  exposure_time: float,
                  milling_current: float,
                  beam_type: BeamType = BeamType.ION) -> None:
    """Run a spot burner job on the microscope. Exposes the specified coordinates for a the specified
    time at the specified current.
    Args:
        microscope: The microscope object.
        coordinates: List of points to burn.
        exposure_time: Time to expose each point in seconds.
        milling_current: Current to use for the spot.
        beam_type: The type of beam to use. (Default: BeamType.ION)
    Returns:
        None
    """

    imaging_current = microscope.get(key="current", beam_type=beam_type)
    microscope.set(key="current", value=milling_current, beam_type=beam_type)

    for pt in coordinates:
        logging.info(f'burning spot: {pt}, exposure time: {exposure_time}, milling current: {milling_current}')
        microscope.set(key="blanked", value=True, beam_type=beam_type)
        microscope.set(key="spot_mode", value=pt, beam_type=beam_type)
        microscope.set(key="blanked", value=False, beam_type=beam_type)
        time.sleep(exposure_time)
        microscope.set(key="full_frame", value=True, beam_type=beam_type)

    microscope.set("current", imaging_current, beam_type=beam_type)
    return