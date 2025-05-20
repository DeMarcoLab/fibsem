import logging
import time
from typing import List, Optional

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point

SLEEP_TIME = 1

def run_spot_burn(microscope: FibsemMicroscope,
                  coordinates: List[Point],
                  exposure_time: float,
                  milling_current: float,
                  beam_type: BeamType = BeamType.ION,
                parent_ui: Optional['FibsemSpotBurnWidget'] = None)-> None:
    """Run a spot burner job on the microscope. Exposes the specified coordinates for a the specified
    time at the specified current.
    Args:
        microscope: The microscope object.
        coordinates: List of points to burn. (0 - 1 in image coordinates)
        exposure_time: Time to expose each point in seconds.
        milling_current: Current to use for the spot.
        beam_type: The type of beam to use. (Default: BeamType.ION)
        parent_ui: The parent UI object to emit progress signals. (Default: None)
    Returns:
        None
    """
    # - TODO: support cancelling the task
    # - QUERY: do we need to set the full frame scanning mode each time, or only at the end?

    total_estimated_time = len(coordinates) * exposure_time
    total_remaining_time = total_estimated_time

    # emit initial progress signal
    if parent_ui is not None:
        parent_ui.spot_burn_progress_signal.emit(
            {
                "current_point": 0,
                "total_points": len(coordinates),
                "remaining_time": exposure_time,
                "total_remaining_time": total_remaining_time,
                "total_estimated_time": total_estimated_time,
            }
        )

    # set the beam current to the milling current
    imaging_current = microscope.get_beam_current(beam_type=beam_type)
    microscope.set_beam_current(current=milling_current, beam_type=beam_type)

    for i, pt in enumerate(coordinates, 1):
        logging.info(f'burning spot {i}: {pt}, exposure time: {exposure_time}, milling current: {milling_current}')

        microscope.blank(beam_type=beam_type)
        microscope.set_spot_scanning_mode(point=pt, beam_type=beam_type)
        microscope.unblank(beam_type=beam_type)

        # countdown for the exposure time, emit progress signal
        remaining_time = exposure_time
        while remaining_time > 0:
            time.sleep(SLEEP_TIME)
            remaining_time -= SLEEP_TIME
            total_remaining_time -= SLEEP_TIME
            if parent_ui is not None:
                parent_ui.spot_burn_progress_signal.emit(
                    {
                        "current_point": i,
                        "total_points": len(coordinates),
                        "remaining_time": remaining_time,
                        "total_remaining_time": total_remaining_time,
                        "total_estimated_time": total_estimated_time,
                    }
                )
        microscope.set_full_frame_scanning_mode(beam_type=beam_type)

    # emit finished signal
    if parent_ui is not None:
        parent_ui.spot_burn_progress_signal.emit({"finished": True})

    microscope.set_beam_current(current=imaging_current, beam_type=beam_type)
    return
