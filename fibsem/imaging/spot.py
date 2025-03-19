from fibsem.microscope import FibsemMicroscope
from fibsem.structures import Point, BeamType
from typing import List
import time

def let_it_burn(microscope: FibsemMicroscope, coordinates: List[Point], exposure_time: float, milling_current: float) -> None:

    imaging_current = microscope.get("current", BeamType.ION)
    microscope.set("current", milling_current, BeamType.ION)

    for pt in coordinates:
        print(f'burning spot: {pt}')
        microscope.set("blanked", True, BeamType.ION)
        microscope.set("spot_mode", pt, BeamType.ION)
        microscope.set("blanked", False, BeamType.ION)
        time.sleep(exposure_time)
        microscope.set("full_frame", True, BeamType.ION)

    microscope.set("current", imaging_current, BeamType.ION)
    return