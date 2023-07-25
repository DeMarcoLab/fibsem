import time
import logging

import numpy as np

try:
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.enumerations import (
        ManipulatorCoordinateSystem,
        ManipulatorSavedPosition,
    )
    from autoscript_sdb_microscope_client.structures import (
        ManipulatorPosition,
        MoveSettings,
        StagePosition,
    )
    THERMO = True
except:
    THERMO = False

from fibsem.structures import BeamType, MicroscopeSettings
from fibsem.microscope import FibsemMicroscope

# from fibsem.detection.detection import DetectionResult, FeatureType


############################## NEEDLE ##############################

if THERMO:
    def retract_multichem(microscope: SdbMicroscopeClient) -> None:
        # Retract the multichem
        logging.info(f"retracting multichem")
        multichem = microscope.gas.get_multichem()
        multichem.retract()
        logging.info(f"retract multichem complete")

        return

def rotation_angle_is_larger(angle1: float, angle2: float, atol: float = 90) -> bool:
    """Check the rotation angles are large

    Args:
        angle1 (float): angle1 (radians)
        angle2 (float): angle2 (radians)
        atol : tolerance (degrees)

    Returns:
        bool: rotation angle is larger than atol
    """

    return angle_difference(angle1, angle2) > (np.deg2rad(atol))


def rotation_angle_is_smaller(angle1: float, angle2: float, atol: float = 5) -> bool:
    """Check the rotation angles are large

    Args:
        angle1 (float): angle1 (radians)
        angle2 (float): angle2 (radians)
        atol : tolerance (degrees)

    Returns:
        bool: rotation angle is smaller than atol
    """

    return angle_difference(angle1, angle2) < (np.deg2rad(atol))


def angle_difference(angle1: float, angle2: float) -> float:
    """Return the difference between two angles, accounting for greater than 360, less than 0 angles

    Args:
        angle1 (float): angle1 (radians)
        angle2 (float): angle2 (radians)

    Returns:
        float: _description_
    """
    angle1 %= 2 * np.pi
    angle2 %= 2 * np.pi

    large_angle = np.max([angle1, angle2])
    small_angle = np.min([angle1, angle2])

    return min((large_angle - small_angle), ((2 * np.pi + small_angle - large_angle)))
