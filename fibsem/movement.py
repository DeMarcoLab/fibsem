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

    # # TODO: make these consistenly use degrees or radians...
    # def rotation_angle_is_larger(angle1: float, angle2: float, atol: float = 90) -> bool:
    #     """Check the rotation angles are large

    #     Args:
    #         angle1 (float): angle1 (radians)
    #         angle2 (float): angle2 (radians)
    #         atol : tolerance (degrees)

    #     Returns:
    #         bool: rotation angle is larger than atol
    #     """

    #     return angle_difference(angle1, angle2) > (np.deg2rad(atol))


    # def rotation_angle_is_smaller(angle1: float, angle2: float, atol: float = 5) -> bool:
    #     """Check the rotation angles are large

    #     Args:
    #         angle1 (float): angle1 (radians)
    #         angle2 (float): angle2 (radians)
    #         atol : tolerance (degrees)

    #     Returns:
    #         bool: rotation angle is smaller than atol
    #     """

    #     return angle_difference(angle1, angle2) < (np.deg2rad(atol))


    # def angle_difference(angle1: float, angle2: float) -> float:
    #     """Return the difference between two angles, accounting for greater than 360, less than 0 angles

    #     Args:
    #         angle1 (float): angle1 (radians)
    #         angle2 (float): angle2 (radians)

    #     Returns:
    #         float: _description_
    #     """
    #     angle1 %= 2 * np.pi
    #     angle2 %= 2 * np.pi

    #     large_angle = np.max([angle1, angle2])
    #     small_angle = np.min([angle1, angle2])

    #     return min((large_angle - small_angle), ((2 * np.pi + small_angle - large_angle)))


    def safe_rotation_movement(
        microscope: SdbMicroscopeClient, stage_position: StagePosition
    ):
        """Tilt the stage flat when performing a large rotation to prevent collision.

        Args:
            microscope (SdbMicroscopeClient): autoscript microscope instance
            stage_position (StagePosition): desired stage position.
        """
        stage = microscope.specimen.stage
        stage_settings = MoveSettings(rotate_compucentric=True)

        # tilt flat for large rotations to prevent collisions
        if rotation_angle_is_larger(stage_position.r, stage.current_position.r):

            stage.absolute_move(
                StagePosition(
                    t=np.deg2rad(0), coordinate_system=stage_position.coordinate_system
                ),
                stage_settings,
            )
            logging.info(f"tilting to flat for large rotation.")

        return


    def safe_absolute_stage_movement(
        microscope: SdbMicroscopeClient, stage_position: StagePosition
    ) -> None:
        """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system

        Args:
            microscope (SdbMicroscopeClient): autoscript microscope instance
            stage_position (StagePosition): desired stage position

        Returns:
            StagePosition: _description_
        """

        # tilt flat for large rotations to prevent collisions
        safe_rotation_movement(microscope, stage_position)

        stage = microscope.specimen.stage
        stage_settings = MoveSettings(rotate_compucentric=True)

        stage.absolute_move(
            StagePosition(
                r=stage_position.r, coordinate_system=stage_position.coordinate_system
            ),
            stage_settings,
        )
        logging.info(f"safe moving to {stage_position}")
        stage.absolute_move(stage_position, stage_settings)

        # # rotation check
        # while rotation_angle_is_larger(stage_position.r, stage.current_position.r, 0.5):

        #     # rotate the difference
        #     diff = stage_position.r - stage.current_position.r
        #     logging.info(f"rotation angle is larger ({np.rad2deg(diff):.2f} deg) than desired, moving again.")
        #     stage.relative_move(StagePosition(r=diff), stage_settings)

        logging.info(f"safe movement complete.")

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
