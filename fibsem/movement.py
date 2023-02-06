import time
import logging

import numpy as np
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
from fibsem.structures import BeamType, MicroscopeSettings
from fibsem.microscope import FibsemMicroscope

# from fibsem.detection.detection import DetectionResult, FeatureType


############################## NEEDLE ##############################


def insert_needle(
    microscope: SdbMicroscopeClient,
    insert_position: ManipulatorSavedPosition = ManipulatorSavedPosition.PARK,
) -> None:
    """Insert the needle to the selected saved insert position.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        insert_position (ManipulatorSavedPosition, optional): saved needle position. Defaults to ManipulatorSavedPosition.PARK.
    """
    needle = microscope.specimen.manipulator
    # logging.info(f"inserting needle to {insert_position} position.")
    insert_position = needle.get_saved_position(
        insert_position, ManipulatorCoordinateSystem.RAW
    )
    needle.insert(insert_position)
    logging.info(f"inserted needle to {insert_position}.")


def retract_multichem(microscope: SdbMicroscopeClient) -> None:
    # Retract the multichem
    logging.info(f"retracting multichem")
    multichem = microscope.gas.get_multichem()
    multichem.retract()
    logging.info(f"retract multichem complete")

    return


def retract_needle(microscope: SdbMicroscopeClient) -> None:
    """Retract the needle and multichem, preserving the correct park position."""

    # retract multichem
    retract_multichem(microscope)

    # Retract the needle, preserving the correct parking postiion
    needle = microscope.specimen.manipulator
    park_position = needle.get_saved_position(
        ManipulatorSavedPosition.PARK, ManipulatorCoordinateSystem.RAW
    )

    logging.info(f"retracting needle to {park_position}")

    needle.absolute_move(park_position)
    time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
    logging.info(f"retracting needle...")
    needle.retract()
    logging.info(f"retract needle complete")


def move_needle_to_position_offset(
    microscope: SdbMicroscopeClient,
    position: ManipulatorPosition = None,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> None:
    """Move the needle to a position offset, based on the eucentric position

    Args:
        microscope (SdbMicroscopeClient): AutoScript Microscope Instance
        dx (float, optional): x-axis offset, image coordinates. Defaults to 0.0.
        dy (float, optional): y-axis offset, image coordinates. Defaults to 0.0.
        dz (float, optional): z-axis offset, image coordinates. Defaults to 0.0.
    """

    if position is None:
        # move to relative to the eucentric point
        position = microscope.specimen.manipulator.get_saved_position(
            ManipulatorSavedPosition.EUCENTRIC, ManipulatorCoordinateSystem.STAGE
        )

    yz_move = z_corrected_needle_movement(
        dz, microscope.specimen.stage.current_position.t
    )
    position.x += dx
    position.y += yz_move.y + dy
    position.z += yz_move.z  # RAW, up = negative, STAGE: down = negative
    position.r = None  # rotation is not supported
    microscope.specimen.manipulator.absolute_move(position)


def x_corrected_needle_movement(expected_x: float) -> ManipulatorPosition:
    """Calculate the corrected needle movement to move in the x-axis.

    Args:
        expected_x (float): distance along the x-axis (image coordinates)
    Returns:
        ManipulatorPosition: x-corrected needle movement (relative position)
    """
    return ManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed


def y_corrected_needle_movement(
    expected_y: float, stage_tilt: float
) -> ManipulatorPosition:
    """Calculate the corrected needle movement to move in the y-axis.

    Args:
        expected_y (float): distance along the y-axis (image coordinates)
        stage_tilt (float, optional): stage tilt.

    Returns:
        ManipulatorPosition: y-corrected needle movement (relative position)
    """
    y_move = +np.cos(stage_tilt) * expected_y
    z_move = +np.sin(stage_tilt) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def z_corrected_needle_movement(
    expected_z: float, stage_tilt: float
) -> ManipulatorPosition:
    """Calculate the corrected needle movement to move in the z-axis.

    Args:
        expected_z (float): distance along the z-axis (image coordinates)
        stage_tilt (float, optional): stage tilt.

    Returns:
        ManipulatorPosition: z-corrected needle movement (relative position)
    """
    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def move_needle_relative_with_corrected_movement(
    microscope: SdbMicroscopeClient,
    dx: float,
    dy: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> None:
    """Calculate the required corrected needle movements based on the BeamType to move in the desired image coordinates.
    Then move the needle relatively.

    BeamType.ELECTRON:  move in x, y (raw coordinates)
    BeamType.ION:       move in x, z (raw coordinates)

    Args:
        microscope (SdbMicroscopeClient): autoScript microscope instance
        dx (float): distance along the x-axis (image coordinates)
        dy (float): distance along the y-axis (image corodinates)
        beam_type (BeamType, optional): the beam type to move in. Defaults to BeamType.ELECTRON.
    """
    
    needle = microscope.specimen.manipulator
    stage_tilt = microscope.specimen.stage.current_position.t

    # xy
    if beam_type is BeamType.ELECTRON:
        x_move = x_corrected_needle_movement(expected_x=dx)
        yz_move = y_corrected_needle_movement(dy, stage_tilt=stage_tilt)

    # xz,
    if beam_type is BeamType.ION:

        x_move = x_corrected_needle_movement(expected_x=dx)
        yz_move = z_corrected_needle_movement(expected_z=dy, stage_tilt=stage_tilt)

    # move needle (relative)
    needle_position = ManipulatorPosition(x=x_move.x, y=yz_move.y, z=yz_move.z)
    logging.info(f"Moving needle: {needle_position}.")
    needle.relative_move(needle_position)

    return


############################## STAGE ##############################


def move_flat_to_beam(
    microscope: SdbMicroscopeClient,
    settings: MicroscopeSettings,
    beam_type: BeamType = BeamType.ELECTRON,
) -> None:
    """Make the sample surface flat to the electron or ion beam.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        settings (MicroscopeSettings): microscope settings
        beam_type (BeamType, optional): beam type to move flat to. Defaults to BeamType.ELECTRON.
    """

    stage_settings = settings.system.stage

    if beam_type is BeamType.ELECTRON:
        rotation = np.deg2rad(stage_settings.rotation_flat_to_electron)
        tilt = np.deg2rad(stage_settings.tilt_flat_to_electron)

    if beam_type is BeamType.ION:
        rotation = np.deg2rad(stage_settings.rotation_flat_to_ion)
        tilt = np.deg2rad(
            stage_settings.tilt_flat_to_ion - stage_settings.tilt_flat_to_electron
        )

    # updated safe rotation move
    logging.info(f"moving flat to {beam_type.name}")
    stage_position = StagePosition(r=rotation, t=tilt, coordinate_system="Raw")
    safe_absolute_stage_movement(microscope, stage_position)

    return


# TODO: make these consistenly use degrees or radians...
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

