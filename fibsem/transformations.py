import logging
from typing import Optional

import numpy as np

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition


def convert_milling_angle_to_stage_tilt(
    milling_angle: float, pretilt: float, column_tilt: float = np.deg2rad(52)
) -> float:
    """Convert the milling angle to the stage tilt angle, based on pretilt and column tilt.
        milling_angle = 90 - column_tilt + stage_tilt - pretilt
        stage_tilt = milling_angle - 90 + pretilt + column_tilt
    Args:
        milling_angle: milling angle (radians)
        pretilt: pretilt angle (radians)
        column_tilt: column tilt angle (radians)
    Returns:
        stage_tilt: stage tilt (radians)"""

    stage_tilt = milling_angle + column_tilt + pretilt - np.deg2rad(90)

    return stage_tilt


def convert_stage_tilt_to_milling_angle(
    stage_tilt: float, pretilt: float, column_tilt: float = np.deg2rad(52)
) -> float:
    """Convert the stage tilt angle to the milling angle, based on pretilt and column tilt.
        milling_angle = 90 - column_tilt + stage_tilt - pretilt
    Args:
        stage_tilt: stage tilt (radians)
        pretilt: pretilt angle (radians)
        column_tilt: column tilt angle (radians)
    Returns:
        milling_angle: milling angle (radians)"""

    milling_angle = np.deg2rad(90) - column_tilt + stage_tilt - pretilt

    return milling_angle


def get_stage_tilt_from_milling_angle(
    microscope: FibsemMicroscope, milling_angle: float
) -> float:
    """Get the stage tilt angle from the milling angle, based on pretilt and column tilt.
    Args:
        microscope (FibsemMicroscope): microscope connection
        milling_angle (float): milling angle (radians)
    Returns:
        float: stage tilt angle (radians)
    """
    pretilt = np.deg2rad(microscope.system.stage.shuttle_pre_tilt)
    column_tilt = np.deg2rad(microscope.system.ion.column_tilt)
    stage_tilt = convert_milling_angle_to_stage_tilt(
        milling_angle, pretilt, column_tilt
    )
    logging.debug(
        f"milling_angle: {np.rad2deg(milling_angle):.2f} deg, "
        f"pretilt: {np.rad2deg(pretilt)} deg, "
        f"stage_tilt: {np.rad2deg(stage_tilt):.2f} deg"
    )
    return stage_tilt

def is_close_to_milling_angle(
    microscope: FibsemMicroscope, milling_angle: float, atol: float = np.deg2rad(2)
) -> bool:
    """Check if the stage tilt is close to the milling angle, within a tolerance.
    Args:
        microscope (FibsemMicroscope): microscope connection
        milling_angle (float): milling angle (radians)
        atol (float): tolerance in radians
    Returns:
        bool: True if the stage tilt is within the tolerance of the milling angle
    """
    current_stage_tilt = microscope.get_stage_position().t
    pretilt = np.deg2rad(microscope.system.stage.shuttle_pre_tilt)
    column_tilt = np.deg2rad(microscope.system.ion.column_tilt)
    stage_tilt = convert_milling_angle_to_stage_tilt(
        milling_angle, pretilt=pretilt, column_tilt=column_tilt
    )
    logging.info(
        f"The current stage tilt is {np.rad2deg(stage_tilt):.2f} deg, "
        f"the stage tilt for the milling angle is {np.rad2deg(stage_tilt):.2f} deg"
    )
    return np.isclose(stage_tilt, current_stage_tilt, atol=atol)

# TODO: move inside the microscope class
def move_to_milling_angle(
    microscope: FibsemMicroscope,
    milling_angle: float,
    rotation: Optional[float] = None,
) -> bool:
    """Move the stage to the milling angle, based on the current pretilt and column tilt."""

    if rotation is None:
        rotation = microscope.system.stage.rotation_reference

    # calculate the stage tilt from the milling angle
    stage_tilt = get_stage_tilt_from_milling_angle(microscope, milling_angle)
    stage_position = FibsemStagePosition(t=stage_tilt, r=rotation)
    microscope.safe_absolute_stage_movement(stage_position)

    is_close = is_close_to_milling_angle(microscope, milling_angle)
    return is_close