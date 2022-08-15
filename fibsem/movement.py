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
from fibsem.structures import BeamType


def insert_needle(microscope: SdbMicroscopeClient) -> None:
    """Insert the needle and return the needle parking position.
    Returns
    -------
    park_position : autoscript_sdb_microscope_client.structures.ManipulatorPosition
        The parking position for the needle manipulator when inserted.
    """
    needle = microscope.specimen.manipulator
    logging.info(f"movement: inserting needle to park position.")
    park_position = needle.get_saved_position(
        ManipulatorSavedPosition.PARK, ManipulatorCoordinateSystem.RAW
    )
    needle.insert(park_position)
    logging.info(f"movement: inserted needle to {park_position}.")


def insert_needle_v2(microscope: SdbMicroscopeClient, insert_position: ManipulatorSavedPosition = ManipulatorSavedPosition.PARK) -> None:
    """Insert the needle to the selected saved insert position.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        insert_position (ManipulatorSavedPosition, optional): saved needle position. Defaults to ManipulatorSavedPosition.PARK.
    """
    needle = microscope.specimen.manipulator
    # logging.info(f"inserting needle to {insert_position.explain} position.")
    insert_position = needle.get_saved_position(insert_position, ManipulatorCoordinateSystem.RAW
    )
    needle.insert(insert_position)
    logging.info(f"inserted needle to {insert_position}.")




def move_needle_closer(
    microscope: SdbMicroscopeClient,
    x_shift: float = -20e-6,
    z_shift: float = -160e-6,
    y_shift: float = 0.0e-6,
) -> None:
    """Move the needle closer to the sample surface, after inserting.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.sdb_microscope.SdbMicroscopeClient
        The Autoscript microscope object.
    x_shift : float
        Distance to move the needle from the parking position in x, in meters.
    z_shift : float
        Distance to move the needle towards the sample in z, in meters.
        Negative values move the needle TOWARDS the sample surface.
    """
    needle = microscope.specimen.manipulator
    stage = microscope.specimen.stage
    needle.set_default_coordinate_system(ManipulatorCoordinateSystem.STAGE)
    # Needle starts from the parking position (after inserting it)
    # Move the needle back a bit in x, so the needle is not overlapping target
    x_move = x_corrected_needle_movement(
        x_shift
    )  # TODO: replace with move_needle_relative...
    logging.info(f"movement: moving needle by {x_move}")
    needle.relative_move(x_move)

    y_move = y_corrected_needle_movement(y_shift, stage.current_position.t)
    logging.info(f"movement: moving needle by {y_move}")
    needle.relative_move(y_move)

    # Then move the needle towards the sample surface.
    z_move = z_corrected_needle_movement(z_shift, stage.current_position.t)
    logging.info(f"movement: moving needle by {z_move}")
    needle.relative_move(z_move)
    # The park position is always the same,
    # so the needletip will end up about 20 microns from the surface.
    logging.info(f"movement: move needle closer complete.")



def move_needle_to_eucentric_position_offset(microscope:SdbMicroscopeClient, dx: float = 0.0, dy: float = 0.0 , dz:float = 0.0) -> None:
# move to just above the eucentric point
    eucentric_position = microscope.specimen.manipulator.get_saved_position(
        ManipulatorSavedPosition.EUCENTRIC, ManipulatorCoordinateSystem.STAGE
    )
    yz_move = z_corrected_needle_movement(dz, microscope.specimen.stage.current_position.t)
    eucentric_position.x += dx
    eucentric_position.y += yz_move.y
    eucentric_position.z += yz_move.z  # RAW, up = negative, STAGE: down = negative
    microscope.specimen.manipulator.absolute_move(eucentric_position)




def x_corrected_needle_movement(
    expected_x: float, stage_tilt: float = None
) -> ManipulatorPosition:
    """Needle movement in X, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_x : float
        in meters
    Returns
    -------
    ManipulatorPosition
    """
    return ManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed


def y_corrected_needle_movement(
    expected_y: float, stage_tilt: float
) -> ManipulatorPosition:
    """Needle movement in Y, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_y : in meters
    stage_tilt : in radians
    Returns
    -------
    ManipulatorPosition
    """
    y_move = +np.cos(stage_tilt) * expected_y
    z_move = +np.sin(stage_tilt) * expected_y
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def z_corrected_needle_movement(
    expected_z: float, stage_tilt: float
) -> ManipulatorPosition:
    """Needle movement in Z, XTGui coordinates (Electron coordinate).
    Parameters
    ----------
    expected_z : in meters
    stage_tilt : in radians
    Returns
    -------
    ManipulatorPosition
    """
    y_move = -np.sin(stage_tilt) * expected_z
    z_move = +np.cos(stage_tilt) * expected_z
    return ManipulatorPosition(x=0, y=y_move, z=z_move)


def retract_needle(microscope: SdbMicroscopeClient) -> None:
    """Retract the needle and multichem, preserving the correct park position."""

    # Retract the multichem
    logging.info(f"retracting multichem")
    multichem = microscope.gas.get_multichem()
    multichem.retract()
    logging.info(f"retract multichem complete")
    # Retract the needle, preserving the correct parking postiion
    needle = microscope.specimen.manipulator
    current_position = needle.current_position
    park_position = needle.get_saved_position(
        ManipulatorSavedPosition.PARK, ManipulatorCoordinateSystem.RAW
    )

    # To prevent collisions with the sample; first retract in z, then y, then x
    logging.info(
        f"retracting needle to {park_position}"
    )  # TODO: replace with move_needle_relative...
    # needle.relative_move(
    #     ManipulatorPosition(z=park_position.z - current_position.z)
    # )  # noqa: E501
    # needle.relative_move(
    #     ManipulatorPosition(y=park_position.y - current_position.y)
    # )  # noqa: E501
    # needle.relative_move(
    #     ManipulatorPosition(x=park_position.x - current_position.x)
    # )  # noqa: E501
    needle.absolute_move(park_position)
    time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
    logging.info(f"retracting needle...")
    needle.retract()
    logging.info(f"retract needle complete")


def flat_to_beam(
    microscope: SdbMicroscopeClient,
    settings: dict,
    beam_type: BeamType = BeamType.ELECTRON,
) -> StagePosition:
    """Make the sample surface flat to the electron or ion beam."""

    stage = microscope.specimen.stage
    pretilt_angle = settings["system"]["pretilt_angle"]  # 27

    if beam_type is BeamType.ELECTRON:
        rotation = settings["system"]["stage_rotation_flat_to_electron"]
        tilt = np.deg2rad(pretilt_angle)
    if beam_type is BeamType.ION:
        rotation = settings["system"]["stage_rotation_flat_to_ion"]
        tilt = np.deg2rad(settings["system"]["stage_tilt_flat_to_ion"] - pretilt_angle)
    rotation = np.deg2rad(rotation)
    stage_settings = MoveSettings(rotate_compucentric=True)
    logging.info(f"movement: moving flat to {beam_type.name}")

    # TODO: check why we can't just use safe_absolute_stage_movement
    # I think it is because we want to maintain position, and only tilt/rotate
    # # why isnt it the same as going to the same xyz with new rt?
    # print("diff: ", abs(np.rad2deg(stage.current_position.r - rotation)))
    # print((np.rad2deg(rotation)) % 360 - np.rad2deg(stage.current_position.r) % 360)

    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if abs(np.rad2deg(rotation - stage.current_position.r)) % 360 > 90:
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
        logging.info(f"tilting to flat for large rotation.")
    logging.info(f"rotation: {rotation:.4f},  tilt: {tilt:.4f}")
    stage.absolute_move(StagePosition(r=rotation, t=tilt), stage_settings)

    return stage.current_position


def safe_absolute_stage_movement(
    microscope: SdbMicroscopeClient, stage_position: StagePosition
) -> StagePosition:
    """Move the stage to the desired position in a safe manner, using compucentric rotation.
    Supports movements in the stage_position coordinate system
    """

    stage = microscope.specimen.stage
    stage_settings = MoveSettings(rotate_compucentric=True)

    # tilt flat for large rotations to prevent collisions
    if abs(np.rad2deg(stage_position.r - stage.current_position.r)) % 360 > 90:
        stage.absolute_move(
            StagePosition(
                t=np.deg2rad(0), coordinate_system=stage_position.coordinate_system
            ),
            stage_settings,
        )
        logging.info(f"tilting to flat for large rotation.")
    stage.absolute_move(
        StagePosition(
            r=stage_position.r, coordinate_system=stage_position.coordinate_system
        ),
        stage_settings,
    )  # TODO: remove?
    logging.info(f"safe moving to {stage_position}")
    stage.absolute_move(stage_position, stage_settings)
    logging.info(f"safe movement complete.")
    return stage.current_position


def x_corrected_stage_movement(
    expected_x: float,
    settings: dict = None,
    stage_tilt: float = None,
    beam_type: BeamType = None,
) -> StagePosition:
    """Stage movement in X.
    ----------
    expected_x : in meters
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """
    return StagePosition(x=expected_x, y=0, z=0)


def y_corrected_stage_movement(
    microscope: SdbMicroscopeClient = None,
    settings: dict = None,
    expected_y: float = 0.0,
    beam_type: BeamType = BeamType.ELECTRON,
) -> StagePosition:
    """Stage movement in Y, corrected for tilt of sample surface plane.
    ----------
    expected_y : in meters
    stage_tilt : in radians        Can pass this directly microscope.specimen.stage.current_position.t
    beam_type : BeamType, optional
        BeamType.ELECTRON or BeamType.ION
    Returns
    -------
    StagePosition
        Stage position to pass to relative movement function.
    """

    # all angles in radians
    pretilt_angle = np.deg2rad(settings["system"]["pretilt_angle"])
    stage_tilt_flat_to_ion = np.deg2rad(settings["system"]["stage_tilt_flat_to_ion"])

    stage_rotation_flat_to_eb = np.deg2rad(
        settings["system"]["stage_rotation_flat_to_electron"]
    ) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(
        settings["system"]["stage_rotation_flat_to_ion"]
    ) % (2 * np.pi)
    stage_rotation = microscope.specimen.stage.current_position.r % (2 * np.pi)
    stage_tilt = microscope.specimen.stage.current_position.t

    # pretilt angle depends on rotation
    if np.isclose(stage_rotation, stage_rotation_flat_to_eb, atol=np.deg2rad(5)):
        PRETILT_SIGN = 1.0
    if np.isclose(stage_rotation, stage_rotation_flat_to_ion, atol=np.deg2rad(5)):
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * pretilt_angle

    # total tilt adjustment (difference between perspective view and sample coordinate system)
    if beam_type == BeamType.ELECTRON:
        tilt_adjustment = -corrected_pretilt_angle
        SCALE_FACTOR = 0.78342  # patented technology
    elif beam_type == BeamType.ION:
        tilt_adjustment = -corrected_pretilt_angle - stage_tilt_flat_to_ion
        SCALE_FACTOR = 1.0

    # new
    y_sample_move = (expected_y * SCALE_FACTOR) / np.cos(
        stage_tilt + tilt_adjustment
    )
    y_move = y_sample_move * np.cos(corrected_pretilt_angle)
    z_move = y_sample_move * np.sin(corrected_pretilt_angle)

    logging.info(f"rotation:  {microscope.specimen.stage.current_position.r} rad")
    logging.info(f"stage_tilt: {np.rad2deg(stage_tilt)}deg")
    logging.info(f"tilt_adjustment: {np.rad2deg(tilt_adjustment)}deg")
    logging.info(f"expected_y: {expected_y:.3e}m")
    logging.info(f"y_sample_move: {y_sample_move:.3e}m")
    logging.info(f"y-move: {y_move:.3e}m")
    logging.info(f"z-move: {z_move:.3e}m")

    logging.info(f"drift correction: the corrected Y shift is {y_move:.3e} meters")
    logging.info(f"drift correction: the corrected Z shift is  {z_move:.3e} meters")
    return StagePosition(x=0, y=y_move, z=z_move)


# TODO: z_corrected_stage_movement...?
# resetting working distance after vertical movements



def move_stage_relative_with_corrected_movement(
    microscope: SdbMicroscopeClient,
    settings: dict,
    dx: float,
    dy: float,
    beam_type: BeamType,
) -> None:
    """Calculate the corrected stage movements, and then move the stage relatively."""
    # dx, dy are in image coordinates

    stage = microscope.specimen.stage

    # calculate stage movement
    x_move = x_corrected_stage_movement(dx, stage_tilt=stage.current_position.t)
    yz_move = y_corrected_stage_movement(
        microscope=microscope,
        settings=settings,
        expected_y=dy,
        beam_type=beam_type,
    )

    # move stage
    stage_position = StagePosition(x=x_move.x, y=yz_move.y, z=yz_move.z)
    logging.info(f"moving stage: {stage_position}")
    stage.relative_move(stage_position)

    return


def move_stage_eucentric_correction(
    microscope: SdbMicroscopeClient, settings: dict, dy: float, beam_type: BeamType
):
    """Only move the stage in z"""

    # TODO: beam type should always be ION?
    # only move up/down in ion
    # then recentre in eb

    # TODO: finish this?
    stage = microscope.specimen.stage

    # calculate the correct movement?
    yz_move = y_corrected_stage_movement(
        microscope=microscope, settings=settings, expected_y=dy, beam_type=beam_type
    )

    # only move the z-component
    move_settings = MoveSettings(link_z_y=True)
    z_move = StagePosition(z=yz_move.z)
    stage.relative_move(z_move, move_settings)

    return


# TODO: redo with new knowledge about needle coordinate systems etc
def move_needle_relative_with_corrected_movement(
    microscope: SdbMicroscopeClient,
    settings: dict,
    dx: float,
    dy: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> None:
    """Calculate the corrected needle movements, and then move the needle relatively.

    moves in electron: x, y
    moves in ion: x, z

    """

    needle = microscope.specimen.manipulator
    stage_tilt = microscope.specimen.stage.current_position.t

    # xy
    if beam_type is BeamType.ELECTRON:
        x_move = x_corrected_needle_movement(dx, stage_tilt=stage_tilt)
        yz_move = y_corrected_needle_movement(dy, stage_tilt=stage_tilt)

    # xz,
    if beam_type is BeamType.ION:

        x_move = x_corrected_needle_movement(expected_x=dx, stage_tilt=stage_tilt)
        yz_move = z_corrected_needle_movement(expected_z=dy, stage_tilt=stage_tilt)

    # move needle (relative)
    needle_position = ManipulatorPosition(x=x_move.x, y=yz_move.y, z=yz_move.z)
    logging.info(f"Moving needle: {needle_position}.")
    needle.relative_move(needle_position)

    return


def corrected_stage_movement_v2(
    microscope: SdbMicroscopeClient,
    settings: dict,
    dx: float = 0.0,
    dy: float = 0.0,
    zy_link: bool = False,
) -> None:

    stage = microscope.specimen.stage

    # move settings
    move_settings = MoveSettings(
        link_z_y=zy_link, rotate_compucentric=True, tilt_compucentric=True
    )

    # move stage
    stage_position = StagePosition(x=dx, y=dy)
    logging.info(f"moving stage: {stage_position}")
    stage.relative_move(stage_position, settings=move_settings)
