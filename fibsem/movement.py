import time
import logging

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import (
    CoordinateSystem,
    ManipulatorCoordinateSystem,
    ManipulatorSavedPosition,
)
from autoscript_sdb_microscope_client.structures import (
    ManipulatorPosition,
    MoveSettings,
    StagePosition,
    AdornedImage,
)
from fibsem.acquire import BeamType


def pixel_to_realspace_coordinate(coord: list, image: AdornedImage) -> list:
    """Convert pixel image coordinate to real space coordinate.

    This conversion deliberately ignores the nominal pixel size in y,
    as this can lead to inaccuracies if the sample is not flat in y.

    Parameters
    ----------
    coord : listlike, float
        In x, y format & pixel units. Origin is at the top left.

    image : AdornedImage
        Image the coordinate came from.

        # do we have a sample image somewhere?
    Returns
    -------
    realspace_coord
        xy coordinate in real space. Origin is at the image center.
        Output is in (x, y) format.
    """
    coord = np.array(coord).astype(np.float64)
    if len(image.data.shape) > 2:
        y_shape, x_shape = image.data.shape[0:2]
    else:
        y_shape, x_shape = image.data.shape

    pixelsize_x = image.metadata.binary_result.pixel_size.x
    # deliberately don't use the y pixel size, any tilt will throw this off
    coord[1] = y_shape - coord[1]  # flip y-axis for relative coordinate system
    # reset origin to center
    coord -= np.array([x_shape / 2, y_shape / 2]).astype(np.int32)
    realspace_coord = list(np.array(coord) * pixelsize_x)  # to real space
    return realspace_coord # TODO: convert to use Point struct


def move_to_trenching_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """Tilt the sample stage to the correct angle for milling trenches.
    Assumes trenches should be milled with the sample surface flat to ion beam.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    settings: dict
        settings dictionary
    Returns
    -------
    autoscript_sdb_microscope_client.structures.StagePosition
        The position of the microscope stage after moving.
    """
    flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ION,
    )
    return microscope.specimen.stage.current_position


def move_to_liftout_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """Tilt the sample stage to the correct angle for liftout."""
    flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ELECTRON,
    )
    logging.info(f"move to liftout angle complete.")
    return microscope.specimen.stage.current_position


def move_to_landing_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """Tilt the sample stage to the correct angle for the landing posts."""

    landing_angle = np.deg2rad(settings["system"]["stage_tilt_landing"])
    flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ION,
    )  # stage tilt 25
    microscope.specimen.stage.relative_move(
        StagePosition(t=landing_angle)
    )  # more tilt by 13
    logging.info(
        f"movement: move to landing angle ({np.rad2deg(landing_angle)} deg) complete."
    )
    return microscope.specimen.stage.current_position

# TODO: change this to use set_microscope_state? 
def move_to_sample_grid(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """Move stage and zoom out to see the whole sample grid.
    Assumes sample grid is mounted on the left hand side of the holder.
    """

    sample_grid_center = StagePosition(
        x=float(settings["system"]["initial_position"]["sample_grid"]["x"]),
        y=float(settings["system"]["initial_position"]["sample_grid"]["y"]),
        z=float(settings["system"]["initial_position"]["sample_grid"]["z"]),
        r=np.deg2rad(float(settings["system"]["stage_rotation_flat_to_electron"])),
        coordinate_system=settings["system"]["initial_position"]["sample_grid"][
            "coordinate_system"
        ],
    )
    logging.info(f"movement: moving to sample grid {sample_grid_center}")
    safe_absolute_stage_movement(
        microscope=microscope, stage_position=sample_grid_center
    )

    # move flat to the electron beam
    flat_to_beam(
        microscope, settings=settings, beam_type=BeamType.ELECTRON,
    )
    logging.info(f"move to sample grid complete.")
    return microscope.specimen.stage.current_position


def move_to_landing_grid(
    microscope: SdbMicroscopeClient, settings: dict) -> StagePosition:
    """Move stage to landing post grid.
    Assumes the landing grid is mounted on the right hand side of the holder.
    Parameters
    ----------
    microscope : autoscript_sdb_microscope_client.SdbMicroscopeClient
        The AutoScript microscope object instance.
    flat_to_sem : bool, optional
        Whether to keep the landing post grid surface flat to the SEM.
    """

    # move to landing grid initial position
    landing_grid_position = StagePosition(
        x=float(settings["system"]["initial_position"]["landing_grid"]["x"]),
        y=float(settings["system"]["initial_position"]["landing_grid"]["y"]),
        z=float(settings["system"]["initial_position"]["landing_grid"]["z"]),
        r=np.deg2rad(float(settings["system"]["stage_rotation_flat_to_electron"])),
        coordinate_system=settings["system"]["initial_position"]["landing_grid"][
            "coordinate_system"
        ],
    )
    logging.info(f"movement: moving to landing grid {landing_grid_position}")
    safe_absolute_stage_movement(
        microscope=microscope, stage_position=landing_grid_position
    )

    # move to landing angle
    move_to_landing_angle(microscope, settings=settings)

    logging.info(f"movement: move to landing grid complete.")
    return microscope.specimen.stage.current_position


def move_sample_stage_out(microscope: SdbMicroscopeClient) -> StagePosition:
    """Move stage completely out of the way, so it is not visible at all."""
    # Must set tilt to zero, so we don't see reflections from metal stage base
    microscope.specimen.stage.absolute_move(StagePosition(t=0))  # important!
    sample_stage_out = StagePosition(
        x=-0.002507, y=0.025962792, z=0.0039559049
    )  # TODO: make these dynamically set based on initial_position
    logging.info(f"movement: move sample grid out to {sample_stage_out}")
    safe_absolute_stage_movement(microscope, sample_stage_out)
    logging.info(f"movement: move sample stage out complete.")
    return microscope.specimen.stage.current_position


def move_needle_to_liftout_position(
    microscope: SdbMicroscopeClient,
) -> ManipulatorPosition:
    """Move the needle into position, ready for liftout."""
    park_position = insert_needle(microscope)
    move_needle_closer(microscope)
    return park_position


def move_needle_to_landing_position(
    microscope: SdbMicroscopeClient,
) -> ManipulatorPosition:
    """Move the needle into position, ready for landing."""
    park_position = insert_needle(microscope)
    move_needle_closer(microscope, x_shift=-25e-6)
    return park_position


def insert_needle(microscope: SdbMicroscopeClient) -> ManipulatorPosition:
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
    park_position = needle.current_position
    logging.info(f"movement: inserted needle to {park_position}.")
    return park_position


def move_needle_closer(
    microscope: SdbMicroscopeClient, x_shift: float = -20e-6, z_shift: float = -160e-6, y_shift: float = 0.e-6
) -> ManipulatorPosition:
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
    x_move = x_corrected_needle_movement(x_shift) # TODO: replace with move_needle_relative...
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
    return needle.current_position


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
    from autoscript_sdb_microscope_client.structures import ManipulatorPosition

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


def retract_needle(
    microscope: SdbMicroscopeClient) -> ManipulatorPosition:
    """Retract the needle and multichem, preserving the correct park position.
    """

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
    logging.info(f"retracting needle to {park_position}") # TODO: replace with move_needle_relative...
    needle.relative_move(
        ManipulatorPosition(z=park_position.z - current_position.z)
    )  # noqa: E501
    needle.relative_move(
        ManipulatorPosition(y=park_position.y - current_position.y)
    )  # noqa: E501
    needle.relative_move(
        ManipulatorPosition(x=park_position.x - current_position.x) 
    )  # noqa: E501
    time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
    logging.info(f"retracting needle")
    needle.retract()
    retracted_position = needle.current_position
    logging.info(f"retract needle complete")
    return retracted_position


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
    stage_settings = MoveSettings(rotate_compucentric=True, tilt_compucentric=True)
    logging.info(f"movement: moving flat to {beam_type.name}")

    # TODO: check why we can't just use safe_absolute_stage_movement
    # I think it is because we want to maintain position, and only tilt/rotate
    # why isnt it the same as going to the same xyz with new rt?
    # TODO: if we use tilt_compucentric, and rotate_compucentric it should be!
    
    # If we rotating by a lot, tilt to zero so stage doesn't hit anything
    if (
        abs(np.rad2deg(rotation - stage.current_position.r)) % 360 > 90
    ):  
        stage.absolute_move(StagePosition(t=0), stage_settings)  # just in case
        logging.info(f"movement: tilting to flat for large rotation.")
    logging.info(f"movement: rotating stage to {rotation:.4f}")
    stage.absolute_move(StagePosition(r=rotation), stage_settings)
    logging.info(f"movement: tilting stage to {tilt:.4f}")
    stage.absolute_move(StagePosition(t=tilt), stage_settings)

    return stage.current_position

# TODO: use safe_absolute_stage_movement instead
def move_to_thinning_angle(
    microscope: SdbMicroscopeClient, settings: dict
) -> StagePosition:
    """ Rotate and tilt the stage to the thinning angle, assumes from the landing position"""
    stage = microscope.specimen.stage

    # tilt to zero for safety
    stage_settings = MoveSettings(rotate_compucentric=True, tilt_compucentric=True)
    stage.absolute_move(StagePosition(t=np.deg2rad(0)), stage_settings)

    # thinning position
    thinning_rotation_angle = np.deg2rad(settings["thin_lamella"]["rotation_angle"])
    thinning_tilt_angle = np.deg2rad(settings["thin_lamella"]["tilt_angle"])

    # rotate to thinning angle
    logging.info(f"rotate to thinning angle: {thinning_rotation_angle}")
    stage.absolute_move(StagePosition(r=thinning_rotation_angle), stage_settings)

    # tilt to thinning angle
    logging.info(f"tilt to thinning angle: {thinning_tilt_angle}")
    stage.absolute_move(StagePosition(t=thinning_tilt_angle), stage_settings)

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

    if beam_type == BeamType.ELECTRON:
        tilt_adjustment = -corrected_pretilt_angle
        SCALE_FACTOR = 0.78342  # patented technology
    elif beam_type == BeamType.ION:
        tilt_adjustment = -corrected_pretilt_angle - stage_tilt_flat_to_ion
        SCALE_FACTOR = 1.0

    # old
    if microscope is None:
        # tilt_radians = stage_tilt + tilt_adjustment
        tilt_radians = tilt_adjustment
        y_move = +np.cos(tilt_radians) * expected_y
        z_move = -np.sin(tilt_radians) * expected_y
    else:
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


# TODO: test out these functions...
def move_stage_relative_with_corrected_movement(microscope: SdbMicroscopeClient, settings: dict, dx: float, dy:float, beam_type: BeamType) -> None:
    """Calculate the corrected stage movements, and then move the stage relatively."""
    # dx, dy are in image coordinates

    stage = microscope.specimen.stage

    # calculate stage movement
    x_move = x_corrected_stage_movement(
        dx, stage_tilt=stage.current_position.t
    )
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

def move_stage_eucentric_correction(microscope: SdbMicroscopeClient, settings: dict, dy: float, beam_type: BeamType):
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
    z_move = StagePosition(z=yz_move.z)
    stage.relative_move(z_move)

    return 


def move_needle_relative_with_corrected_movement(microscope: SdbMicroscopeClient, settings: dict, dx: float, dy:float, beam_type: BeamType=  BeamType.ELECTRON) -> None:
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
        # z- is divided by cos... then multipled by cos.. no change?
        # calculate shift in xyz coordinates
        z_distance = dy / np.cos(stage_tilt) # TODO: needle to check this

        # TODO: this is used for land lamella
        # z_distance = -det.distance_metres.y / np.sin(
        #     np.deg2rad(settings["system"]["stage_tilt_flat_to_ion"])
        # )

        # Calculate movement
        x_move = x_corrected_needle_movement(expected_x=dx, stage_tilt=stage_tilt)
        yz_move = z_corrected_needle_movement(z_distance, stage_tilt)
    

    # move needle (relative)
    needle_position = ManipulatorPosition(x=x_move.x, y=yz_move.y, z=yz_move.z)
    logging.info(f"Moving needle: {needle_position}.")
    needle.relative_move(needle_position)



    return



def corrected_stage_movement_v2(microscope: SdbMicroscopeClient, 
    settings:dict, 
    dx:float = 0.0, dy: float = 0.0, 
    zy_link: bool = False) -> None:


    stage = microscope.specimen.stage

    # move settings
    move_settings = MoveSettings(link_z_y=zy_link, rotate_compucentric=True, tilt_compucentric=True)
    
    # move stage
    stage_position = StagePosition(x=dx, y=dy)
    logging.info(f"moving stage: {stage_position}")
    stage.relative_move(stage_position, settings=move_settings)