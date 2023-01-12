from abc import ABC, abstractmethod
import copy
import logging
import datetime
import numpy as np
from fibsem.config import load_microscope_manufacturer
import sys

manufacturer = load_microscope_manufacturer()
if manufacturer == "Tescan":
    from tescanautomation import Automation
    from tescanautomation.SEM import HVBeamStatus as SEMStatus
    from tescanautomation.Common import Bpp

    # from tescanautomation.GUI import SEMInfobar
    import re

    # del globals()[tescanautomation.GUI]
    sys.modules.pop("tescanautomation.GUI")
    sys.modules.pop("tescanautomation.pyside6gui")
    sys.modules.pop("tescanautomation.pyside6gui.imageViewer_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_utils")
    sys.modules.pop("tescanautomation.pyside6gui.rc_GUI")
    sys.modules.pop("tescanautomation.pyside6gui.workflow_private")

if manufacturer == "Thermo":
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings
    from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

import sys
import fibsem.movement as move

from fibsem.structures import (
    BeamType,
    ImageSettings,
    Point,
    FibsemImage,
    FibsemImageMetadata,
    MicroscopeState,
    MicroscopeSettings,
    BeamSettings,
    FibsemStagePosition,
)


class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""

    @abstractmethod
    def connect_to_microscope(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def acquire_image(self):
        pass

    @abstractmethod
    def last_image(self):
        pass

    @abstractmethod
    def autocontrast(self):
        pass

    @abstractmethod
    def move_stage_absolute(self):
        pass

    @abstractmethod
    def move_stage_relative(self):
        pass

    @abstractmethod
    def eucentric_move(self):
        pass

class ThermoMicroscope(FibsemMicroscope):
    """ThermoFisher Microscope class, uses FibsemMicroscope as blueprint

    Args:
        FibsemMicroscope (ABC): abstract implementation
    """

    def __init__(self):
        self.connection = SdbMicroscopeClient()

    def disconnect(self):
        self.connection.disconnect()

    # @classmethod
    def connect_to_microscope(self, ip_address: str, port: int = 7520) -> None:
        """Connect to a Thermo Fisher microscope at a specified I.P. Address and Port

        Args:
            ip_address (str): I.P. Address of microscope
            port (int): port of microscope (default: 7520)
        """
        try:
            # TODO: get the port
            logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
            self.connection.connect(host=ip_address, port=port)
            logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {e}")

    def acquire_image(self, image_settings=ImageSettings) -> FibsemImage:
        """Acquire a new image.

        Args:
            settings (GrabFrameSettings, optional): frame grab settings. Defaults to None.
            beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.

        Returns:
            AdornedImage: new image
        """
        # set frame settings
        frame_settings = GrabFrameSettings(
            resolution=image_settings.resolution,
            dwell_time=image_settings.dwell_time,
            reduced_area=image_settings.reduced_area,
        )

        if image_settings.beam_type == BeamType.ELECTRON:
            hfw_limits = (
                self.connection.beams.electron_beam.horizontal_field_width.limits
            )
            image_settings.hfw = np.clip(
                image_settings.hfw, hfw_limits.min, hfw_limits.max
            )
            self.connection.beams.electron_beam.horizontal_field_width.value = (
                image_settings.hfw
            )

        if image_settings.beam_type == BeamType.ION:
            hfw_limits = self.connection.beams.ion_beam.horizontal_field_width.limits
            image_settings.hfw = np.clip(
                image_settings.hfw, hfw_limits.min, hfw_limits.max
            )
            self.connection.beams.ion_beam.horizontal_field_width.value = (
                image_settings.hfw
            )

        logging.info(f"acquiring new {image_settings.beam_type.name} image.")
        self.connection.imaging.set_active_view(image_settings.beam_type.value)
        self.connection.imaging.set_active_device(image_settings.beam_type.value)
        image = self.connection.imaging.grab_frame(frame_settings)

        state = self.get_current_microscope_state()

        fibsem_image = FibsemImage.fromAdornedImage(
            copy.deepcopy(image), copy.deepcopy(image_settings), copy.deepcopy(state)
        )

        return fibsem_image

    def last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage:
        """Get the last previously acquired image.

        Args:
            microscope (SdbMicroscopeClient):  autoscript microscope instance
            beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.

        Returns:
            AdornedImage: last image
        """

        self.connection.imaging.set_active_view(beam_type.value)
        self.connection.imaging.set_active_device(beam_type.value)
        image = self.connection.imaging.get_image()

        state = self.get_current_microscope_state()

        image_settings = FibsemImageMetadata.image_settings_from_adorned(
            image, beam_type
        )

        fibsem_image = FibsemImage.fromAdornedImage(image, image_settings, state)

        return fibsem_image

    def autocontrast(self, beam_type=BeamType.ELECTRON) -> None:
        """Automatically adjust the microscope image contrast."""
        self.connection.imaging.set_active_view(beam_type.value)
        self.connection.auto_functions.run_auto_cb()

    def reset_beam_shifts(self):
        """Set the beam shift to zero for the electron and ion beams

        Args:
            microscope (SdbMicroscopeClient): Autoscript microscope object
        """
        from autoscript_sdb_microscope_client.structures import Point

        # reset zero beamshift
        logging.debug(
            f"reseting ebeam shift to (0, 0) from: {self.connection.beams.electron_beam.beam_shift.value}"
        )
        self.connection.beams.electron_beam.beam_shift.value = Point(0, 0)
        logging.debug(
            f"reseting ibeam shift to (0, 0) from: {self.connection.beams.electron_beam.beam_shift.value}"
        )
        self.connection.beams.ion_beam.beam_shift.value = Point(0, 0)
        logging.debug(f"reset beam shifts to zero complete")

    def get_stage_position(self):
        self.connection.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.RAW
        )
        stage_position = self.connection.specimen.stage.current_position
        self.connection.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.SPECIMEN
        )
        return stage_position

    def get_current_microscope_state(self) -> MicroscopeState:
        """Get the current microscope state

        Returns:
            MicroscopeState: current microscope state
        """

        current_microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            # get absolute stage coordinates (RAW)
            absolute_position=self.get_stage_position(),
            # electron beam settings
            eb_settings=BeamSettings(
                beam_type=BeamType.ELECTRON,
                working_distance=self.connection.beams.electron_beam.working_distance.value,
                beam_current=self.connection.beams.electron_beam.beam_current.value,
                hfw=self.connection.beams.electron_beam.horizontal_field_width.value,
                resolution=self.connection.beams.electron_beam.scanning.resolution.value,
                dwell_time=self.connection.beams.electron_beam.scanning.dwell_time.value,
            ),
            # ion beam settings
            ib_settings=BeamSettings(
                beam_type=BeamType.ION,
                working_distance=self.connection.beams.ion_beam.working_distance.value,
                beam_current=self.connection.beams.ion_beam.beam_current.value,
                hfw=self.connection.beams.ion_beam.horizontal_field_width.value,
                resolution=self.connection.beams.ion_beam.scanning.resolution.value,
                dwell_time=self.connection.beams.ion_beam.scanning.dwell_time.value,
            ),
        )

        return current_microscope_state

    def move_stage_absolute(self, position: FibsemStagePosition):
        """Move the stage to the specified coordinates.

        Args:
            x (float): The x-coordinate to move to (in meters).
            y (float): The y-coordinate to move to (in meters).
            z (float): The z-coordinate to move to (in meters).
            r (float): The rotation to apply (in degrees).
            tx (float): The x-axis tilt to apply (in degrees).

        Returns:
            None
        """
        stage = self.connection.specimen.stage

        stage.absolute_move(
            position.x,
            position.y,
            position.z,
            position.r,
            position.t,
        )

    def move_stage_relative(
        self,
        dx: float = None,
        dy: float = None,
        dz: float = None,
        dr: float = None,
        dt: float = None,
    ):
        """Move the stage by the specified relative move.

        Args:
            x (float): The x-coordinate to move to (in meters).
            y (float): The y-coordinate to move to (in meters).
            z (float): The z-coordinate to move to (in meters).
            r (float): The rotation to apply (in degrees).
            tx (float): The x-axis tilt to apply (in degrees).

        Returns:
            None
        """

        stage = self.connection.specimen.stage
        new_position = FibsemStagePosition(dx, dy, dz, dr, dt, "raw")
        stage.relative_move(new_position)

    def eucentric_move(
        self,
        settings: MicroscopeSettings,
        dx: float,
        dy: float,
        beam_type: BeamType,
    ) -> None:
        """Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        Args:
            microscope (SdbMicroscopeClient): autoscript microscope instance
            settings (MicroscopeSettings): microscope settings
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
            beam_type (BeamType): beam type to move in
        """
        wd = self.connection.beams.electron_beam.working_distance.value

        # calculate stage movement
        x_move = x_corrected_stage_movement(dx)
        yz_move = y_corrected_stage_movement(
            microscope=self.connection,
            settings=settings,
            expected_y=dy,
            beam_type=beam_type,
        )

        # move stage
        stage_position = FibsemStagePosition(
            x=x_move.x * 1000, y=yz_move.y * 1000, z=yz_move.z * 1000
        )
        logging.info(f"moving stage ({beam_type.name}): {stage_position}")
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        self.connection.beams.electron_beam.working_distance.value = wd
        self.connection.specimen.stage.link() 

        return


class TescanMicroscope(FibsemMicroscope):
    """TESCAN Microscope class, uses FibsemMicroscope as blueprint

    Args:
        FibsemMicroscope (ABC): abstract implementation
    """

    def __init__(self, ip_address: str = "localhost"):
        self.connection = Automation(ip_address)
        detectors = self.connection.FIB.Detector.Enum()
        self.ion_detector_active = detectors[0]
        self.last_image_eb = None
        self.last_image_ib = None

    def disconnect(self):
        self.connection.Disconnect()

    # @classmethod
    def connect_to_microscope(self, ip_address: str, port: int = 8300) -> None:
        self.connection = Automation(ip_address, port)

    def acquire_image(self, image_settings=ImageSettings) -> FibsemImage:
        if image_settings.beam_type.value == 1:
            image = self._get_eb_image(image_settings)
            self.last_image_eb = image
        if image_settings.beam_type.value == 2:
            image = self._get_ib_image(image_settings)
            self.last_image_ib = image

        return image

    def _get_eb_image(self, image_settings=ImageSettings) -> FibsemImage:
        # At first make sure the beam is ON
        self.connection.SEM.Beam.On()
        # important: stop the scanning before we start scanning or before automatic procedures,
        # even before we configure the detectors
        self.connection.SEM.Scan.Stop()
        # Select the detector for image i.e.:
        # 1. assign the detector to a channel
        # 2. enable the channel for acquisition
        detector = self.connection.SEM.Detector.SESuitable()
        self.connection.SEM.Detector.Set(0, detector, Bpp.Grayscale_16_bit)

        # resolution
        numbers = re.findall(r"\d+", image_settings.resolution)
        imageWidth = int(numbers[0])
        imageHeight = int(numbers[1])

        self.connection.SEM.Optics.SetViewfield(image_settings.hfw * 1000)

        image = self.connection.SEM.Scan.AcquireImageFromChannel(
            0, imageWidth, imageHeight, 1000
        )

        microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            absolute_position=FibsemStagePosition(
                x=float(image.Header["SEM"]["StageX"]),
                y=float(image.Header["SEM"]["StageY"]),
                z=float(image.Header["SEM"]["StageZ"]),
                r=float(image.Header["SEM"]["StageRotation"]),
                t=float(image.Header["SEM"]["StageTilt"]),
                coordinate_system="Raw",
            ),
            eb_settings=BeamSettings(
                beam_type=BeamType.ELECTRON,
                working_distance=float(image.Header["SEM"]["WD"]),
                beam_current=float(image.Header["SEM"]["BeamCurrent"]),
                resolution="{}x{}".format(imageWidth, imageHeight),
                dwell_time=float(image.Header["SEM"]["DwellTime"]),
                stigmation=Point(
                    float(image.Header["SEM"]["StigmatorX"]),
                    float(image.Header["SEM"]["StigmatorY"]),
                ),
                shift=Point(
                    float(image.Header["SEM"]["ImageShiftX"]),
                    float(image.Header["SEM"]["ImageShiftY"]),
                ),
            ),
            ib_settings=BeamSettings(beam_type=BeamType.ION),
        )
        fibsem_image = FibsemImage.fromTescanImage(
            image, image_settings, microscope_state
        )

        return fibsem_image

    def _get_ib_image(self, image_settings=ImageSettings):
        # At first make sure the beam is ON
        self.connection.FIB.Beam.On()
        # important: stop the scanning before we start scanning or before automatic procedures,
        # even before we configure the detectors
        self.connection.FIB.Scan.Stop()
        # Select the detector for image i.e.:
        # 1. assign the detector to a channel
        # 2. enable the channel for acquisition
        self.connection.FIB.Detector.Set(
            0, self.ion_detector_active, Bpp.Grayscale_8_bit
        )

        # resolution
        numbers = re.findall(r"\d+", image_settings.resolution)
        imageWidth = int(numbers[0])
        imageHeight = int(numbers[1])

        self.connection.FIB.Optics.SetViewfield(image_settings.hfw * 1000)

        image = self.connection.FIB.Scan.AcquireImageFromChannel(
            0, imageWidth, imageHeight, 1000
        )

        microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            absolute_position=FibsemStagePosition(
                x=float(image.Header["FIB"]["StageX"]),
                y=float(image.Header["FIB"]["StageY"]),
                z=float(image.Header["FIB"]["StageZ"]),
                r=float(image.Header["FIB"]["StageRotation"]),
                t=float(image.Header["FIB"]["StageTilt"]),
                coordinate_system="Raw",
            ),
            eb_settings=BeamSettings(beam_type=BeamType.ELECTRON),
            ib_settings=BeamSettings(
                beam_type=BeamType.ION,
                working_distance=float(image.Header["FIB"]["WD"]),
                beam_current=float(image.Header["FIB"]["BeamCurrent"]),
                resolution="{}x{}".format(imageWidth, imageHeight),
                dwell_time=float(image.Header["FIB"]["DwellTime"]),
                stigmation=Point(
                    float(image.Header["FIB"]["StigmatorX"]),
                    float(image.Header["FIB"]["StigmatorY"]),
                ),
                shift=Point(
                    float(image.Header["FIB"]["ImageShiftX"]),
                    float(image.Header["FIB"]["ImageShiftY"]),
                ),
            ),
        )

        fibsem_image = FibsemImage.fromTescanImage(
            image, image_settings, microscope_state
        )

        return fibsem_image

    def last_image(self, beam_type: BeamType.ELECTRON) -> FibsemImage:
        if beam_type == BeamType.ELECTRON:
            image = self.last_image_eb
        elif beam_type == BeamType.ION:
            image = self.last_image_ib
        else:
            raise Exception("Beam type error")
        return image

    def get_stage_position(self):
        x, y, z, r, t = self.connection.Stage.GetPosition()
        stage_position = FibsemStagePosition(x, y, z, r, t, "raw")
        return stage_position

    def get_current_microscope_state(self) -> MicroscopeState:
        """Get the current microscope state

        Returns:
            MicroscopeState: current microscope state
        """
        image_eb = self.last_image(BeamType.ELECTRON)
        image_ib = self.last_image(BeamType.ION)

        if image_ib is not None:
            ib_settings = (
                BeamSettings(
                    beam_type=BeamType.ION,
                    working_distance=image_ib.metadata.microscope_state.ib_settings.working_distance,
                    beam_current=self.connection.FIB.Beam.ReadProbeCurrent() / (10e12),
                    hfw=self.connection.FIB.Optics.GetViewfield() / 1000,
                    resolution=image_ib.metadata.image_settings.resolution,
                    dwell_time=image_ib.metadata.image_settings.dwell_time,
                    stigmation=image_ib.metadata.microscope_state.ib_settings.stigmation,
                    shift=image_ib.metadata.microscope_state.ib_settings.shift,
                ),
            )
        else:
            ib_settings = BeamSettings(BeamType.ION)

        if image_eb is not None:
            eb_settings = BeamSettings(
                beam_type=BeamType.ELECTRON,
                working_distance=self.connection.SEM.Optics.GetWD() / 1000,
                beam_current=self.connection.SEM.Beam.GetCurrent() / (10e6),
                hfw=self.connection.SEM.Optics.GetViewfield() / 1000,
                resolution=image_eb.metadata.image_settings.resolution,  # TODO fix these empty parameters
                dwell_time=image_eb.metadata.image_settings.dwell_time,
                stigmation=image_eb.metadata.microscope_state.eb_settings.stigmation,
                shift=image_eb.metadata.microscope_state.eb_settings.shift,
            )
        else:
            eb_settings = BeamSettings(BeamType.ELECTRON)

        current_microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            # get absolute stage coordinates (RAW)
            absolute_position=self.get_stage_position(),
            # electron beam settings
            eb_settings=eb_settings,
            # ion beam settings
            ib_settings=ib_settings,
        )

        return current_microscope_state

    def autocontrast(self, beam_type: BeamType) -> None:
        if beam_type.name == BeamType.ELECTRON:
            self.connection.SEM.Detector.StartAutoSignal(0)
        if beam_type.name == BeamType.ION:
            self.connection.FIB.Detector.AutoSignal(0)

    def reset_beam_shifts(self):
        pass

    def move_stage_absolute(self, position: FibsemStagePosition):
        """Move the stage to the specified coordinates.

        Args:
            x (float): The x-coordinate to move to (in meters).
            y (float): The y-coordinate to move to (in meters).
            z (float): The z-coordinate to move to (in meters).
            r (float): The rotation to apply (in degrees).
            tx (float): The x-axis tilt to apply (in degrees).

        Returns:
            None
        """
        self.connection.Stage.MoveTo(
            position.x * 1000,
            position.y * 1000,
            position.z * 1000,
            position.r,
            position.t,
        )

    def move_stage_relative(
        self,
        dx: float = None,
        dy: float = None,
        dz: float = None,
        dr: float = None,
        dtx: float = None,
    ):
        """Move the stage by the specified relative move.

        Args:
            x (float): The x-coordinate to move to (in meters).
            y (float): The y-coordinate to move to (in meters).
            z (float): The z-coordinate to move to (in meters).
            r (float): The rotation to apply (in degrees).
            tx (float): The x-axis tilt to apply (in degrees).

        Returns:
            None
        """

        current_position = self.get_stage_position()
        new_position = FibsemStagePosition(
            current_position.x + dx * 1000,
            current_position.y + dy * 1000,
            current_position.z + dz * 1000,
            current_position.r + dr,
            current_position.t + dtx,
            "raw",
        )
        self.move_stage_absolute(new_position)

    def eucentric_move(
        self,
        settings: MicroscopeSettings,
        dx: float,
        dy: float,
        beam_type: BeamType,
    ) -> None:
        """Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        Args:
            microscope (Tescan Automation): Tescan microscope instance
            settings (MicroscopeSettings): microscope settings
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
        """
        wd = self.connection.SEM.Optics.GetWD()

        # calculate stage movement
        x_move = x_corrected_stage_movement(dx)
        yz_move = y_corrected_stage_movement(
            microscope=self.connection,
            settings=settings,
            expected_y=dy,
            beam_type=beam_type,
        )

        # move stage
        stage_position = FibsemStagePosition(
            x=x_move.x * 1000, y=yz_move.y * 1000, z=yz_move.z * 1000
        )
        logging.info(f"moving stage ({beam_type.name}): {stage_position}")
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        self.connection.SEM.Optics.SetWD(wd)
        # self.connection.specimen.stage.link() # TODO how to link for TESCAN?

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

def x_corrected_stage_movement(
    expected_x: float,
) -> FibsemStagePosition:
    """Calculate the x corrected stage movement.

    Args:
        expected_x (float): distance along x-axis

    Returns:
        StagePosition: x corrected stage movement (relative position)
    """
    return FibsemStagePosition(x=expected_x, y=0, z=0)


def y_corrected_stage_movement(
    self,
    settings: MicroscopeSettings,
    expected_y: float,
    beam_type: BeamType = BeamType.ELECTRON,
) -> FibsemStagePosition:
    """Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).

    Args:
        microscope (SdbMicroscopeClient, optional): autoscript microscope instance
        settings (MicroscopeSettings): microscope settings
        expected_y (float, optional): distance along y-axis.
        beam_type (BeamType, optional): beam_type to move in. Defaults to BeamType.ELECTRON.

    Returns:
        StagePosition: y corrected stage movement (relative position)
    """

    # TODO: replace with camera matrix * inverse kinematics
    # TODO: replace stage_tilt_flat_to_electron with pre-tilt

    # all angles in radians
    stage_tilt_flat_to_electron = np.deg2rad(
        settings.system.stage.tilt_flat_to_electron
    )
    stage_tilt_flat_to_ion = np.deg2rad(settings.system.stage.tilt_flat_to_ion)

    stage_rotation_flat_to_eb = np.deg2rad(
        settings.system.stage.rotation_flat_to_electron
    ) % (2 * np.pi)
    stage_rotation_flat_to_ion = np.deg2rad(
        settings.system.stage.rotation_flat_to_ion
    ) % (2 * np.pi)

    # current stage position
    current_stage_position = self.connection.get_stage_position()
    stage_rotation = current_stage_position.r % (2 * np.pi)
    stage_tilt = current_stage_position.t

    PRETILT_SIGN = 1.0
    # pretilt angle depends on rotation
    if rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
        PRETILT_SIGN = 1.0
    if rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
        PRETILT_SIGN = -1.0

    corrected_pretilt_angle = PRETILT_SIGN * stage_tilt_flat_to_electron

    # perspective tilt adjustment (difference between perspective view and sample coordinate system)
    if beam_type == BeamType.ELECTRON:
        perspective_tilt_adjustment = -corrected_pretilt_angle
        SCALE_FACTOR = 1.0  # 0.78342  # patented technology
    elif beam_type == BeamType.ION:
        perspective_tilt_adjustment = -corrected_pretilt_angle - stage_tilt_flat_to_ion
        SCALE_FACTOR = 1.0

    # the amount the sample has to move in the y-axis
    y_sample_move = (expected_y * SCALE_FACTOR) / np.cos(
        stage_tilt + perspective_tilt_adjustment
    )

    # the amount the stage has to move in each axis
    y_move = y_sample_move * np.cos(corrected_pretilt_angle)
    z_move = y_sample_move * np.sin(corrected_pretilt_angle)

    return FibsemStagePosition(x=0, y=y_move, z=z_move)