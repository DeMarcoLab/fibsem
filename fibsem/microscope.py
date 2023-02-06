from abc import ABC, abstractmethod
import copy
import logging
from copy import deepcopy
import datetime
import numpy as np
from fibsem.config import load_microscope_manufacturer
import sys
import time


# for easier usage
import fibsem.constants as constants
from typing import Union


manufacturer = load_microscope_manufacturer()
if manufacturer == "Tescan":
    from tescanautomation import Automation
    from tescanautomation.SEM import HVBeamStatus as SEMStatus
    from tescanautomation.Common import Bpp
    from tescanautomation.DrawBeam import IEtching
    from tescanautomation.DrawBeam import IEtching
    from tescanautomation.DrawBeam import Status as DBStatus


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
    sys.modules.pop("PySide6.QtCore")

if manufacturer == "Thermo":
    from autoscript_sdb_microscope_client.structures import (
        GrabFrameSettings,
        MoveSettings,
    )
    from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client._dynamic_object_proxies import (
        RectanglePattern,
        CleaningCrossSectionPattern,
    )

import sys


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
    FibsemMillingSettings,
    FibsemPatternSettings,
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
    def reset_beam_shifts(self):
        pass

    @abstractmethod
    def beam_shift(self):
        pass

    @abstractmethod
    def move_stage_absolute(self):
        pass

    @abstractmethod
    def move_stage_relative(self):
        pass

    @abstractmethod
    def stable_move(self):
        pass

    @abstractmethod
    def eucentric_move(self):
        pass

    @abstractmethod
    def move_flat_to_beam(self):
        pass

    @abstractmethod
    def setup_milling(self):
        pass

    @abstractmethod
    def run_milling(self):
        pass

    @abstractmethod
    def finish_milling(self):
        pass

    @abstractmethod
    def draw_rectangle(self):
        pass

    @abstractmethod
    def draw_line(self):
        pass

    @abstractmethod
    def set_microscope_state(self):
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

    def acquire_image(self, image_settings:ImageSettings) -> FibsemImage:
        """Acquire a new image.

        Args:
            settings (GrabFrameSettings, optional): frame grab settings. Defaults to None.
            beam_type (BeamType, optional): imaging beam type. Defaults to BeamType.ELECTRON.

        Returns:
            AdornedImage: new image
        """
        # set frame settings
        if image_settings.reduced_area is not None:
            image_settings.reduced_area = image_settings.reduced_area.__to_FEI__()
        
        frame_settings = GrabFrameSettings(
            resolution=f"{image_settings.resolution[0]}x{image_settings.resolution[1]}",
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
            self (FibsemMicroscope): Fibsem microscope object
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

    def beam_shift(self, dx: float, dy: float):
        '''Adjusts the beam shift based on relative values that are provided.
        
        Args:
            self (FibsemMicroscope): Fibsem microscope object
            dx (float): the relative x term
            dy (float): the relative y term
        '''
        self.connection.beams.ion_beam.beam_shift.value += (-dx, dy)

    def get_stage_position(self):
        self.connection.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.RAW
        )
        stage_position = self.connection.specimen.stage.current_position
        print(stage_position)
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
            r (float): The rotation to apply (in radians).
            tx (float): The x-axis tilt to apply (in radians).

        Returns:
            None
        """
        stage = self.connection.specimen.stage
        thermo_position = position.to_autoscript_position()
        thermo_position.coordinate_system = CoordinateSystem.RAW
        stage.absolute_move(thermo_position)

    def move_stage_relative(
        self,
        position: FibsemStagePosition,
    ):
        """Move the stage by the specified relative move.

        Args:
            x (float): The x-coordinate to move to (in meters).
            y (float): The y-coordinate to move to (in meters).
            z (float): The z-coordinate to move to (in meters).
            r (float): The rotation to apply (in radians).
            tx (float): The x-axis tilt to apply (in radians).

        Returns:
            None
        """
        stage = self.connection.specimen.stage
        thermo_position = position.to_autoscript_position()
        thermo_position.coordinate_system = CoordinateSystem.RAW
        stage.relative_move(thermo_position)

    def stable_move(
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
        x_move = self.x_corrected_stage_movement(dx)
        yz_move = self.y_corrected_stage_movement(
            settings=settings,
            expected_y=dy,
            beam_type=beam_type,
        )

        # move stage
        stage_position = FibsemStagePosition(
            x=x_move.x,
            y=yz_move.y,
            z=yz_move.z,
            r=0,
            t=0,
            coordinate_system="raw",
        )
        logging.info(f"moving stage ({beam_type.name}): {stage_position}")
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        self.connection.beams.electron_beam.working_distance.value = wd
        self.connection.specimen.stage.link()

        return

    def eucentric_move(
        self,
        settings: MicroscopeSettings,
        dy: float,
        static_wd: bool = True,
    ) -> None:
        """Move the stage vertically to correct eucentric point

        Args:
            microscope (SdbMicroscopeClient): autoscript microscope instance
            dy (float): distance in y-axis (image coordinates)
        """
        wd = self.connection.beams.electron_beam.working_distance.value

        z_move = dy / np.cos(np.deg2rad(90 - settings.system.stage.tilt_flat_to_ion))  # TODO: MAGIC NUMBER, 90 - fib tilt

        move_settings = MoveSettings(link_z_y=True)
        z_move = FibsemStagePosition(
            z=z_move, coordinate_system="Specimen"
        ).to_autoscript_position()
        self.connection.specimen.stage.relative_move(z_move, move_settings)
        logging.info(f"eucentric movement: {z_move}")

        if static_wd:
            self.connection.beams.electron_beam.working_distance.value = (
                settings.system.electron.eucentric_height
            )
            self.connection.beams.ion_beam.working_distance.value = (
                settings.system.ion.eucentric_height
            )
        else:
            self.connection.beams.electron_beam.working_distance.value = wd
        self.connection.specimen.stage.link()

    def x_corrected_stage_movement(
        self,
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
        current_stage_position = self.get_stage_position()
        stage_rotation = current_stage_position.r % (2 * np.pi)
        stage_tilt = current_stage_position.t

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        if rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
            PRETILT_SIGN = 1.0
        if rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        corrected_pretilt_angle = PRETILT_SIGN * stage_tilt_flat_to_electron

        # perspective tilt adjustment (difference between perspective view and sample coordinate system)
        if beam_type == BeamType.ELECTRON:
            perspective_tilt_adjustment = -corrected_pretilt_angle
            SCALE_FACTOR = 1.0  # 0.78342  # patented technology
        elif beam_type == BeamType.ION:
            perspective_tilt_adjustment = (
                -corrected_pretilt_angle - stage_tilt_flat_to_ion
            )
            SCALE_FACTOR = 1.0

        # the amount the sample has to move in the y-axis
        y_sample_move = (expected_y * SCALE_FACTOR) / np.cos(
            stage_tilt + perspective_tilt_adjustment
        )

        # the amount the stage has to move in each axis
        y_move = y_sample_move * np.cos(corrected_pretilt_angle)
        z_move = y_sample_move * np.sin(corrected_pretilt_angle)

        return FibsemStagePosition(x=0, y=y_move, z=z_move)

    def move_flat_to_beam(
        self, settings: MicroscopeSettings, beam_type: BeamType = BeamType.ELECTRON
    ):

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

        position = self.get_stage_position()

        # updated safe rotation move
        logging.info(f"moving flat to {beam_type.name}")
        stage_position = FibsemStagePosition(x = position.x, y = position.y, z=position.z, r=rotation, t=tilt)
        self.move_stage_absolute(stage_position)

    def setup_milling(
        self,
        application_file: str,
        patterning_mode: str,
        hfw: float,
        mill_settings: FibsemMillingSettings,
    ):
        self.connection.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
        self.connection.imaging.set_active_device(BeamType.ION.value)
        self.connection.patterning.set_default_beam_type(
            BeamType.ION.value
        )  # ion beam default
        self.connection.patterning.set_default_application_file(application_file)
        self.connection.patterning.mode = patterning_mode
        self.connection.patterning.clear_patterns()  # clear any existing patterns
        self.connection.beams.ion_beam.horizontal_field_width.value = hfw

    def run_milling(self, milling_current: float, asynch: bool = False):
        # change to milling current
        self.connection.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
        if self.connection.beams.ion_beam.beam_current.value != milling_current:
            logging.info(f"changing to milling current: {milling_current:.2e}")
            self.connection.beams.ion_beam.beam_current.value = milling_current

        # run milling (asynchronously)
        logging.info(f"running ion beam milling now... asynchronous={asynch}")
        if asynch:
            self.connection.patterning.start()
        else:
            self.connection.patterning.run()
            self.connection.patterning.clear_patterns()
        # NOTE: Make tescan logs the same??

    def finish_milling(self, imaging_current: float):
        self.connection.patterning.clear_patterns()
        self.connection.beams.ion_beam.beam_current.value = imaging_current
        self.connection.patterning.mode = "Serial"

    def draw_rectangle(
        self,
        pattern_settings: FibsemPatternSettings,
        mill_settings: FibsemMillingSettings,
    ):

        if pattern_settings.cleaning_cross_section:
            pattern = self.connection.patterning.create_cleaning_cross_section(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                width=pattern_settings.width,
                height=pattern_settings.height,
                depth=pattern_settings.depth,
            )
        else:
            pattern = self.connection.patterning.create_rectangle(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                width=pattern_settings.width,
                height=pattern_settings.height,
                depth=pattern_settings.depth,
            )

        pattern.rotation = pattern_settings.rotation
        pattern.scan_direction = pattern_settings.scan_direction

        return pattern

    def draw_line(self, pattern_settings: FibsemPatternSettings):
        pattern = self.connection.patterning.create_line(
            start_x=pattern_settings.start_x,
            start_y=pattern_settings.start_y,
            end_x=pattern_settings.end_x,
            end_y=pattern_settings.end_y,
            depth=pattern_settings.depth,
        )

        return pattern

    def set_microscope_state(self, microscope_state: MicroscopeState):
        """Reset the microscope state to the provided state"""

        logging.info(f"restoring microscope state...")

        # move to position
        self.move_stage_absolute(stage_position=microscope_state.absolute_position)

        # restore electron beam
        logging.info(f"restoring electron beam settings...")
        self.connection.beams.electron_beam.working_distance.value = (
            microscope_state.eb_settings.working_distance
        )
        self.connection.beams.electron_beam.beam_current.value = (
            microscope_state.eb_settings.beam_current
        )
        self.connection.beams.electron_beam.horizontal_field_width.value = (
            microscope_state.eb_settings.hfw
        )
        self.connection.beams.electron_beam.scanning.resolution.value = (
            microscope_state.eb_settings.resolution
        )
        self.connection.beams.electron_beam.scanning.dwell_time.value = (
            microscope_state.eb_settings.dwell_time
        )
        # microscope.beams.electron_beam.stigmator.value = (
        #     microscope_state.eb_settings.stigmation
        # )

        # restore ion beam
        logging.info(f"restoring ion beam settings...")
        self.connection.beams.ion_beam.working_distance.value = (
            microscope_state.ib_settings.working_distance
        )
        self.connection.beams.ion_beam.beam_current.value = (
            microscope_state.ib_settings.beam_current
        )
        self.connection.beams.ion_beam.horizontal_field_width.value = (
            microscope_state.ib_settings.hfw
        )
        self.connection.beams.ion_beam.scanning.resolution.value = (
            microscope_state.ib_settings.resolution
        )
        self.connection.beams.ion_beam.scanning.dwell_time.value = (
            microscope_state.ib_settings.dwell_time
        )
        # microscope.beams.ion_beam.stigmator.value = microscope_state.ib_settings.stigmation

        self.connection.specimen.stage.link()
        logging.info(f"microscope state restored")
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

        if image_settings.beam_type.name == "ELECTRON":
            image = self._get_eb_image(image_settings)
            self.last_image_eb = image
        if image_settings.beam_type.name == "ION":
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
        self.connection.SEM.Detector.Set(0, detector, Bpp.Grayscale_8_bit)

        dwell_time = image_settings.dwell_time * constants.SI_TO_NANO
        # resolution
        imageWidth = image_settings.resolution[0]
        imageHeight = image_settings.resolution[1]

        self.connection.SEM.Optics.SetViewfield(
            image_settings.hfw * constants.METRE_TO_MILLIMETRE
        )

        if image_settings.reduced_area is not None:
            image = self.connection.SEM.Scan.AcquireROIFromChannel(
                Channel= 0,
                Width= image_settings.reduced_area.width,
                Height= image_settings.reduced_area.height,
                Left= image_settings.reduced_area.left,
                Top= image_settings.reduced_area.top,
                Right= image_settings.reduced_area.left - image_settings.reduced_area.width - 1,
                Bottom= image_settings.reduced_area.top - image_settings.reduced_area.height - 1,
                DwellTime= dwell_time
            )
        else:
            image = self.connection.SEM.Scan.AcquireImageFromChannel(
                0, imageWidth, imageHeight, dwell_time
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
                resolution=(imageWidth, imageHeight), #"{}x{}".format(imageWidth, imageHeight),
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
            image, deepcopy(image_settings), microscope_state
        )

        fibsem_image.metadata.image_settings.resolution = (imageWidth, imageHeight)

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

        dwell_time = image_settings.dwell_time * constants.SI_TO_NANO

        # resolution
        imageWidth = image_settings.resolution[0]
        imageHeight = image_settings.resolution[1]

        self.connection.FIB.Optics.SetViewfield(
            image_settings.hfw * constants.METRE_TO_MILLIMETRE
        )

        if image_settings.reduced_area is not None:
            image = self.connection.FIB.Scan.AcquireROIFromChannel(
                Channel= 0,
                Width= image_settings.reduced_area.width,
                Height= image_settings.reduced_area.height,
                Left= image_settings.reduced_area.left,
                Top= image_settings.reduced_area.top,
                Right= image_settings.reduced_area.left - image_settings.reduced_area.width - 1,
                Bottom= image_settings.reduced_area.top - image_settings.reduced_area.height - 1,
                DwellTime= dwell_time
            )
        else:
            image = self.connection.FIB.Scan.AcquireImageFromChannel(
                0, imageWidth, imageHeight, dwell_time
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
                resolution=(imageWidth, imageHeight), #"{}x{}".format(imageWidth, imageHeight),
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
            image, deepcopy(image_settings), microscope_state
        )

        fibsem_image.metadata.image_settings.resolution = (imageWidth, imageHeight)

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
        stage_position = FibsemStagePosition(
            x * constants.MILLIMETRE_TO_METRE,
            y * constants.MILLIMETRE_TO_METRE,
            z * constants.MILLIMETRE_TO_METRE,
            r * constants.DEGREES_TO_RADIANS,
            t * constants.DEGREES_TO_RADIANS,
            "raw",
        )
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
                    beam_current=self.connection.FIB.Beam.ReadProbeCurrent()
                    * constants.PICO_TO_SI,
                    hfw=self.connection.FIB.Optics.GetViewfield()
                    * constants.MILLIMETRE_TO_METRE,
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
                working_distance=self.connection.SEM.Optics.GetWD()
                * constants.MILLIMETRE_TO_METRE,
                beam_current=self.connection.SEM.Beam.GetCurrent()
                * constants.MICRO_TO_SI,
                hfw=self.connection.SEM.Optics.GetViewfield()
                * constants.MILLIMETRE_TO_METRE,
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
        """
        Resets the beam shifts back to default.
        """
        self.connection.FIB.Optics.SetImageShift(0, 0)
        self.connection.SEM.Optics.SetImageShift(0, 0)

    def beam_shift(self, dx: float, dy: float):
        """
        Relative shift of ION Beam. The inputs dx and dy are in metres as that is the OpenFIBSEM standard, however TESCAN uses mm so conversions must be made. 
        """
        x, y = self.connection.FIB.Optics.GetImageShift()
        dx *=  constants.METRE_TO_MILLIMETRE # Convert to mm from m.
        dy *=  constants.METRE_TO_MILLIMETRE
        x -= dx # NOTE: Not sure why the dx is -dx, this may be thermo specific and doesn't apply to TESCAN?
        y += dy
        self.connection.FIB.Optics.SetImageShift(x,y) 
        

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
            position.x * constants.METRE_TO_MILLIMETRE,
            position.y * constants.METRE_TO_MILLIMETRE,
            position.z * constants.METRE_TO_MILLIMETRE,
            position.r * constants.RADIANS_TO_DEGREES,
            position.t * constants.RADIANS_TO_DEGREES,
        )

    def move_stage_relative(
        self,
        position: FibsemStagePosition,
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
        x_m = current_position.x
        y_m = current_position.y
        z_m = current_position.z
        new_position = FibsemStagePosition(
            x_m + position.x,
            y_m + position.y,
            z_m + position.z,
            current_position.r + position.r,
            current_position.t + position.t,
            "raw",
        )
        self.move_stage_absolute(new_position)

    def stable_move(
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
        x_move = self.x_corrected_stage_movement(dx)
        yz_move = self.y_corrected_stage_movement(
            settings=settings,
            expected_y=dy,
            beam_type=beam_type,
        )

        # move stage
        stage_position = FibsemStagePosition(
            x=x_move.x, y=yz_move.y, z=yz_move.z, r=0, t=0
        )
        logging.info(f"moving stage ({beam_type.name}): {stage_position}")
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        self.connection.SEM.Optics.SetWD(wd)
        # self.connection.specimen.stage.link() # TODO how to link for TESCAN?

        return

    def eucentric_move(
        self,
        settings: MicroscopeSettings,
        dy: float,
        static_wd: bool = True,
    ) -> None:
        """Move the stage vertically to correct eucentric point

        Args:
            microscope (SdbMicroscopeClient): autoscript microscope instance
            dy (float): distance in y-axis (image coordinates)
        """
        wd = self.connection.SEM.Optics.GetWD()

        z_move = dy / np.cos(
            np.deg2rad(90 - settings.system.stage.tilt_flat_to_ion)
        )  # TODO: MAGIC NUMBER, 90 - fib tilt

        z_move = FibsemStagePosition(x=0, y=0, z=z_move, r=0, t=0)
        self.move_stage_relative(z_move)
        logging.info(f"eucentric movement: {z_move}")

        self.connection.SEM.Optics.SetWD(wd)

    def x_corrected_stage_movement(
        self,
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
        # stage_tilt_flat_to_ion = np.deg2rad(settings.system.stage.tilt_flat_to_ion)

        # stage_rotation_flat_to_eb = np.deg2rad(
        #     settings.system.stage.rotation_flat_to_electron
        # ) % (2 * np.pi)
        stage_rotation_flat_to_ion = np.deg2rad(
            settings.system.stage.rotation_flat_to_ion
        ) % (2 * np.pi)

        # current stage position
        current_stage_position = self.get_stage_position()
        stage_rotation = current_stage_position.r % (2 * np.pi)
        stage_tilt = current_stage_position.t

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        # if rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
        #     PRETILT_SIGN = 1.0
        if rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        corrected_pretilt_angle = PRETILT_SIGN * stage_tilt_flat_to_electron
        
        y_move = expected_y/np.cos((stage_tilt + corrected_pretilt_angle))
         
        z_move = y_move*np.sin((stage_tilt + corrected_pretilt_angle)) 
        print(f'Stage tilt: {stage_tilt}, corrected pretilt: {corrected_pretilt_angle}, y_move: {y_move} z_move: {z_move}')

        return FibsemStagePosition(x=0, y=y_move, z=z_move)

    def move_flat_to_beam(
        self, settings=MicroscopeSettings, beam_type: BeamType = BeamType.ELECTRON
    ):

        # BUG if I set or pass BeamType.ION it still sees beam_type as BeamType.ELECTRON
        stage_settings = settings.system.stage

        if beam_type is BeamType.ION:
            tilt = stage_settings.tilt_flat_to_ion
        elif beam_type is BeamType.ELECTRON:
            tilt = stage_settings.tilt_flat_to_electron

        logging.info(f"Moving Stage Flat to {beam_type.name} Beam")
        self.connection.Stage.MoveTo(tiltx=tilt)

    def setup_milling(
        self,
        application_file: str,
        patterning_mode: str,
        hfw: float,
        mill_settings: FibsemMillingSettings,
    ):

        fieldsize = (
            self.connection.SEM.Optics.GetViewfield()
        )  # application_file.ajhsd or mill settings
        beam_current = mill_settings.milling_current
        spot_size = mill_settings.spot_size  # application_file
        rate = mill_settings.rate  ## in application file called Volume per Dose (m3/C)
        dwell_time = mill_settings.dwell_time  # in seconds ## in application file

        if patterning_mode == "Serial":
            parallel_mode = False
        else:
            parallel_mode = True

        layer_settings = IEtching(
            syncWriteField=False,
            writeFieldSize=hfw,
            beamCurrent=beam_current,
            spotSize=spot_size,
            rate=rate,
            dwellTime=dwell_time,
            parallel=parallel_mode,
        )
        self.layer = self.connection.DrawBeam.Layer("Layer1", layer_settings)

    def run_milling(self, milling_current: float, asynch: bool = False):

        self.connection.FIB.Beam.On()
        self.connection.DrawBeam.LoadLayer(self.layer)
        self.connection.DrawBeam.Start()
        self.connection.Progress.Show(
            "DrawBeam", "Layer 1 in progress", False, False, 0, 100
        )
        while True:
            status = self.connection.DrawBeam.GetStatus()
            running = status[0] == DBStatus.ProjectLoadedExpositionInProgress
            if running:
                progress = 0
                if status[1] > 0:
                    progress = min(100, status[2] / status[1] * 100)
                printProgressBar(progress, 100)
                self.connection.Progress.SetPercents(progress)
                time.sleep(1)
            else:
                if status[0] == DBStatus.ProjectLoadedExpositionIdle:
                    printProgressBar(100, 100, suffix="Finished")
                break

        print()  # new line on complete
        self.connection.Progress.Hide()

    def finish_milling(self, imaging_current: float):
        self.connection.DrawBeam.UnloadLayer()

    def draw_rectangle(
        self,
        pattern_settings: FibsemPatternSettings,
        mill_settings: FibsemMillingSettings,
    ):

        centre_x = pattern_settings.centre_x
        centre_y = pattern_settings.centre_y
        depth = pattern_settings.depth
        width = pattern_settings.width
        height = pattern_settings.height
        rotation = pattern_settings.rotation  # CHECK UNITS (TESCAN Takes Degrees)

        if pattern_settings.cleaning_cross_section:
            self.layer.addRectanglePolish(
                CenterX=centre_x,
                CenterY=centre_y,
                Depth=depth,
                Width=width,
                Height=height,
                Angle=rotation,
            )
        else:
            self.layer.addRectangleFilled(
                CenterX=centre_x,
                CenterY=centre_y,
                Depth=depth,
                Width=width,
                Height=height,
                Angle=rotation,
            )

        pattern = self.layer
        return pattern

    def draw_line(self, pattern_settings: FibsemPatternSettings):

        start_x = pattern_settings.start_x
        start_y = pattern_settings.start_y
        end_x = pattern_settings.end_x
        end_y = pattern_settings.end_y
        depth = pattern_settings.depth

        self.layer.addLine(
            BeginX=start_x, BeginY=start_y, EndX=end_x, EndY=end_y, Depth=depth
        )

        pattern = self.layer
        return pattern

    def set_microscope_state(self, microscope_state: MicroscopeState):
        """Reset the microscope state to the provided state"""

        logging.info(f"restoring microscope state...")

        # move to position
        self.move_stage_absolute(stage_position=microscope_state.absolute_position)

        # restore electron beam
        logging.info(f"restoring electron beam settings...")
        self.connection.SEM.Optics.SetWD(
            microscope_state.eb_settings.working_distance
            * constants.METRE_TO_MILLIMETRE
        )

        self.connection.SEM.Beam.SetCurrent(
            microscope_state.eb_settings.beam_current * constants.SI_TO_PICO
        )

        self.connection.SEM.Optics.SetViewfield(
            microscope_state.eb_settings.hfw * constants.METRE_TO_MILLIMETRE
        )

        # microscope.beams.electron_beam.stigmator.value = (
        #     microscope_state.eb_settings.stigmation
        # )

        # restore ion beam
        logging.info(f"restoring ion beam settings...")

        self.connection.FIB.Optics.SetViewfield(
            microscope_state.ib_settings.hfw * constants.METRE_TO_MILLIMETRE
        )

        # microscope.beams.ion_beam.stigmator.value = microscope_state.ib_settings.stigmation

        logging.info(f"microscope state restored")
        return


######################################## Helper functions ########################################


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


def printProgressBar(
    value, total, prefix="", suffix="", decimals=0, length=100, fill="█"
):
    """
    terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (value / float(total)))
    filled_length = int(length * value // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="")
