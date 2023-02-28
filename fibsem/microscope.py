from abc import ABC, abstractmethod
import copy
import logging
from copy import deepcopy
import datetime
import numpy as np
import sys
import time


# for easier usage
import fibsem.constants as constants
from typing import Union


try:
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
except:
    print("Automation (TESCAN) not installed.")

try:
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
except:
    print("Autoscript (ThermoFisher) not installed.")

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
    FibsemPattern,
    BeamSystemSettings
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
    def get_stage_position(self) -> FibsemStagePosition:
        pass

    @abstractmethod
    def get_current_microscope_state(self):
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
    def setup_sputter(self):
        pass

    @abstractmethod
    def draw_sputter_pattern(self):
        pass

    @abstractmethod
    def run_sputter(self):
        pass

    @abstractmethod
    def finish_sputter(self):
        pass

    @abstractmethod
    def set_microscope_state(self):
        pass
    
    @abstractmethod
    def get_available(self):
        pass


class ThermoMicroscope(FibsemMicroscope):
    """
    A class representing a Thermo Fisher FIB-SEM microscope.

    This class inherits from the abstract base class `FibsemMicroscope`, which defines the core functionality of a
    microscope. In addition to the methods defined in the base class, this class provides additional methods specific
    to the Thermo Fisher FIB-SEM microscope.

    Attributes:
        connection (SdbMicroscopeClient): The microscope client connection.

    Inherited Methods:
        connect_to_microscope(self, ip_address: str, port: int = 7520) -> None: 
            Connect to a Thermo Fisher microscope at the specified IP address and port.

        disconnect(self) -> None: 
            Disconnects the microscope client connection.

        acquire_image(self, image_settings: ImageSettings) -> FibsemImage: 
            Acquire a new image with the specified settings.

        last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage: 
            Get the last previously acquired image.

        autocontrast(self, beam_type=BeamType.ELECTRON) -> None: 
            Automatically adjust the microscope image contrast for the specified beam type.

        reset_beam_shifts(self) -> None:
            Set the beam shift to zero for the electron and ion beams.
        
        beam_shift(self, dx: float, dy: float) -> None:
            Adjusts the beam shift based on relative values that are provided.

        get_stage_position(self) -> FibsemStagePosition:
            Get the current stage position.

        get_current_microscope_state(self) -> MicroscopeState:
            Get the current microscope state

        move_stage_absolute(self, position: FibsemStagePosition):
            Move the stage to the specified coordinates.

        move_stage_relative(self, position: FibsemStagePosition):
            Move the stage by the specified relative move.

        stable_move(self, settings: MicroscopeSettings, dx: float, dy: float, beam_type: BeamType,) -> None:
            Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        eucentric_move(self, settings: MicroscopeSettings, dy: float, static_wd: bool = True) -> None:
            Move the stage vertically to correct eucentric point
        
        move_flat_to_beam(self, settings: MicroscopeSettings, beam_type: BeamType = BeamType.ELECTRON):
            Make the sample surface flat to the electron or ion beam.

        setup_milling(self, application_file: str, patterning_mode: str, hfw: float, mill_settings: FibsemMillingSettings):
            Configure the microscope for milling using the ion beam.

        run_milling(self, milling_current: float, asynch: bool = False):
            Run ion beam milling using the specified milling current.

        finish_milling(self, imaging_current: float):
            Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        draw_rectangle(self, pattern_settings: FibsemPatternSettings):
            Draws a rectangle pattern using the current ion beam.

        draw_line(self, pattern_settings: FibsemPatternSettings):
            Draws a line pattern on the current imaging view of the microscope.

        setup_sputter(self, protocol: dict):
            Set up the sputter coating process on the microscope.

        draw_sputter_pattern(self, hfw: float, line_pattern_length: float, sputter_time: float):
            Draws a line pattern for sputtering with the given parameters.

        run_sputter(self, **kwargs):
            Runs the GIS Platinum Sputter.

        finish_sputter(self, application_file: str) -> None:
            Finish the sputter process by clearing patterns and resetting beam and imaging settings.

        set_microscope_state(self, microscope_state: MicroscopeState) -> None:
            Reset the microscope state to the provided state.

    New methods:
        __init__(self): 
            Initializes a new instance of the class.

        _y_corrected_stage_movement(self, settings: MicroscopeSettings, expected_y: float, beam_type: BeamType = BeamType.ELECTRON) -> FibsemStagePosition:
            Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).
    """

    def __init__(self):
        self.connection = SdbMicroscopeClient()

    def disconnect(self):
        self.connection.disconnect()

    # @classmethod
    def connect_to_microscope(self, ip_address: str, port: int = 7520) -> None:
        """
        Connect to a Thermo Fisher microscope at the specified IP address and port.

        Args:
            ip_address (str): The IP address of the microscope to connect to.
            port (int): The port number of the microscope (default: 7520).

        Returns:
            None: This function doesn't return anything.

        Raises:
            Exception: If there's an error while connecting to the microscope.

        Example:
            To connect to a microscope with IP address 192.168.0.10 and port 7520:

            >>> microscope = ThermoMicroscope()
            >>> microscope.connect_to_microscope("192.168.0.10", 7520)

        """
        try:
            # TODO: get the port
            logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
            self.connection.connect(host=ip_address, port=port)
            logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {e}")

    def acquire_image(self, image_settings:ImageSettings) -> FibsemImage:
        """
        Acquire a new image with the specified settings.

            Args:
            image_settings (ImageSettings): The settings for the new image.

        Returns:
            FibsemImage: A new FibsemImage object representing the acquired image.
        """
        # set frame settings
        if image_settings.reduced_area is not None:
            reduced_area = image_settings.reduced_area.__to_FEI__()
        else:
            reduced_area = None
        
        frame_settings = GrabFrameSettings(
            resolution=f"{image_settings.resolution[0]}x{image_settings.resolution[1]}",
            dwell_time=image_settings.dwell_time,
            reduced_area=reduced_area,
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
        """
        Get the last previously acquired image.

        Args:
            beam_type (BeamType, optional): The imaging beam type of the last image.
                Defaults to BeamType.ELECTRON.

        Returns:
            FibsemImage: A new FibsemImage object representing the last acquired image.

        Raises:
            Exception: If there's an error while getting the last image.
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
        """
        Automatically adjust the microscope image contrast for the specified beam type.

        Args:
            beam_type (BeamType, optional): The imaging beam type for which to adjust the contrast.
                Defaults to BeamType.ELECTRON.

        Returns:
            None

        Example:
            To automatically adjust the image contrast for an ion beam type:

            >>> microscope = ThermoMicroscope()
            >>> microscope.connect_to_microscope()
            >>> microscope.autocontrast(beam_type=BeamType.ION)

        """
        self.connection.imaging.set_active_view(beam_type.value)
        self.connection.auto_functions.run_auto_cb()

    def reset_beam_shifts(self) -> None:
        """
        Set the beam shift to zero for the electron and ion beams.

        Resets the beam shift for both the electron and ion beams to (0,0), effectively centering the beams on the sample.

        Args:
            self (FibsemMicroscope): instance of the FibsemMicroscope object
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

    def beam_shift(self, dx: float, dy: float) -> None:
        """
        Adjusts the beam shift based on relative values that are provided.
        
        Args:
            self (FibsemMicroscope): Fibsem microscope object
            dx (float): the relative x term
            dy (float): the relative y term
        """
        self.connection.beams.ion_beam.beam_shift.value += (-dx, dy)

    def get_stage_position(self) -> FibsemStagePosition:
        """
        Get the current stage position.

        This method retrieves the current stage position from the microscope and returns it as
        a FibsemStagePosition object.

        Returns:
            FibsemStagePosition: The current stage position.
        """
        self.connection.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.RAW
        )
        stage_position = self.connection.specimen.stage.current_position
        print(stage_position)
        self.connection.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.SPECIMEN
        )
        return FibsemStagePosition.from_autoscript_position(stage_position)

    def get_current_microscope_state(self) -> MicroscopeState:
        """
        Get the current microscope state

        This method retrieves the current microscope state from the microscope and returns it as
        a MicroscopeState object.

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
        """
        Move the stage to the specified coordinates.

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

    def move_stage_relative(self, position: FibsemStagePosition):
        """
        Move the stage by the specified relative move.

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
        """
        Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        Args:
            settings (MicroscopeSettings): microscope settings
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
            beam_type (BeamType): beam type to move in
        """
        wd = self.connection.beams.electron_beam.working_distance.value

        # calculate stage movement
        x_move = FibsemStagePosition(x=dx, y=0, z=0)
        yz_move = self._y_corrected_stage_movement(
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
        """
        Move the stage vertically to correct eucentric point

        Args:
            settings (MicroscopeSettings): microscope settings
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

    def _y_corrected_stage_movement(
        self,
        settings: MicroscopeSettings,
        expected_y: float,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> FibsemStagePosition:
        """
        Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).

        Args:
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
    ) -> None:
        """
        Moves the microscope stage to the tilt angle corresponding to the given beam type,
        so that the stage is flat with respect to the beam.

        Args:
            settings (MicroscopeSettings): The current microscope settings.
            beam_type (BeamType): The type of beam to which the stage should be made flat.
                Must be one of BeamType.ELECTRON or BeamType.ION.

        Returns:
            None.
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
        patterning_mode: str,
        mill_settings: FibsemMillingSettings,
    ):
        """
        Configure the microscope for milling using the ion beam.

        Args:
            application_file (str): Path to the milling application file.
            patterning_mode (str): Patterning mode to use.
            hfw (float): Desired horizontal field width in meters.
            mill_settings (FibsemMillingSettings): Milling settings.

        Returns:
            None.

        Raises:
            NotImplementedError: If the specified patterning mode is not supported.

        Note:
            This method sets up the microscope imaging and patterning for milling using the ion beam.
            It sets the active view and device to the ion beam, the default beam type to the ion beam,
            the specified application file, and the specified patterning mode.
            It also clears any existing patterns and sets the horizontal field width to the desired value.
            The method does not start the milling process.
        """
        self.connection.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
        self.connection.imaging.set_active_device(BeamType.ION.value)
        self.connection.patterning.set_default_beam_type(
            BeamType.ION.value
        )  # ion beam default
        self.connection.patterning.set_default_application_file(mill_settings.application_file)
        self.connection.patterning.mode = patterning_mode
        self.connection.patterning.clear_patterns()  # clear any existing patterns
        self.connection.beams.ion_beam.horizontal_field_width.value = mill_settings.hfw

    def run_milling(self, milling_current: float, asynch: bool = False):
        """
        Run ion beam milling using the specified milling current.

        Args:
            milling_current (float): The current to use for milling in amps.
            asynch (bool, optional): If True, the milling will be run asynchronously. 
                                     Defaults to False, in which case it will run synchronously.

        Returns:
            None

        Raises:
            None
        """
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
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        """
        self.connection.patterning.clear_patterns()
        self.connection.beams.ion_beam.beam_current.value = imaging_current
        self.connection.patterning.mode = "Serial"

    def draw_rectangle(
        self,
        pattern_settings: FibsemPatternSettings,
    ):
        """
        Draws a rectangle pattern using the current ion beam.

        Args:
            pattern_settings (FibsemPatternSettings): the settings for the pattern to draw.

        Returns:
            Pattern: the created pattern.

        Raises:
            AutoscriptError: if an error occurs while creating the pattern.

        Notes:
            The rectangle pattern will be centered at the specified coordinates (centre_x, centre_y) with the specified
            width, height and depth (in nm). If the cleaning_cross_section attribute of pattern_settings is True, a
            cleaning cross section pattern will be created instead of a rectangle pattern.

            The pattern will be rotated by the angle specified in the rotation attribute of pattern_settings (in degrees)
            and scanned in the direction specified in the scan_direction attribute of pattern_settings ('horizontal' or
            'vertical').

            The created pattern can be added to the patterning queue and executed using the methods in the PatterningQueue
            class of the autoscript_sdb_microscope_client package.
        """
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
        """
        Draws a line pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemPatternSettings): A data class object specifying the pattern parameters,
                including the start and end points, and the depth of the pattern.

        Returns:
            LinePattern: A line pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.

        Raises:
            autoscript.exceptions.InvalidArgumentException: if any of the pattern parameters are invalid.
        """
        pattern = self.connection.patterning.create_line(
            start_x=pattern_settings.start_x,
            start_y=pattern_settings.start_y,
            end_x=pattern_settings.end_x,
            end_y=pattern_settings.end_y,
            depth=pattern_settings.depth,
        )

        return pattern

    def setup_sputter(self, protocol: dict):
        """
        Set up the sputter coating process on the microscope.

        Args:
            protocol (dict): Dictionary containing the protocol details for sputter coating.

        Returns:
            None

        Raises:
            None

        Notes:
            This function sets up the sputter coating process on the microscope. 
            It sets the active view to the electron beam, clears any existing patterns, and sets the default beam type to the electron beam. 
            It then inserts the multichem and turns on the heater for the specified gas according to the given protocol. 
            This function also waits for 3 seconds to allow the heater to warm up.
        """
        self.original_active_view = self.connection.imaging.get_active_view()
        self.connection.imaging.set_active_view(BeamType.ELECTRON.value)
        self.connection.patterning.clear_patterns()
        self.connection.patterning.set_default_application_file(protocol["application_file"])
        self.connection.patterning.set_default_beam_type(BeamType.ELECTRON.value)
        self.multichem = self.connection.gas.get_multichem()
        self.multichem.insert(protocol["position"])
        self.multichem.turn_heater_on(protocol["gas"])  # "Pt cryo")
        time.sleep(3)

    def draw_sputter_pattern(self, hfw: float, line_pattern_length: float, sputter_time: float):
        """
        Draws a line pattern for sputtering with the given parameters.

        Args:
            hfw (float): The horizontal field width of the electron beam.
            line_pattern_length (float): The length of the line pattern to draw.
            sputter_time (float): The time to sputter the line pattern.

        Returns:
            None

        Notes:
            Sets the horizontal field width of the electron beam to the given value.
            Draws a line pattern for sputtering with the given length and milling depth.
            Sets the sputter time of the line pattern to the given value.

        """
        self.connection.beams.electron_beam.horizontal_field_width.value = hfw
        pattern = self.connection.patterning.create_line(
            -line_pattern_length / 2,  # x_start
            +line_pattern_length,  # y_start
            +line_pattern_length / 2,  # x_end
            +line_pattern_length,  # y_end
            2e-6,
        )  # milling depth
        pattern.time = sputter_time + 0.1

    def run_sputter(self, **kwargs):
        """
        Runs the GIS Platinum Sputter.

        Args:
            **kwargs: Optional keyword arguments for the sputter function. The required argument for
        the Thermo version is "sputter_time" (int), which specifies the time to sputter in seconds. 

        Returns:
            None

        Notes:
        - Blanks the electron beam.
        - Starts sputtering with platinum for the specified sputter time, and waits until the sputtering
        is complete before continuing.
        - If the patterning state is not ready, raises a RuntimeError.
        - If the patterning state is running, stops the patterning.
        - If the patterning state is idle, logs a warning message suggesting to adjust the patterning
        line depth.
        """
        sputter_time = kwargs["sputter_time"]

        self.connection.beams.electron_beam.blank()
        if self.connection.patterning.state == "Idle":
            logging.info("Sputtering with platinum for {} seconds...".format(sputter_time))
            self.connection.patterning.start()  # asynchronous patterning
            time.sleep(sputter_time + 5)
        else:
            raise RuntimeError("Can't sputter platinum, patterning state is not ready.")
        if self.connection.patterning.state == "Running":
            self.connection.patterning.stop()
        else:
            logging.warning("Patterning state is {}".format(self.connection.patterning.state))
            logging.warning("Consider adjusting the patterning line depth.")

    def finish_sputter(self, application_file: str) -> None:
        """
        Finish the sputter process by clearing patterns and resetting beam and imaging settings.

        Args:
            application_file (str): The path to the default application file to use.

        Returns:
            None

        Raises:
            None

        Notes:
            This function finishes the sputter process by clearing any remaining patterns and restoring the beam and imaging settings to their
            original state. It sets the beam current back to imaging current and sets the default beam type to ion beam.
            It also retracts the multichem and logs that the sputtering process has finished.
        """
        # Clear any remaining patterns
        self.connection.patterning.clear_patterns()

        # Restore beam and imaging settings to their original state
        self.connection.beams.electron_beam.unblank()
        self.connection.patterning.set_default_application_file(application_file)
        self.connection.imaging.set_active_view(self.original_active_view)
        self.connection.patterning.set_default_beam_type(BeamType.ION.value)  # set ion beam
        self.multichem.retract()

        # Log that the sputtering process has finished
        logging.info("Platinum sputtering process completed.")

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        """Reset the microscope state to the provided state.

        Args:
            microscope_state (MicroscopeState): A `MicroscopeState` object that contains the desired state of the microscope.

        Returns:
            None.

        Raises:
            None.

        Notes:
            This function restores the microscope state to the provided state. It moves the stage to the absolute position specified
            in the `MicroscopeState` object, and then restores the electron and ion beam settings to their values in the `MicroscopeState`
            object. It also logs messages indicating the progress of the operation.
        """

        logging.info(f"Restoring microscope state...")

        # Move to position
        self.move_stage_absolute(stage_position=microscope_state.absolute_position)

        # Restore electron beam settings
        logging.info(f"Restoring electron beam settings...")
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

        # Restore ion beam settings
        logging.info(f"Restoring ion beam settings...")
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

        # Link the specimen stage
        self.connection.specimen.stage.link()

        # Log the completion of the operation
        logging.info(f"Microscope state restored.")
    
    def get_available(self, key: str, beam_type: BeamType = None)-> list:
        values = []
        if key == "application_file":
            values = self.connection.patterning.list_all_application_files()

        return values

class TescanMicroscope(FibsemMicroscope):
    """
    A class representing a TESCAN FIB-SEM microscope.

    This class inherits from the abstract base class `FibsemMicroscope`, which defines the core functionality of a
    microscope. In addition to the methods defined in the base class, this class provides additional methods specific
    to the TESCAN FIB-SEM microscope.

    Attributes:
        connection (Automation): The microscope client connection.
        ion_detector_active (Automation.FIB.Detector): The active ion beam detector.
        last_image_eb (FibsemImage): A saved copy of the most recent electron beam image.
        last_image_ib (FibsemImage): A saved copy of the most recent ion beam image.

    Inherited Methods:
        connect_to_microscope(self, ip_address: str, port: int = 7520) -> None: 
            Connect to a Thermo Fisher microscope at the specified IP address and port.

        disconnect(self) -> None: 
            Disconnects the microscope client connection.

        acquire_image(self, image_settings: ImageSettings) -> FibsemImage: 
            Acquire a new image with the specified settings.

        last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage: 
            Get the last previously acquired image.

        autocontrast(self, beam_type=BeamType.ELECTRON) -> None: 
            Automatically adjust the microscope image contrast for the specified beam type.

        reset_beam_shifts(self) -> None:
            Set the beam shift to zero for the electron and ion beams.
        
        beam_shift(self, dx: float, dy: float) -> None:
            Adjusts the beam shift based on relative values that are provided.

        get_stage_position(self) -> FibsemStagePosition:
            Get the current stage position.

        get_current_microscope_state(self) -> MicroscopeState:
            Get the current microscope state

        move_stage_absolute(self, position: FibsemStagePosition):
            Move the stage to the specified coordinates.

        move_stage_relative(self, position: FibsemStagePosition):
            Move the stage by the specified relative move.

        stable_move(self, settings: MicroscopeSettings, dx: float, dy: float, beam_type: BeamType,) -> None:
            Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        eucentric_move(self, settings: MicroscopeSettings, dy: float, static_wd: bool = True) -> None:
            Move the stage vertically to correct eucentric point
        
        move_flat_to_beam(self, settings: MicroscopeSettings, beam_type: BeamType = BeamType.ELECTRON):
            Make the sample surface flat to the electron or ion beam.

        setup_milling(self, application_file: str, patterning_mode: str, hfw: float, mill_settings: FibsemMillingSettings):
            Configure the microscope for milling using the ion beam.

        run_milling(self, milling_current: float, asynch: bool = False):
            Run ion beam milling using the specified milling current.

        finish_milling(self, imaging_current: float):
            Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        draw_rectangle(self, pattern_settings: FibsemPatternSettings):
            Draws a rectangle pattern using the current ion beam.

        draw_line(self, pattern_settings: FibsemPatternSettings):
            Draws a line pattern on the current imaging view of the microscope.

        setup_sputter(self, protocol: dict):
            Set up the sputter coating process on the microscope.

        draw_sputter_pattern(self, hfw: float, line_pattern_length: float, sputter_time: float):
            Draws a line pattern for sputtering with the given parameters.

        run_sputter(self, **kwargs):
            Runs the GIS Platinum Sputter.

        finish_sputter(self, application_file: str) -> None:
            Finish the sputter process by clearing patterns and resetting beam and imaging settings.

        set_microscope_state(self, microscope_state: MicroscopeState) -> None:
            Reset the microscope state to the provided state.

    New methods:
        __init__(self): 
            Initializes a new instance of the class.

        _get_eb_image(self, image_settings=ImageSettings) -> FibsemImage:
            Acquires an electron beam (EB) image with the given settings and returns a FibsemImage object.

        _get_ib_image(self, image_settings=ImageSettings):
            Acquires an ion beam (IB) image with the given settings and returns a FibsemImage object.

        _y_corrected_stage_movement(self, settings: MicroscopeSettings, expected_y: float, beam_type: BeamType = BeamType.ELECTRON) -> FibsemStagePosition:
            Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).
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
        """
            Connects to a microscope with the specified IP address and port.

            Args:
                ip_address: A string that represents the IP address of the microscope.
                port: An integer that represents the port number to use (default 8300).

            Returns:
                None.
        """
        self.connection = Automation(ip_address, port)

    def acquire_image(self, image_settings=ImageSettings) -> FibsemImage:
        """
            Acquires an image using the specified image settings.

            Args:
                image_settings: An instance of the `ImageSettings` class that represents the image settings to use (default `ImageSettings`).

            Returns:
                A `FibsemImage` object that represents the acquired image.
        """
        if image_settings.beam_type.name == "ELECTRON":
            image = self._get_eb_image(image_settings)
            self.last_image_eb = image
        if image_settings.beam_type.name == "ION":
            image = self._get_ib_image(image_settings)
            self.last_image_ib = image

        return image

    def _get_eb_image(self, image_settings: ImageSettings) -> FibsemImage:
        """
        Acquires an electron beam (EB) image with the given settings and returns a FibsemImage object.

        Args:
            image_settings (ImageSettings): An object containing the settings for the acquired image. 

        Returns:
            FibsemImage: The acquired image as a FibsemImage object.

        Notes:
            This function acquires an electron beam (EB) image with the given settings and returns it as a FibsemImage object. 
            The function sets up the microscope parameters, including the electron beam dwell time, image resolution, and 
            region of interest (ROI), if specified. It then acquires the image and creates a FibsemImage object from it, 
            including metadata on the microscope state and the beam and ion settings.

            Before acquiring the image, the function ensures that the electron beam is turned on and that the SEM scan is stopped. 
            It selects the most suitable detector for the image, assigns it to a channel, and enables the channel for acquisition. 
            The function then acquires the image from the channel using the specified dwell time and resolution.

            The function also records the microscope state at the time of image acquisition, including the stage position, beam 
            settings, and ion beam settings. The acquired image is returned as a FibsemImage object, which includes the image 
            data and metadata on the microscope state and the image settings.
        """
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
            left =  imageWidth - int(image_settings.reduced_area.left * imageWidth)
            top = imageHeight - int(image_settings.reduced_area.top * imageHeight)
            image = self.connection.SEM.Scan.AcquireROIFromChannel(
                Channel= 0,
                Width= imageWidth,
                Height= imageHeight,
                Left= left,
                Top= top,
                Right= left - imageWidth -1 ,
                Bottom= top - imageHeight - 1,
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

    def _get_ib_image(self, image_settings: ImageSettings):
        """
        Acquires an ion beam (IB) image with the given settings and returns a FibsemImage object.

        Args:
            image_settings (ImageSettings): The settings for the acquired image.

        Returns:
            FibsemImage: The acquired image as a FibsemImage object.

        Notes:
            - The function acquires an IB image with the given settings by configuring the detectors, scanning, and selecting the dwell time, resolution, and viewfield.
            - If the image settings include a reduced area, the function will acquire an image within the reduced area.
            - The function also captures the microscope state at the time of acquisition and includes this information in the metadata of the acquired image.
        """
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
            left =  imageWidth - int(image_settings.reduced_area.left * imageWidth)
            top = imageHeight - int(image_settings.reduced_area.top * imageHeight)
            image = self.connection.FIB.Scan.AcquireROIFromChannel(
                Channel= 0,
                Width= imageWidth,
                Height= imageHeight,
                Left= left,
                Top= top,
                Right= left - imageWidth -1 ,
                Bottom= top - imageHeight - 1,
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
        """    
        Returns the last acquired image for the specified beam type.

        Args:
            beam_type (BeamType.ELECTRON or BeamType.ION): The type of beam used to acquire the image.

        Returns:
            FibsemImage: The last acquired image of the specified beam type.

        """
        if beam_type == BeamType.ELECTRON:
            image = self.last_image_eb
        elif beam_type == BeamType.ION:
            image = self.last_image_ib
        else:
            raise Exception("Beam type error")
        return image

    def autocontrast(self, beam_type: BeamType) -> None:
        """Automatically adjust the microscope image contrast for the specified beam type.

        Args:
            beam_type (BeamType, optional): The imaging beam type for which to adjust the contrast.
                Defaults to BeamType.ELECTRON.

        Returns:
            None

        Example:
            To automatically adjust the image contrast for an ion beam type:

            >>> microscope = TescanMicroscope()
            >>> microscope.connect_to_microscope()
            >>> microscope.autocontrast(beam_type=BeamType.ION)

        """
        if beam_type.name == BeamType.ELECTRON:
            self.connection.SEM.Detector.StartAutoSignal(0)
        if beam_type.name == BeamType.ION:
            self.connection.FIB.Detector.AutoSignal(0)

    def reset_beam_shifts(self):
        """
        Set the beam shift to zero for the electron and ion beams.

        Resets the beam shift for both the electron and ion beams to (0,0), effectively centering the beams on the sample.

        Args:
            self (FibsemMicroscope): instance of the FibsemMicroscope object
        """
        self.connection.FIB.Optics.SetImageShift(0, 0)
        self.connection.SEM.Optics.SetImageShift(0, 0)

    def beam_shift(self, dx: float, dy: float):
        """Adjusts the beam shift based on relative values that are provided.
        
        Args:
            self (FibsemMicroscope): Fibsem microscope object
            dx (float): the relative x term
            dy (float): the relative y term
        """
        x, y = self.connection.FIB.Optics.GetImageShift()
        dx *=  constants.METRE_TO_MILLIMETRE # Convert to mm from m.
        dy *=  constants.METRE_TO_MILLIMETRE
        x -= dx # NOTE: Not sure why the dx is -dx, this may be thermo specific and doesn't apply to TESCAN?
        y += dy
        self.connection.FIB.Optics.SetImageShift(x,y) 
        
    def get_stage_position(self):
        """
        Get the current stage position.

        This method retrieves the current stage position from the microscope and returns it as
        a FibsemStagePosition object.

        Returns:
            FibsemStagePosition: The current stage position.
        """
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
        """
        Get the current microscope state

        This method retrieves the current microscope state from the microscope and returns it as
        a MicroscopeState object.

        Returns:
            MicroscopeState: current microscope state
        """
        image_eb = self.last_image(BeamType.ELECTRON)
        image_ib = self.last_image(BeamType.ION)

        if image_ib is not None:
            ib_settings = BeamSettings(
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

    def move_stage_absolute(self, position: FibsemStagePosition):
        """
        Move the stage to the specified coordinates.

        Args:
            x (float): The x-coordinate to move to (in meters).
            y (float): The y-coordinate to move to (in meters).
            z (float): The z-coordinate to move to (in meters).
            r (float): The rotation to apply (in radians).
            tx (float): The x-axis tilt to apply (in radians).

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
        """
        Move the stage by the specified relative move.

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
        """
        Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        Args:
            settings (MicroscopeSettings): microscope settings
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
        """
        wd = self.connection.SEM.Optics.GetWD()

        # calculate stage movement
        x_move = FibsemStagePosition(x=-dx, y=0, z=0) 
        yz_move = self._y_corrected_stage_movement(
            settings=settings,
            expected_y=-dy,
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
        """
        Move the stage vertically to correct eucentric point

        Args:
            settings (MicroscopeSettings): microscope settings
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

    def _y_corrected_stage_movement(
        self,
        settings: MicroscopeSettings,
        expected_y: float,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> FibsemStagePosition:
        """
        Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).

        Args:
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
        """
        Moves the microscope stage to the tilt angle corresponding to the given beam type,
        so that the stage is flat with respect to the beam.

        Args:
            settings (MicroscopeSettings): The current microscope settings.
            beam_type (BeamType): The type of beam to which the stage should be made flat.
                Must be one of BeamType.ELECTRON or BeamType.ION.

        Returns:
            None.
        """
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
        patterning_mode: str,
        mill_settings: FibsemMillingSettings,
    ):
        """
        Configure the microscope for milling using the ion beam.

        Args:
            application_file (str): Path to the milling application file.
            patterning_mode (str): Patterning mode to use.
            hfw (float): Desired horizontal field width in meters.
            mill_settings (FibsemMillingSettings): Milling settings.

        Returns:
            None.

        Raises:
            NotImplementedError: If the specified patterning mode is not supported.

        Note:
            This method sets up the microscope imaging and patterning for milling using the ion beam.
            It sets the active view and device to the ion beam, the default beam type to the ion beam,
            the specified application file, and the specified patterning mode.
            It also clears any existing patterns and sets the horizontal field width to the desired value.
            The method does not start the milling process.
        """
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
            writeFieldSize=mill_settings.hfw,
            beamCurrent=beam_current,
            spotSize=spot_size,
            rate=rate,
            dwellTime=dwell_time,
            parallel=parallel_mode,
        )
        self.layer = self.connection.DrawBeam.Layer("Layer1", layer_settings)

    def run_milling(self, milling_current: float, asynch: bool = False):
        """
        Runs the ion beam milling process using the specified milling current.

        Args:
            milling_current (float): The milling current to use, in amps.
            asynch (bool, optional): Whether to run the milling asynchronously. Defaults to False.

        Returns:
            None
        """
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
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        """
        self.connection.DrawBeam.UnloadLayer()

    def draw_rectangle(
        self,
        pattern_settings: FibsemPatternSettings,
    ):
        """
        Draws a rectangle pattern using the current ion beam.

        Args:
            pattern_settings (FibsemPatternSettings): the settings for the pattern to draw.

        Returns:
            Pattern: the created pattern.

        Raises:
            AutomationError: if an error occurs while creating the pattern.

        Notes:
            The rectangle pattern will be centered at the specified coordinates (centre_x, centre_y) with the specified
            width, height and depth (in nm). If the cleaning_cross_section attribute of pattern_settings is True, a
            cleaning cross section pattern will be created instead of a rectangle pattern.

            The pattern will be rotated by the angle specified in the rotation attribute of pattern_settings (in degrees)
            and scanned in the direction specified in the scan_direction attribute of pattern_settings ('horizontal' or
            'vertical').

            The created pattern can be added to the patterning queue and executed using the layer methods in Automation.
        """
        centre_x = pattern_settings.centre_x
        centre_y = pattern_settings.centre_y
        depth = pattern_settings.depth
        width = pattern_settings.width
        height = pattern_settings.height
        rotation = pattern_settings.rotation * constants.RADIANS_TO_DEGREES # CHECK UNITS (TESCAN Takes Degrees)

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
        """
        Draws a line pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemPatternSettings): A data class object specifying the pattern parameters,
                including the start and end points, and the depth of the pattern.

        Returns:
            LinePattern: A line pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.
        """
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

    def setup_sputter(self, protocol: dict):
        """
        Set up the sputter coating process on the microscope.

        Args:
            protocol (dict): Contains all of the necessary values to setup up platinum sputtering.
                For TESCAN:
                    - hfw: Horizontal field width (m).
                    - beam_current: Ion beam current in [A].
                    - spot_size: Ion beam spot size in [m].
                    - rate: Ion/electron etching rate (deposition rate) in [m3/A/s]. E.g. for silicone 4.7e-10 m3/A/s.
                    - dwell time: Pixel dwell time in [s].

        Returns:
            None

        Raises:
            None

        Notes:
            This function sets up the sputter coating process on the microscope. 
            It sets the active view to the electron beam, clears any existing patterns, and sets the default beam type to the electron beam. 
            It then inserts the multichem and turns on the heater for the specified gas according to the given protocol. 
            This function also waits for 3 seconds to allow the heater to warm up.
        """
        self.connection.FIB.Beam.On()
        lines = self.connection.GIS.Enum()
        for line in lines:
            if line.name == "Platinum":
                self.line = line

                # Start GIS heating
                self.connection.GIS.PrepareTemperature(line, True)

                # Insert GIS into working position
                self.connection.GIS.MoveTo(line, Automation.GIS.Position.Working)

                # Wait for GIS heated
                self.connection.GIS.WaitForTemperatureReady(line)

        try:
            layerSettings = self.connection.DrawBeam.LayerSettings.IDeposition(
                syncWriteField=True,
                writeFieldSize=protocol["weld"]["hfw"],
                beamCurrent=protocol["beam_current"],
                spotSize=protocol["spot_size"],
                rate=3e-10, # Value for platinum
                dwellTime=protocol["dwell_time"],
            )
            self.layer = self.connection.DrawBeam.LoadLayer(layerSettings)
        except:
            defaultLayerSettings = self.connection.DrawBeam.Layer.fromDbp('.\\fibsem\\config\\deposition.dbp')
            self.layer = self.connection.DrawBeam.LoadLayer(defaultLayerSettings[0])

    def draw_sputter_pattern(self, hfw, line_pattern_length, *args, **kwargs):
        """
        Draws a line pattern for sputtering with the given parameters.

        Args:
            hfw (float): The horizontal field width of the electron beam.
            line_pattern_length (float): The length of the line pattern to draw.
            *args, **kwargs: This represents the arguments used by ThermoMicroscope that are not required for the TescanMicroscope.

        Returns:
            None

        Notes:
            Sets the horizontal field width of the electron beam to the given value.
            Draws a line pattern for sputtering with the given length and milling depth.
            Sets the sputter time of the line pattern to the given value.

        """
        self.connection.FIB.Optics.SetViewfield(
            hfw * constants.METRE_TO_MILLIMETRE
        )

        start_x=-line_pattern_length/2, 
        start_y=+line_pattern_length,
        end_x=+line_pattern_length/2,
        end_y=+line_pattern_length,
        depth=2e-6
        
        pattern = self.layer.addLine(
            BeginX=start_x, BeginY=start_y, EndX=end_x, EndY=end_y, Depth=depth
        )
        
        return pattern

    def run_sputter(self, *args, **kwargs):
        """
        Runs the GIS Platinum Sputter.

        Args:
            *args, **kwargs: Used to maintain functionality and compatability between microscopes. No arguments required.
            
        Runs the GIS Platinum Sputter.

        Returns:
            None
        """
        # Open GIS valve to let the gas flow onto the sample
        self.connection.GIS.OpenValve(self.line)

        try:
            # Run predefined deposition process
            self.connection.DrawBeam.Start()
            self.connection.Progress.Show("DrawBeam", "Layer 1 in progress", False, False, 0, 100)
            while True:
                status = self.connection.DrawBeam.GetStatus()
                running = status[0] == self.connection.DrawBeam.Status.ProjectLoadedExpositionInProgress
                if running:
                    progress = 0
                    if status[1] > 0:
                        progress = min(100, status[2] / status[1] * 100)
                    printProgressBar(progress, 100)
                    self.connection.Progress.SetPercents(progress)
                    time.sleep(1)
                else:
                    if status[0] == self.connection.DrawBeam.Status.ProjectLoadedExpositionIdle:
                        printProgressBar(100, 100, suffix='Finished')
                        print('')
                    break
        finally:
            # Close GIS Valve in both - success and failure
            self.connection.GIS.CloseValve(self.line)
        
    def finish_sputter(self, *args, **kwargs):
        """
        Finish the sputter process by retracting the GIS chamber and turning off the heating.

        Args:
            *args, **kwargs: This represents the arguments used by ThermoMicroscope that are not required for the TescanMicroscope.

        Returns:
            None

        Raises:
            None
        """
        # Move GIS out from chamber and turn off heating
        self.connection.GIS.MoveTo(self.line, Automation.GIS.Position.Home)
        self.connection.GIS.PrepareTemperature(self.line, False)

    def set_microscope_state(self, microscope_state: MicroscopeState):
        """Reset the microscope state to the provided state.

        Args:
            microscope_state (MicroscopeState): A `MicroscopeState` object that contains the desired state of the microscope.

        Returns:
            None.

        Raises:
            None.

        Notes:
            This function restores the microscope state to the provided state. This function cannot be fully implemented as their are certain aspects of
            the state that cannot be set for the TESCAN microscope by the TESCAN Automation API.
        """

        logging.info(f"restoring microscope state...")

        # move to position
        self.move_stage_absolute(position=microscope_state.absolute_position)

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

    def get_available(self, key: str, beam_type: BeamType = None)-> list:

        return []

########################
class DemoMicroscope(FibsemMicroscope):

    def __init__(self):            
        self.connection = None
        self.stage_position = FibsemStagePosition()
        self.electron_beam = self.get_beam_settings(BeamType.ELECTRON)
        self.ion_beam = self.get_beam_settings(BeamType.ION)

    def connect_to_microscope(self):
        logging.info(f"Connected to Demo Microscope")
        return

    def disconnect(self):
        logging.info(f"Disconnected from Demo Microscope")

    def acquire_image(self, image_settings: ImageSettings) -> FibsemImage:

        vfw = image_settings.hfw * image_settings.resolution[1] / image_settings.resolution[0]
        pixelsize = Point(image_settings.hfw / image_settings.resolution[0], 
                          vfw / image_settings.resolution[1])
        
        image = FibsemImage(
            data=np.random.randint(low=0, high=256, 
                size=(image_settings.resolution[1],image_settings.resolution[0]), 
                dtype=np.uint8),
            metadata=FibsemImageMetadata(image_settings=image_settings, pixel_size=pixelsize,
                                         microscope_state=MicroscopeState()))
                                 
        return image

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        logging.info(f"Getting last image: {beam_type}")
        return NotImplemented
    
    def autocontrast(self, beam_type: BeamType) -> None:
        logging.info(f"Autocontrast: {beam_type}")

    def reset_beam_shifts(self) -> None:
        logging.info(f"Resetting beam shifts")
        self.electron_beam.shift = Point(0,0)
        self.ion_beam.shift = Point(0,0)

    def beam_shift(self, dx: float, dy: float) -> None:
        beam_type = BeamType.ION # TODO: add beam_type to params for ABC
        logging.info(f"Beam shift: dx={dx}, dy={dy} ({beam_type})")
        if beam_type == BeamType.ELECTRON:
            self.electron_beam.shift += Point(dx, dy)
        elif beam_type == BeamType.ION:
            self.ion_beam.shift += Point(dx, dy)

    def get_stage_position(self) -> FibsemStagePosition:
        logging.info(f"Getting stage position: {self.stage_position}")
        return self.stage_position
    
    def get_current_microscope_state(self) -> MicroscopeState:
        logging.info(f"Getting microscope state")
        return MicroscopeState(absolute_position=self.stage_position)

    def move_stage_absolute(self, position: FibsemStagePosition) -> None:
        logging.info(f"Moving stage: {position} (Absolute)")
        self.stage_position = position

    def move_stage_relative(self, position: FibsemStagePosition) -> None:
        logging.info(f"Moving stage: {position} (Relative)")
        self.stage_position += position

    def stable_move(self, settings: MicroscopeSettings, dx: float, dy:float, beam_type: BeamType) -> None:
        logging.info(f"Moving stage: dx={dx}, dy={dy}, beam_type = {beam_type.name} (Stable)")
        self.stage_position.x += dx
        self.stage_position.y += dy

    def eucentric_move(self, settings:MicroscopeSettings, dy: float, static_wd: bool=True) -> None:
        logging.info(f"Moving stage: dy={dy} (Eucentric)")
        self.stage_position.z += dy / np.cos(np.deg2rad(90-settings.system.stage.tilt_flat_to_ion))

    def move_flat_to_beam(self, settings: MicroscopeSettings, beam_type: BeamType) -> None:
        logging.info(f"Moving stage: Flat to {beam_type.name} beam")

        if beam_type is BeamType.ELECTRON:
            r = settings.system.stage.tilt_flat_to_electron
            t = settings.system.stage.tilt_flat_to_electron
        if beam_type is BeamType.ION:
            r = settings.system.stage.tilt_flat_to_ion
            t = settings.system.stage.tilt_flat_to_ion

        # TODO: pre-tilt shuttle

        self.stage_position.r = np.deg2rad(r)
        self.stage_position.t = np.deg2rad(t)

    def setup_milling(self, patterning_mode:str, mill_settings: FibsemMillingSettings):
        pass

    def run_milling(self, milling_current: float, asynch: bool = False) -> None:
        pass

    def finish_milling(self, imaging_current: float) -> None:
        pass

    def draw_rectangle(self, pattern_settings: FibsemPatternSettings) -> None:
        pass

    def draw_line(self, pattern_settings: FibsemPatternSettings) -> None:
        pass

    def setup_sputter(self):
        pass

    def draw_sputter_pattern(self):
        pass

    def run_sputter(self):
        pass

    def finish_sputter(self):
        pass

    def set_microscope_state(self, state: MicroscopeState):
        logging.info(f"Setting microscope state")

    def get_available(self, key: str, beam_type: BeamType = None) -> list[float]:
        
        values = []
        if key == "current":
            if beam_type == BeamType.ELECTRON:
                values = [1.0e-12]
            if beam_type == BeamType.ION:
                values = [20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.6e-9, 28.0e-9, 120e-9]
        

        if key == "application_file":
            values = ["Si", "autolamella", "cryo_Pt_dep"]


        return values


    def get_beam_system_state(self, beam_type: BeamType) -> BeamSystemSettings:

        # get current beam settings
        voltage = 30000
        current = 20e-12 
        detector_type = "ETD"
        detector_mode = "SecondaryElectrons"
        shift = Point(0, 0)

        if beam_type is BeamType.ION:
            eucentric_height = 16.5e-3
            plasma_gas = "Argon"
        else:
            eucentric_height = 4.0e-3
            plasma_gas = None

        return BeamSystemSettings(
            beam_type=beam_type,
            voltage=voltage,
            current=current,
            detector_type=detector_type,
            detector_mode=detector_mode,
            eucentric_height=eucentric_height,
            plasma_gas=plasma_gas,
        )


    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:

        return BeamSettings(
            beam_type=beam_type,
            beam_current=20e-12,
            working_distance=16.5e-3,
            resolution=[1526, 1024],
            dwell_time=1e-6,
            hfw=150e-6,
            stigmation=Point(0, 0),
            shift=Point(0, 0),
        )






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
