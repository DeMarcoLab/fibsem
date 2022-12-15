from abc import ABC, abstractmethod
import copy
import logging
import datetime
import numpy as np
from fibsem.config import load_microscope_manufacturer
import re

manufacturer = load_microscope_manufacturer()
if manufacturer == "Tescan":
    from tescanautomation import Automation
    from tescanautomation.SEM import HVBeamStatus as SEMStatus
    from tescanautomation.Common import Bpp
    from tescanautomation.GUI import SEMInfobar


if manufacturer == "Thermo":
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings
    from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

import sys

from fibsem.structures import (
    BeamType,
    ImageSettings,
    Point,
    FibsemImage,
    FibsemImageMetadata,
    MicroscopeState,
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

        numbers = re.findall(r"\d+", image_settings.resolution)
        imageWidth = int(numbers[0])


        if image_settings.pixel_size is not None:
            image_settings.hfw = image_settings.pixel_size.x * imageWidth

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


class TescanMicroscope(FibsemMicroscope):
    """TESCAN Microscope class, uses FibsemMicroscope as blueprint

    Args:
        FibsemMicroscope (ABC): abstract implementation
    """

    def __init__(self, ip_address: str = "localhost"):
        self.connection = Automation(ip_address)
        detectors = self.connection.FIB.Detector.Enum()
        self.ion_detector_active = detectors[0]
        self.electron_detector_active = self.connection.SEM.Detector.SESuitable()
        self.last_image_eb = None
        self.last_image_ib = None

    def disconnect(self):
        self.connection.Disconnect()

    # @classmethod
    def connect_to_microscope(self, ip_address: str, port: int = 8300) -> None:
        self.connection = Automation(ip_address, port)

    def acquire_image(self, image_settings=ImageSettings) -> FibsemImage:

        """
        Acquires an image from the FIB/SEM instrument using the specified settings. Also saves the acquired image
        to a last image instance so that the most recent image taken can be easily accessed without reacquiring an image.
        
        Parameters:
        self: The instance of the client connection to the TESCAN FIBSEM microscope.

        image_settings (ImageSettings): The settings to use when acquiring the image.
        
        Returns:
        FibsemImage: The image acquired by the microscope as a FibsemImage data type
        """

        if image_settings.beam_type.value == 1:
            image = self._get_eb_image(image_settings)
            self.last_image_eb = image
        if image_settings.beam_type.value == 2:
            image = self._get_ib_image(image_settings)
            self.last_image_ib = image

        return image

    def _get_eb_image(self, image_settings=ImageSettings) -> FibsemImage:
        """Gets an image with the electron beam with the specified settings

        Args:
            image_settings (_type_, optional): Image settings related to the electron beam. 
            Defaults to ImageSettings initially created.

        Returns:
            FibsemImage: returns the electron beam image in FibsemImage format
        """
        # At first make sure the beam is ON
        self.connection.SEM.Beam.On()
        # important: stop the scanning before we start scanning or before automatic procedures,
        # even before we configure the detectors
        self.connection.SEM.Scan.Stop()
        # Select the detector for image i.e.:
        # 1. assign the detector to a channel
        # 2. enable the channel for acquisition
        
        self.connection.SEM.Detector.Set(0, self.electron_detector_active, Bpp.Grayscale_16_bit)

        # resolution
        numbers = re.findall(r"\d+", image_settings.resolution)
        imageWidth = int(numbers[0])
        imageHeight = int(numbers[1])

        if image_settings.pixel_size is None:
            self.connection.SEM.Optics.SetViewfield(image_settings.hfw*1000)
        else:
            self.connection.SEM.Optics.SetViewfield(image_settings.pixel_size.x * imageWidth * 1000)
            image_settings.hfw = image_settings.pixel_size.x * imageWidth
        
        #dwell time conversion s to ns
        dwell_time_ns = int(image_settings.dwell_time * 1e9)

        image = self.connection.SEM.Scan.AcquireImageFromChannel(
            0, imageWidth, imageHeight, DwellTime=dwell_time_ns
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
            copy.deepcopy(image), copy.deepcopy(image_settings), copy.deepcopy(microscope_state)
        )

        return fibsem_image

    def _get_ib_image(self, image_settings=ImageSettings):
        """Gets an image with the electron beam with the specified settings

        Args:
            image_settings (_type_, optional): Image settings related to the ion beam. 
            Defaults to ImageSettings initially created.

        Returns:
            FibsemImage: returns the ion beam image in FibsemImage format
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

        # resolution
        numbers = re.findall(r"\d+", image_settings.resolution)
        imageWidth = int(numbers[0])
        imageHeight = int(numbers[1])

        if image_settings.pixel_size is None:
            self.connection.FIB.Optics.SetViewfield(image_settings.hfw*1000)
        else:
            self.connection.FIB.Optics.SetViewfield(image_settings.pixel_size.x * imageWidth * 1000)
            image_settings.hfw = image_settings.pixel_size.x * imageWidth

        
        #dwell time conversion s to ms
        dwell_time_ms = image_settings.dwell_time * 1000

        image = self.connection.FIB.Scan.AcquireImageFromChannel(
            0, imageWidth, imageHeight, dwell_time_ms
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
            copy.deepcopy(image), copy.deepcopy(image_settings), copy.deepcopy(microscope_state)
        )

        return fibsem_image

    def last_image(self, beam_type: BeamType.ELECTRON) -> FibsemImage:
        """Get the last image acquired for the specified beam type.

        Args:
            beam_type (BeamType): The type of beam to use (either electron or ion).

        Returns:
            image (FibsemImage): The last image acquired for the specified beam type.

        Raises:
            Exception: If the beam type is not electron or ion.
        """
        if beam_type == BeamType.ELECTRON:
            image = self.last_image_eb
        elif beam_type == BeamType.ION:
            image = self.last_image_ib
        else:
            raise Exception("Beam type error")
        return image

    def get_stage_position(self):
        """Get the current position of the stage.

        Returns:
            stage_position (FibsemStagePosition): An object containing the current 
            x, y, z, r, and t coordinates of the stage.
        """
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
        """Adjust the contrast of the active detector for the specified beam type.

        Args:
            beam_type (BeamType): The type of beam to use (either electron or ion).

        Returns:
            None
        """
        if beam_type == BeamType.ELECTRON:
            self.connection.SEM.Detector.AutoSignal(self.electron_detector_active.name)
        if beam_type == BeamType.ION:
            self.connection.FIB.Detector.AutoSignal(self.ion_detector_active.name)

    def reset_beam_shifts(self):
        pass


    def move_stage(self, x: float = None, y: float = None, z: float = None, r: float = None, tx: float = None, ty: float = None,):
        """Move the stage to the specified coordinates.

        Args:
            x (float): The x-coordinate to move to (in meters).
            y (float): The y-coordinate to move to (in meters).
            z (float): The z-coordinate to move to (in meters).
            r (float): The rotation to apply (in degrees).
            tx (float): The x-axis tilt to apply (in degrees).
            ty (float): The y-axis tilt to apply (in degrees).

        Returns:
            None
        """
        self.connection.Stage.MoveTo(x*1000,y*1000,z*1000,r,tx,ty)


