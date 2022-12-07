from abc import ABC, abstractmethod
import copy
import logging
import datetime
import numpy as np
from autoscript_sdb_microscope_client.structures import GrabFrameSettings
from autoscript_sdb_microscope_client.enumerations import CoordinateSystem
import sys
from tescanautomation import Automation

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem.structures import BeamType, ImageSettings, FibsemImage, FibsemImageMetadata, MicroscopeState, BeamSettings


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

    def acquire_image(
        self, image_settings=ImageSettings
    ) -> FibsemImage:
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
            hfw_limits = self.connection.beams.electron_beam.horizontal_field_width.limits
            image_settings.hfw = np.clip(image_settings.hfw, hfw_limits.min, hfw_limits.max)
            self.connection.beams.electron_beam.horizontal_field_width.value = image_settings.hfw
    
        if image_settings.beam_type == BeamType.ION:
            hfw_limits = self.connection.beams.ion_beam.horizontal_field_width.limits
            image_settings.hfw = np.clip(image_settings.hfw, hfw_limits.min, hfw_limits.max)
            self.connection.beams.ion_beam.horizontal_field_width.value = image_settings.hfw


        logging.info(f"acquiring new {image_settings.beam_type.name} image.")
        self.connection.imaging.set_active_view(image_settings.beam_type.value)
        self.connection.imaging.set_active_device(image_settings.beam_type.value)
        image = self.connection.imaging.grab_frame(frame_settings)

        state = self.get_current_microscope_state()

        fibsem_image = FibsemImage.fromAdornedImage(copy.deepcopy(image), copy.deepcopy(image_settings), copy.deepcopy(state))

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

        image_settings = FibsemImageMetadata.image_settings_from_adorned(image)

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
        self.connection.specimen.stage.set_default_coordinate_system(CoordinateSystem.RAW)
        stage_position = self.connection.specimen.stage.current_position
        self.connection.specimen.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
        return stage_position

    def get_current_microscope_state(self) -> MicroscopeState:
        """Get the current microscope state

        Returns:
            MicroscopeState: current microscope state
        """

        current_microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            # get absolute stage coordinates (RAW)
            absolute_position= self.get_stage_position,
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

    def __init__(self,ip_address: str ='localhost'):
        self.connection = Automation(ip_address)

    def disconnect(self):
        self.connection.Disconnect()
        

    # @classmethod
    def connect_to_microscope(self, ip_address: str, port: int = 8300) -> None:
        self.connection = Automation(ip_address,port)

    def acquire_image(
        self, frame_settings: GrabFrameSettings = None, image_settings=ImageSettings
    ) -> FibsemImage:
        pass

    def last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage:
        pass
    
    def autocontrast(self, beam_type=BeamType.ELECTRON) -> None:
        pass
    def reset_beam_shifts(self):
        pass