import copy
import datetime
import logging
import os
import sys
import threading
import time
import warnings
from copy import deepcopy
from queue import Queue
from typing import List, Union

import numpy as np

import fibsem.constants as constants
from fibsem.microscope import (
    FibsemMicroscope,
    _check_beam,
    _check_manipulator,
    _check_stage,
    _check_sputter,
)

_TESCAN_API_AVAILABLE = False

try:

    from tescanautomation import Automation
    from tescanautomation.Common import Bpp
    from tescanautomation.DrawBeam import IEtching
    from tescanautomation.DrawBeam import Status as DBStatus
    from tescanautomation.SEM import HVBeamStatus as SEMStatus

    sys.modules.pop("tescanautomation.GUI")
    sys.modules.pop("tescanautomation.pyside6gui")
    sys.modules.pop("tescanautomation.pyside6gui.imageViewer_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_utils")
    sys.modules.pop("tescanautomation.pyside6gui.rc_GUI")
    sys.modules.pop("tescanautomation.pyside6gui.workflow_private")
    sys.modules.pop("PySide6.QtCore")
    _TESCAN_API_AVAILABLE = True
except Exception as e:
    logging.debug(f"Automation (TESCAN) not installed. {e}")

from fibsem.structures import (
    BeamSettings,
    BeamSystemSettings,
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemExperiment,
    FibsemGasInjectionSettings,
    FibsemImage,
    FibsemImageMetadata,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    Point,
    SystemSettings,
)

def printProgressBar(
    value, total, prefix="", suffix="", decimals=0, length=100, fill="█"
):
    """
    terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (value / float(total)))
    filled_length = int(length * value // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")


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

        auto_focus(self, beam_type: BeamType) -> None:
            Automatically adjust the microscope focus for the specified beam type.

        beam_shift(self, dx: float, dy: float,  beam_type: BeamType) -> None:
            Adjusts the beam shift of given beam based on relative values that are provided.

        move_stage_absolute(self, position: FibsemStagePosition):
            Move the stage to the specified coordinates.

        move_stage_relative(self, position: FibsemStagePosition):
            Move the stage by the specified relative move.

        stable_move(self, dx: float, dy: float, beam_type: BeamType,) -> None:
            Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        vertical_move(self, dy: float, dx: float = 0.0, static_wd: bool = True) -> None:
            Move the stage vertically to correct eucentric point
                
        insert_manipulator(self, name: str) -> None:
            Insert the manipulator into the sample.
        
        retract_manipulator(self) -> None:
            Retract the manipulator from the sample.

        move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
            Move the manipulator by the specified relative move.
        
        move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
            Move the manipulator to the specified coordinates.

        move_manipulator_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
            Move the manipulator by the specified relative move, correcting for the beam type.      

        move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str) -> None:
            Move the manipulator to the specified position offset.

        _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
            Get the saved manipulator position with the specified name.
        setup_milling(self, mill_settings: FibsemMillingSettings):
            Configure the microscope for milling using the ion beam.

        run_milling(self, milling_current: float, asynch: bool = False):
            Run ion beam milling using the specified milling current.

        def run_milling_drift_corrected(self, milling_current: float, image_settings: ImageSettings, ref_image: FibsemImage, reduced_area: FibsemRectangle = None, asynch: bool = False):
        Run ion beam milling using the specified milling current, and correct for drift using the provided reference image.

        finish_milling(self, imaging_current: float):
            Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

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
        
        get(self, key:str, beam_type: BeamType = None):
            Returns the value of the specified key.

        set(self, key: str, value, beam_type: BeamType = None) -> None:
            Sets the value of the specified key.

    New methods:
        __init__(self): 
            Initializes a new instance of the class.

        _get_eb_image(self, image_settings=ImageSettings) -> FibsemImage:
            Acquires an electron beam (EB) image with the given settings and returns a FibsemImage object.

        _get_ib_image(self, image_settings=ImageSettings):
            Acquires an ion beam (IB) image with the given settings and returns a FibsemImage object.

        _y_corrected_stage_movement(self, expected_y: float, beam_type: BeamType = BeamType.ELECTRON) -> FibsemStagePosition:
            Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).
    """

    def __init__(self, system_settings: SystemSettings, ip_address: str = None):
        if _TESCAN_API_AVAILABLE == False:
            raise ImportError("The TESCAN Automation API is not available. Please see the user guide for installation instructions.")
        
        if ip_address is None:
            ip_address = system_settings.info.ip_address
        
        # create microscope client
        self.connection = Automation(ip_address)

        # set up detectors                
        detectors = self.connection.FIB.Detector.Enum()
        self.ion_detector_active = detectors[1] # find the se detector?
        self.connection.FIB.Detector.Set(Channel = 0, Detector= self.ion_detector_active)
        self.electron_detector_active = self.connection.SEM.Detector.SESuitable()
        self.connection.SEM.Detector.Set(Channel = 0, Detector = self.electron_detector_active)
        # TODO: rename to active_detector_beam_type
        # TODO: move to connect_to_microscope

        # initialise system settings
        self.system: SystemSettings = system_settings
        
        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

        # initialise last images
        self.last_image_eb: FibsemImage = None
        self.last_image_ib: FibsemImage = None

        # logging
        logging.debug({"msg": "create_microscope_client", "system_settings": system_settings.to_dict()})
    

    def disconnect(self) -> None:
        self.connection.Disconnect()
        del self.connection
        self.connection = None

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
        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        self.connection = Automation(ip_address, port)
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        self.connection.SEM.Detector.Set(0, self.electron_detector_active, Bpp.Grayscale_8_bit)
        image = self.connection.SEM.Scan.AcquireImageFromChannel(0, 1, 1, 100)

        # system info
        self.system.info.manufacturer = "TESCAN"
        self.system.info.model = image.Header["MAIN"]["DeviceModel"]
        self.system.info.serial_number = image.Header["MAIN"]["SerialNumber"]
        self.system.info.software_version = image.Header["MAIN"]["SoftwareVersion"]

        info = self.system.info
        logging.info(f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}.")

        # reset beam shifts
        self.reset_beam_shifts()

        logging.debug({"msg": "connect_to_microscope", "ip_address": ip_address, "port": port, "system_info": self.system.info.to_dict()})

    def acquire_image(self, image_settings=ImageSettings) -> FibsemImage:
        """
            Acquires an image using the specified image settings.

            Args:
                image_settings: An instance of the `ImageSettings` class that represents the image settings to use (default `ImageSettings`).

            Returns:
                A `FibsemImage` object that represents the acquired image.
        """
        if image_settings.beam_type.name == "ELECTRON":
            image_settings.hfw = np.clip(
                    image_settings.hfw, 1.0e-6, 2580.0e-6
                )
        else:
            image_settings.hfw = np.clip(
                    image_settings.hfw, 1.0e-6, 450.0e-6
                )
        logging.info(f"acquiring new {image_settings.beam_type.name} image.")
        if image_settings.beam_type.name == "ELECTRON":
            _check_beam(BeamType.ELECTRON, self.system)
            image = self._get_eb_image(image_settings)
            self.last_image_eb = image
        if image_settings.beam_type.name == "ION":
            _check_beam(BeamType.ION, self.system)
            image = self._get_ib_image(image_settings)
            self.last_image_ib = image

        image.metadata.user = self.user
        image.metadata.experiment = self.experiment 
        image.metadata.system = self.system

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
        status = self.connection.SEM.Beam.GetStatus()
        if status != Automation.SEM.Beam.Status.BeamOn:
            self.connection.SEM.Beam.On()
        # important: stop the scanning before we start scanning or before automatic procedures,
        # even before we configure the detectors
        self.connection.SEM.Scan.Stop()
        # Select the detector for image i.e.:
        # 1. assign the detector to a channel
        # 2. enable the channel for acquisition
        
        self.connection.SEM.Detector.Set(0, self.electron_detector_active, Bpp.Grayscale_8_bit)

        dwell_time = image_settings.dwell_time * constants.SI_TO_NANO
        # resolution
        imageWidth = image_settings.resolution[0]
        imageHeight = image_settings.resolution[1]

        self.connection.SEM.Optics.SetViewfield(
            image_settings.hfw * constants.METRE_TO_MILLIMETRE
        )
        if image_settings.reduced_area is not None:
            left =  int(image_settings.reduced_area.left * imageWidth)
            top = int(image_settings.reduced_area.top * imageHeight) 
            width = int(image_settings.reduced_area.width * imageWidth)
            height = int(image_settings.reduced_area.height * imageHeight)
            image = self.connection.SEM.Scan.AcquireROIFromChannel(
                Channel= 0,
                Width= imageWidth,
                Height= imageHeight,
                Left= left,
                Top= top,
                Right= left + width -1 ,
                Bottom= top + height - 1,
                DwellTime= dwell_time
            )
        else:
            image = self.connection.SEM.Scan.AcquireImageFromChannel(
                0, imageWidth, imageHeight, dwell_time
            )

        microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            stage_position=FibsemStagePosition(
                x=float(image.Header["SEM"]["StageX"]),
                y=float(image.Header["SEM"]["StageY"]),
                z=float(image.Header["SEM"]["StageZ"]),
                r=float(image.Header["SEM"]["StageRotation"]),
                t=float(image.Header["SEM"]["StageTilt"]),
                coordinate_system="RAW",
            ),
            electron_beam=BeamSettings(
                beam_type=BeamType.ELECTRON,
                working_distance=float(image.Header["SEM"]["WD"]),
                beam_current=float(image.Header["SEM"]["PredictedBeamCurrent"]),
                voltage=float(self.connection.SEM.Beam.GetVoltage()),
                resolution=[imageWidth, imageHeight], #"{}x{}".format(imageWidth, imageHeight),
                dwell_time=float(image.Header["SEM"]["DwellTime"]),
                stigmation=Point(
                    float(image.Header["SEM"]["StigmatorX"]),
                    float(image.Header["SEM"]["StigmatorY"]),
                ),
                shift=Point(
                    float(image.Header["SEM"]["ImageShiftX"]),
                    float(image.Header["SEM"]["ImageShiftY"]),
                ),
                scan_rotation=self.connection.SEM.Optics.GetImageRotation(),
            ),
            ion_beam=BeamSettings(beam_type=BeamType.ION),
        )

        detector = FibsemDetectorSettings(
                type = self.get("detector_type", image_settings.beam_type),
                mode = "N/A",
                contrast = self.get("detector_contrast", image_settings.beam_type),
                brightness= self.get("detector_brightness", image_settings.beam_type),

            )

        image_settings.resolution = [imageWidth, imageHeight]
        fibsem_image = FibsemImage.fromTescanImage(
            image, deepcopy(image_settings), deepcopy(microscope_state), detector= detector
        )

        #fibsem_image.metadata.image_settings.resolution = [imageWidth, imageHeight]

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
        status = self.connection.FIB.Beam.GetStatus()
        if status != Automation.FIB.Beam.Status.BeamOn:
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
            left =  int(image_settings.reduced_area.left * imageWidth)
            top = int(image_settings.reduced_area.top * imageHeight) 
            width = int(image_settings.reduced_area.width * imageWidth)
            height = int(image_settings.reduced_area.height * imageHeight)
            image = self.connection.FIB.Scan.AcquireROIFromChannel(
                Channel= 0,
                Width= imageWidth,
                Height= imageHeight,
                Left= left,
                Top= top,
                Right= left + width -1 ,
                Bottom= top + height - 1,
                DwellTime= dwell_time
            )
        else:
            image = self.connection.FIB.Scan.AcquireImageFromChannel(
                0, imageWidth, imageHeight, dwell_time
            )

        microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            stage_position=FibsemStagePosition(
                x=float(image.Header["FIB"]["StageX"]),
                y=float(image.Header["FIB"]["StageY"]),
                z=float(image.Header["FIB"]["StageZ"]),
                r=float(image.Header["FIB"]["StageRotation"]),
                t=float(image.Header["FIB"]["StageTilt"]),
                coordinate_system="RAW",
            ),
            electron_beam=BeamSettings(beam_type=BeamType.ELECTRON),
            ion_beam=BeamSettings(
                beam_type=BeamType.ION,
                working_distance=float(image.Header["FIB"]["WD"]),
                beam_current=float(self.connection.FIB.Beam.ReadProbeCurrent()),
                voltage = float(self.connection.FIB.Beam.GetVoltage()),
                resolution=[imageWidth, imageHeight], #"{}x{}".format(imageWidth, imageHeight),
                dwell_time=float(image.Header["FIB"]["DwellTime"]),
                stigmation=Point(
                    float(image.Header["FIB"]["StigmatorX"]),
                    float(image.Header["FIB"]["StigmatorY"]),
                ),
                shift=Point(
                    float(image.Header["FIB"]["ImageShiftX"]),
                    float(image.Header["FIB"]["ImageShiftY"]),
                ),
                scan_rotation=self.connection.FIB.Optics.GetImageRotation(),
            ),
        )

        detector = FibsemDetectorSettings(
                type = self.get("detector_type", image_settings.beam_type),
                mode = "N/A",
                contrast = self.get("detector_contrast", image_settings.beam_type),
                brightness= self.get("detector_brightness", image_settings.beam_type),

            )
        image_settings.resolution = [imageWidth, imageHeight]
        fibsem_image = FibsemImage.fromTescanImage(
            image, deepcopy(image_settings), deepcopy(microscope_state), detector= detector
        )

        # fibsem_image.metadata.image_settings.resolution = [imageWidth, imageHeight]

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
            _check_beam(BeamType.ELECTRON, self.system)
            image = self.last_image_eb
        elif beam_type == BeamType.ION:
            _check_beam(BeamType.ION, self.system)
            image = self.last_image_ib
        else:
            raise Exception("Beam type error")
        if image is not None:
            image.metadata.user = self.user
            image.metadata.experiment = self.experiment 
            image.metadata.system = self.system
        
        return image

    def _get_presets(self):
        presets = self.connection.FIB.Preset.Enum()	
        return presets

    def acquire_chamber_image(self) -> FibsemImage:
        """Acquire an image of the chamber inside."""
        image = self.connection.Camera.AcquireImage()
        logging.debug({"msg": "acquire_chamber_image"})
        return FibsemImage(data=np.array(image.Image), metadata=None)   

    def live_imaging(self, image_settings: ImageSettings, image_queue: Queue, stop_event: threading.Event):
        return
        # self.image_queue = image_queue
        # self.stop_event = stop_event
        # _check_beam(image_settings.beam_type, self.system)
        # logging.info(f"Live imaging: {image_settings.beam_type}")
        # while not self.stop_event.is_set():
        #     image = self.acquire_image(deepcopy(image_settings))
        #     image_queue.put(image)



    def consume_image_queue(self, parent_ui = None, sleep = 0.1):
        return
        # logging.info("Consuming image queue")

        # while not self.stop_event.is_set():
        #     try:
        #         time.sleep(sleep)
        #         if not self.image_queue.empty():
        #             image = self.image_queue.get(timeout=1)
        #             if image.metadata.image_settings.save:
        #                 image.metadata.image_settings.filename = f"{image.metadata.image_settings.filename}_{utils.current_timestamp()}"                        
        #                 filename = os.path.join(image.metadata.image_settings.path, image.metadata.image_settings.filename)
        #                 image.save(path=filename)
        #                 logging.info(f"Saved image to {filename}")

        #             logging.info(f"Image: {image.data.shape}")
        #             logging.info("-" * 50)

        #             if parent_ui is not None:
        #                     parent_ui.live_imaging_signal.emit({"image": image})


        #     except KeyboardInterrupt:
        #         self.stop_event
        #         logging.info("Keyboard interrupt, stopping live imaging")
        #     except Exception as e:
        #         self.stop_event.set()
        #         import traceback
        #         logging.error(traceback.format_exc())
        
    def autocontrast(self, beam_type: BeamType) -> None:
        """Automatically adjust the microscope image contrast for the specified beam type.

        Args:
            beam_type (BeamType, optional): The imaging beam type for which to adjust the contrast.
                Defaults to BeamType.ELECTRON.
        """
        _check_beam(beam_type, self.system)
        logging.info(f"Running autocontrast on {beam_type.name}.")
        if beam_type == BeamType.ELECTRON:
            self.connection.SEM.Detector.AutoSignal(Detector=self.electron_detector_active)
        if beam_type == BeamType.ION:
            self.connection.FIB.Detector.AutoSignal(Detector=self.ion_detector_active)

    
    def auto_focus(self, beam_type: BeamType) -> None:
        _check_beam(beam_type, self.system)
        if beam_type == BeamType.ELECTRON:
            logging.info("Running autofocus on electron beam.")
            self.connection.SEM.AutoWDFine(self.electron_detector_active)
        else:
            logging.info("Auto focus is not supported for ion beam type.")
        return 

    def reset_beam_shifts(self):
        """
        Set the beam shift to zero for the electron and ion beams.

        Resets the beam shift for both the electron and ion beams to (0,0), effectively centering the beams on the sample.

        Args:
            self (FibsemMicroscope): instance of the FibsemMicroscope object
        """
        _check_beam(BeamType.ELECTRON, self.system)
        logging.debug(
            f"reseting ebeam shift to (0, 0) from: {self.connection.FIB.Optics.GetImageShift()} (mm)"
        )
        self.connection.FIB.Optics.SetImageShift(0, 0)
        _check_beam(BeamType.ION, self.system)
        logging.debug(
            f"reseting ebeam shift to (0, 0) from: {self.connection.SEM.Optics.GetImageShift()} (mm)"
        )
        self.connection.SEM.Optics.SetImageShift(0, 0)


    def beam_shift(self, dx: float, dy: float, beam_type: BeamType = BeamType.ION):
        """Adjusts the beam shift based on relative values that are provided.
        
        Args:
            self (FibsemMicroscope): Fibsem microscope object
            dx (float): the relative x term
            dy (float): the relative y term
        """
        _check_beam(beam_type, self.system)
        if beam_type == BeamType.ION:
            beam = self.connection.FIB.Optics
        elif beam_type == BeamType.ELECTRON:
            beam = self.connection.SEM.Optics
        logging.info(f"{beam_type.name} shifting by ({dx}, {dy})")
        x, y = beam.GetImageShift()
        dx *=  constants.METRE_TO_MILLIMETRE # Convert to mm from m.
        dy *=  constants.METRE_TO_MILLIMETRE
        x += dx 
        y += dy
        beam.SetImageShift(x,y) 

        logging.debug({"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name}) 
        
    def get_stage_position(self):
        """
        Get the current stage position.

        This method retrieves the current stage position from the microscope and returns it as
        a FibsemStagePosition object.

        Returns:
            FibsemStagePosition: The current stage position.
        """
        if self.system.stage.enabled is False:
            raise NotImplementedError("Stage is not enabled.")
        x, y, z, r, t = self.connection.Stage.GetPosition()
        stage_position = FibsemStagePosition(
            x = x * constants.MILLIMETRE_TO_METRE,
            y = y * constants.MILLIMETRE_TO_METRE,
            z = z * constants.MILLIMETRE_TO_METRE,
            r = r * constants.DEGREES_TO_RADIANS,
            t = t * constants.DEGREES_TO_RADIANS,
            coordinate_system= "RAW",
        )
        logging.debug({"msg": "get_stage_position", "pos": stage_position.to_dict()})
        return stage_position

    def get_microscope_state(self) -> MicroscopeState:
        """
        Get the current microscope state

        This method retrieves the current microscope state from the microscope and returns it as
        a MicroscopeState object.

        Returns:
            MicroscopeState: current microscope state
        """

        if self.system.electron is True:
            image_eb = self.last_image(BeamType.ELECTRON)
            if image_eb is not None:
                electron_beam = BeamSettings(
                beam_type=BeamType.ELECTRON,
                working_distance=self.connection.SEM.Optics.GetWD() * constants.MILLIMETRE_TO_METRE,
                beam_current=self.connection.SEM.Beam.GetCurrent() * constants.PICO_TO_SI,
                voltage=self.connection.SEM.Beam.GetVoltage(),
                hfw=self.connection.SEM.Optics.GetViewfield() * constants.MILLIMETRE_TO_METRE,
                resolution=image_eb.metadata.image_settings.resolution,  # TODO fix these empty parameters
                dwell_time=image_eb.metadata.image_settings.dwell_time,
                stigmation=image_eb.metadata.microscope_state.electron_beam.stigmation,
                shift=image_eb.metadata.microscope_state.electron_beam.shift,
                scan_rotation=self.connection.SEM.Optics.GetImageRotation()
            )
            else:
                electron_beam = BeamSettings(BeamType.ELECTRON)
        else:
            electron_beam = BeamSettings(BeamType.ELECTRON)
        
        if self.system.ion is True:
            image_ib = self.last_image(BeamType.ION)
            if image_ib is not None:
                ion_beam = BeamSettings(
                        beam_type=BeamType.ION,
                        working_distance=image_ib.metadata.microscope_state.ion_beam.working_distance,
                        beam_current=self.connection.FIB.Beam.ReadProbeCurrent() * constants.PICO_TO_SI,
                        voltage=self.connection.FIB.Beam.GetVoltage(),
                        hfw=self.connection.FIB.Optics.GetViewfield() * constants.MILLIMETRE_TO_METRE,
                        resolution=image_ib.metadata.image_settings.resolution,
                        dwell_time=image_ib.metadata.image_settings.dwell_time,
                        stigmation=image_ib.metadata.microscope_state.ion_beam.stigmation,
                        shift=image_ib.metadata.microscope_state.ion_beam.shift,
                        scan_rotation=self.connection.FIB.Optics.GetImageRotation()
                    )
                
            else:
                ion_beam = BeamSettings(BeamType.ION)
        else:
            ion_beam = BeamSettings(BeamType.ION)
    
        electron_detector = self.get_detector_settings(BeamType.ELECTRON)
        ion_detector = self.get_detector_settings(BeamType.ION)

        current_microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            # get absolute stage coordinates (RAW)
            stage_position=self.get_stage_position(),
            # electron beam settings
            electron_beam=electron_beam,
            # ion beam settings
            ion_beam=ion_beam,
            # electron detector settings
            electron_detector=electron_detector,
            # ion detector settings
            ion_detector=ion_detector,
        )

        logging.debug({"msg": "get_microscope_state", "state": current_microscope_state.to_dict()})

        return current_microscope_state

    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition
        ) -> None:

        # TODO: implement if required.
        self.move_stage_absolute(stage_position)
    
    def project_stable_move(self, dx:float, dy:float, beam_type:BeamType, base_position:FibsemStagePosition) -> FibsemStagePosition:
        if beam_type == BeamType.ELECTRON:
            image_rotation = self.connection.SEM.Optics.GetImageRotation()
        else:
            image_rotation = self.connection.FIB.Optics.GetImageRotation()

        if np.isnan(image_rotation):
            image_rotation = 0.0

        dx =  -(dx*np.cos(image_rotation*np.pi/180) + dy*np.sin(image_rotation*np.pi/180))
        dy = -(dy*np.cos(image_rotation*np.pi/180) - dx*np.sin(image_rotation*np.pi/180))
        point_yz = self._y_corrected_stage_movement(dy, beam_type)
        dy, dz = point_yz.y, point_yz.z

        # calculate the corrected move to reach that point from base-state?
        _new_position = deepcopy(base_position)
        _new_position.x += dx
        _new_position.y += dy
        _new_position.z += dz

        return _new_position # TODO: implement

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
        if position.r is not None:
            rotation = True
        else:
            rotation = False
        if position.t is not None:
            tilt = True
        else:
            tilt = False
        _check_stage(self.system, rotation=rotation, tilt=tilt)
        logging.info(f"Moving stage to {position}.")
        self.connection.Stage.MoveTo(
            position.x * constants.METRE_TO_MILLIMETRE if position.x is not None else None,
            position.y * constants.METRE_TO_MILLIMETRE if position.y is not None else None,
            position.z * constants.METRE_TO_MILLIMETRE if position.z is not None else None,
            position.r * constants.RADIANS_TO_DEGREES if position.r is not None else None,
            position.t * constants.RADIANS_TO_DEGREES if position.t is not None else None,
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
        if position.r is not None:
            rotation = True
        else:
            rotation = False
        if position.t is not None:
            tilt = True
        else:
            tilt = False
        _check_stage(self.system, rotation=rotation, tilt=tilt)
        logging.info(f"Moving stage by {position}.")
        # current_position = self.get_stage_position()
        # x2,y2,z2 = self.connection.Stage.KVF.Compute(
        #     wd= self.get("working_distance", beam_type=BeamType.ELECTRON),
        #     x1= current_position.x * constants.METRE_TO_MILLIMETRE,
        #     y1= current_position.y * constants.METRE_TO_MILLIMETRE,
        #     z1= current_position.z * constants.METRE_TO_MILLIMETRE,
        #     r1= current_position.r * constants.RADIANS_TO_DEGREES,
        #     tx1= current_position.t * constants.RADIANS_TO_DEGREES,
        #     ty1 = 0,
        #     r2 = position.r * constants.RADIANS_TO_DEGREES + current_position.r * constants.RADIANS_TO_DEGREES,
        #     tx2 = position.t * constants.RADIANS_TO_DEGREES + current_position.t * constants.RADIANS_TO_DEGREES,
        #     ty2 = 0,
        # )
        # self.move_stage_absolute(FibsemStagePosition(x2,y2,z2))
        current_position = self.get_stage_position()
        x_m = current_position.x
        y_m = current_position.y
        z_m = current_position.z
        new_position = FibsemStagePosition(
            x = (x_m + position.x) if position.x is not None else x_m,
            y = (y_m + position.y )if position.y is not None else y_m,
            z = (z_m + position.z )if position.z is not None else z_m,
            r = (current_position.r + position.r) if position.r is not None else current_position.r,
            t = (current_position.t + position.t) if position.t is not None else current_position.t,
            coordinate_system =  "RAW",
        )
        self.move_stage_absolute(new_position)

    def stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        static_wd: bool = False,
    ) -> None:
        """
        Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        Args:
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
        """
        _check_stage(self.system)
        wd = self.connection.SEM.Optics.GetWD()

        if beam_type == BeamType.ELECTRON:
            image_rotation = self.connection.SEM.Optics.GetImageRotation()
        else:
            image_rotation = self.connection.FIB.Optics.GetImageRotation()

        if np.isnan(image_rotation):
            image_rotation = 0.0
        # if image_rotation == 0:
        #     dx_move = -dx
        #     dy_move = dy
        # elif image_rotation == 180:
        #     dx_move = dx
        #     dy_move = -dy

        dx_move =  -(dx*np.cos(image_rotation*np.pi/180) + dy*np.sin(image_rotation*np.pi/180))
        dy_move = -(dy*np.cos(image_rotation*np.pi/180) - dx*np.sin(image_rotation*np.pi/180))

        # calculate stage movement
        x_move = FibsemStagePosition(x=dx_move, y=0, z=0) 
        yz_move = self._y_corrected_stage_movement(
            expected_y=dy_move,
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

        return

    def vertical_move(
        self,
        dy: float,
        dx: float = 0.0,
        static_wd: bool = True,
    ) -> None:
        """
        Move the stage vertically to correct eucentric point

        Args:
            dy (float): distance in y-axis (image coordinates)
        """
        _check_stage(self.system)
        wd = self.connection.SEM.Optics.GetWD()
        image_rotation = self.connection.FIB.Optics.GetImageRotation()
            
        if np.isclose(image_rotation, 0.0):
            dy_move = dy
        elif np.isclose(image_rotation, 180):
            dy_move = -dy


            
        PRETILT_SIGN = 1.0
        from fibsem import movement
        # current stage position
        current_stage_position = self.get_stage_position()
        stage_rotation = current_stage_position.r % (2 * np.pi)
        stage_tilt = current_stage_position.t
        stage_tilt_flat_to_electron = np.deg2rad(self.system.electron.column_tilt)
        stage_tilt_flat_to_ion = np.deg2rad(self.system.ion.column_tilt)

        stage_rotation_flat_to_ion = np.deg2rad(
            self.system.stage.rotation_180
        ) % (2 * np.pi)

        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        # TODO: check this pre-tilt angle calculation
        corrected_pretilt_angle = PRETILT_SIGN * (stage_tilt_flat_to_electron - self.system.stage.shuttle_pre_tilt*constants.DEGREES_TO_RADIANS)
        perspective_tilt = (- corrected_pretilt_angle - stage_tilt_flat_to_ion)
        z_perspective = - dy_move/np.cos((stage_tilt + corrected_pretilt_angle + perspective_tilt))
        z_move = z_perspective*np.sin(90*constants.DEGREES_TO_RADIANS - stage_tilt_flat_to_ion) 
        # z_move = dy / np.cos(
        #     np.deg2rad(90 - stage_tilt_flat_to_ion + sself.system.stage.shuttle_pre_tilt)
        # )  # TODO: MAGIC NUMBER, 90 - fib tilt
        logging.info(f"eucentric movement: {z_move}")
        z_move = FibsemStagePosition(x=dx, y=0, z=z_move, r=0, t=0)
        self.move_stage_relative(z_move)

        self.connection.SEM.Optics.SetWD(wd)

    def _y_corrected_stage_movement(
        self,
        expected_y: float,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> FibsemStagePosition:
        """
        Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).

        Args:
            expected_y (float, optional): distance along y-axis.
            beam_type (BeamType, optional): beam_type to move in. Defaults to BeamType.ELECTRON.

        Returns:
            StagePosition: y corrected stage movement (relative position)
        """

        # TODO: replace with camera matrix * inverse kinematics
        # TODO: replace stage_tilt_flat_to_electron with pre-tilt

        # all angles in radians
        stage_tilt_flat_to_electron = np.deg2rad(self.system.electron.column_tilt)
        stage_tilt_flat_to_ion = np.deg2rad(self.system.ion.column_tilt)

        stage_rotation_flat_to_ion = np.deg2rad(
            self.system.stage.rotation_180
        ) % (2 * np.pi)

        # current stage position
        current_stage_position = self.get_stage_position()
        stage_rotation = current_stage_position.r % (2 * np.pi)
        stage_tilt = current_stage_position.t

        PRETILT_SIGN = 1.0
        from fibsem import movement

        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        corrected_pretilt_angle = PRETILT_SIGN * (stage_tilt_flat_to_electron - self.system.stage.shuttle_pre_tilt*constants.DEGREES_TO_RADIANS)
        
        perspective_tilt = - corrected_pretilt_angle if beam_type is BeamType.ELECTRON else (- corrected_pretilt_angle - stage_tilt_flat_to_ion)

        y_move = expected_y/np.cos((stage_tilt + corrected_pretilt_angle + perspective_tilt))
         
        z_move = y_move*np.sin((corrected_pretilt_angle)) 
        print(f'Stage tilt: {stage_tilt}, corrected pretilt: {corrected_pretilt_angle}, y_move: {y_move} z_move: {z_move}')

        return FibsemStagePosition(x=0, y=y_move, z=z_move)

    def move_flat_to_beam(
        self, beam_type: BeamType = BeamType.ELECTRON, _safe: bool = True
    ):
        """
        Moves the microscope stage to the tilt angle corresponding to the given beam type,
        so that the stage is flat with respect to the beam.

        Args:
            beam_type (BeamType): The type of beam to which the stage should be made flat.
                Must be one of BeamType.ELECTRON or BeamType.ION.

        Returns:
            None.
        """
        _check_beam(beam_type, self.system)
        # BUG if I set or pass BeamType.ION it still sees beam_type as BeamType.ELECTRON
    
        if beam_type is BeamType.ION:
            tilt = self.system.ion.column_tilt
        elif beam_type is BeamType.ELECTRON:
            tilt = self.system.electron.column_tilt 
        #TODO: no pre-tilt fix this

        logging.info(f"Moving Stage Flat to {beam_type.name} Beam")
        self.connection.Stage.MoveTo(tiltx=tilt)


    def get_manipulator_state(self) -> bool:

        """returns true if nanomanipulator is inserted. Manipulator positions must be calibrated and stored in system.yaml file if not done so

        Raises:
            ValueError: _description_

        Returns:
            _type_: True if Inserted, False if retracted
        """

        manipulator_positions = cfg.load_tescan_manipulator_calibration()

        if not manipulator_positions["calibrated"]:
            logging.warning("Manipulator positions not calibrated, cannot get state")
            return False

        retracted_position_x = manipulator_positions["parking"]["x"]*constants.METRE_TO_MILLIMETRE
        retracted_position_y = manipulator_positions["parking"]["y"]*constants.METRE_TO_MILLIMETRE
        retracted_position_z = manipulator_positions["parking"]["z"]*constants.METRE_TO_MILLIMETRE

        current_position = self.get_manipulator_position()

        current_position_array = [current_position.x*constants.METRE_TO_MILLIMETRE, current_position.y*constants.METRE_TO_MILLIMETRE, current_position.z*constants.METRE_TO_MILLIMETRE]

        check_compare = np.isclose(current_position_array, [retracted_position_x, retracted_position_y, retracted_position_z], atol=0.1)

        return True if False in check_compare else False
            

    def get_manipulator_position(self) -> FibsemManipulatorPosition:
        # pass
        _check_manipulator(self.system)
        index = 0
        output_position = self.connection.Nanomanipulator.GetPosition(Index=index)

        # GetPosition returns tuple in the form (x, y, z, r)
        # x,y,z in mm and r in degrees, no tilt information

        x = output_position[0]*constants.MILLIMETRE_TO_METRE
        y = output_position[1]*constants.MILLIMETRE_TO_METRE
        z = output_position[2]*constants.MILLIMETRE_TO_METRE
        r = output_position[3]*constants.DEGREES_TO_RADIANS

        return FibsemManipulatorPosition(x=x, y=y, z=z, r=r)



    def insert_manipulator(self, name: str = "Standby"):
        _check_manipulator(self.system)
        preset_positions = ["Parking","Standby","Working",]

        if name == "PARK":
            name = "Parking"

        for position in preset_positions:
            if name.lower() == position.lower():
                name = position
    

        if name not in preset_positions:
            raise ValueError(f"Position {name} is not a valid preset position. Valid positions are {preset_positions}.")


        insert_position = getattr(self.connection.Nanomanipulator.Position,name)

        index = 0
        logging.info(f"Inserting Nanomanipulator to {name} position")
        self.connection.Nanomanipulator.MoveToPosition(Index=index,Position=insert_position)

    def _check_manipulator_limits(self,x,y,z,r):

        limits = self.connection.Nanomanipulator.GetLimits(Index=0,Type=0)

        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]
        zmin = limits[4]
        zmax = limits[5]
        rmin = limits[6]
        rmax = limits[7]

        assert x >= xmin and x <= xmax, f"X position {x} is outside of manipulator limits {xmin} to {xmax}"
        assert y >= ymin and y <= ymax, f"Y position {y} is outside of manipulator limits {ymin} to {ymax}"
        assert z >= zmin and z <= zmax, f"Z position {z} is outside of manipulator limits {zmin} to {zmax}"
        assert r >= rmin and r <= rmax, f"R position {r} is outside of manipulator limits {rmin} to {rmax}"
    
    def retract_manipulator(self):
        retract_position = getattr(self.connection.Nanomanipulator.Position,"Parking")
        index = 0
        self.connection.Nanomanipulator.MoveToPosition(Index=index,Position=retract_position)
        

    
    def move_manipulator_relative(self,position: FibsemManipulatorPosition, name: str = None):
        if not np.isclose(position.r, 0.0):
            rotation = True
        else:
            rotation = False
        if not np.isclose(position.t, 0.0):
            tilt = True
        else:
            tilt = False
        _check_manipulator(self.system, rotation, tilt)
        if self.connection.Nanomanipulator.IsCalibrated(0) == False:
            logging.info("Calibrating manipulator")
            self.connection.Nanomanipulator.Calibrate(0)

        current_position = self.get_manipulator_position()
        
        x = (current_position.x + position.x)*constants.METRE_TO_MILLIMETRE
        y = (current_position.y + position.y)*constants.METRE_TO_MILLIMETRE
        z = (current_position.z + position.z)*constants.METRE_TO_MILLIMETRE
        r = (current_position.r + position.r)*constants.RADIANS_TO_DEGREES
        index = 0

        # self._check_manipulator_limits(x,y,z,r)

        logging.info(f"moving manipulator by {position}")
        try:
            self.connection.Nanomanipulator.MoveTo(Index=index,X=x, Y=y, Z=z, Rot=r)
        except Exception as e:
            logging.error(e)
            return e

    
    def move_manipulator_absolute(self, position: FibsemManipulatorPosition, name: str = None):
        if not np.isclose(position.r, 0.0):
            rotation = True
        else:
            rotation = False
        if not np.isclose(position.t, 0.0):
            tilt = True
        else:
            tilt = False
        _check_manipulator(self.system, rotation, tilt)
        if self.connection.Nanomanipulator.IsCalibrated(0) == False:
            logging.info("Calibrating manipulator")
            self.connection.Nanomanipulator.Calibrate(0)
        
        x = position.x*constants.METRE_TO_MILLIMETRE
        y = position.y*constants.METRE_TO_MILLIMETRE
        z = position.z*constants.METRE_TO_MILLIMETRE
        r = position.r*constants.RADIANS_TO_DEGREES
        index = 0

        # self._check_manipulator_limits(x,y,z,r)

        logging.info(f"moving manipulator to {position}")

        self.connection.Nanomanipulator.MoveTo(Index=index, X=x, Y=y, Z=z, Rot=r)

    def calibrate_manipulator(self):
        _check_manipulator(self.system)
        logging.info("Calibrating manipulator")
        self.connection.Nanomanipulator.Calibrate(0)

    def _x_corrected_needle_movement(self, expected_x: float) -> FibsemManipulatorPosition:
        """Calculate the corrected needle movement to move in the x-axis.

        Args:
            expected_x (float): distance along the x-axis (image coordinates)
        Returns:
            FibsemManipulatorPosition: x-corrected needle movement (relative position)
        """
        return FibsemManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed


    def _y_corrected_needle_movement(self, 
        expected_y: float, stage_tilt: float
    ) -> FibsemManipulatorPosition:
        """Calculate the corrected needle movement to move in the y-axis.

        Args:
            expected_y (float): distance along the y-axis (image coordinates)
            stage_tilt (float, optional): stage tilt.

        Returns:
            FibsemManipulatorPosition: y-corrected needle movement (relative position)
        """
        y_move = +np.cos(stage_tilt) * expected_y
        z_move = +np.sin(stage_tilt) * expected_y
        return FibsemManipulatorPosition(x=0, y=y_move, z=z_move)


    def _z_corrected_needle_movement(self, 
        expected_z: float, stage_tilt: float
    ) -> FibsemManipulatorPosition:
        """Calculate the corrected needle movement to move in the z-axis.

        Args:
            expected_z (float): distance along the z-axis (image coordinates)
            stage_tilt (float, optional): stage tilt.

        Returns:
            FibsemManipulatorPosition: z-corrected needle movement (relative position)
        """
        y_move = -np.sin(stage_tilt) * expected_z
        z_move = +np.cos(stage_tilt) * expected_z
        return FibsemManipulatorPosition(x=0, y=y_move, z=z_move)

    def move_manipulator_corrected(self, 
        dx: float = 0,
        dy: float = 0,
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
        _check_manipulator(self.system)

        if self.connection.Nanomanipulator.IsCalibrated(0) == False:
            logging.info("Calibrating manipulator")
            self.connection.Nanomanipulator.Calibrate(0)
        stage_tilt = self.get_stage_position().t


        # # xy
        # if beam_type is BeamType.ELECTRON:
        #     x_move = self._x_corrected_needle_movement(expected_x=dx)
        #     yz_move = self._y_corrected_needle_movement(dy, stage_tilt=stage_tilt)

        # # xz,
        # if beam_type is BeamType.ION:

        #     x_move = self._x_corrected_needle_movement(expected_x=dx)
        #     yz_move = self._z_corrected_needle_movement(expected_z=dy, stage_tilt=stage_tilt)

        # move needle (relative)
        #self.connection.Nanomanipulator.MoveTo(Index=0, X=x_move.x, Y=yz_move.y, Z=yz_move.z)
        self.move_manipulator_relative(FibsemManipulatorPosition(x=dx, y=dy, z=0))

        return

    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str = None) -> None:
        logging.warning("Not supported by TESCAN API")
        # raise NotImplementedError("Not supported by TESCAN API")
        # _check_manipulator_movement(self.system, offset)
        pass

    def _get_saved_manipulator_position(self):
        _check_manipulator(self.system)
        logging.warning("Not supported by TESCAN API")
        pass

    def setup_milling(
        self,
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
        _check_beam(BeamType.ION, self.system)

        spot_size = mill_settings.spot_size  # application_file
        rate = mill_settings.rate  ## in application file called Volume per Dose (m3/C)
        dwell_time = mill_settings.dwell_time  # in seconds ## in application file

        if mill_settings.patterning_mode == "Serial":
            parallel_mode = False
        else:
            parallel_mode = True

        print(f"spacing: {mill_settings.spacing}")

        self.set("preset", mill_settings.preset, BeamType.ION)
        beam_current = self.connection.FIB.Beam.ReadProbeCurrent()*constants.PICO_TO_SI
        print(f"beam_current: {beam_current}")
        layer_settings = IEtching(
            syncWriteField=False,
            writeFieldSize=mill_settings.hfw,
            beamCurrent=beam_current,
            spotSize=spot_size,
            rate=rate,
            dwellTime=dwell_time,
            parallel=parallel_mode,
            preset = mill_settings.preset,
            spacing = mill_settings.spacing,
        )

        self.layer = self.connection.DrawBeam.Layer("Layer1", layer_settings)
        

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False):
        """
        Runs the ion beam milling process using the specified milling current.

        Args:
            milling_current (float): The milling current to use, in amps.
            asynch (bool, optional): Whether to run the milling asynchronously. Defaults to False.

        Returns:
            None
        """
        _check_beam(BeamType.ION, self.system)
        status = self.connection.FIB.Beam.GetStatus()
        if status != Automation.FIB.Beam.Status.BeamOn:
            self.connection.FIB.Beam.On()
        self.connection.DrawBeam.LoadLayer(self.layer)
        logging.info("running ion beam milling now...")
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

    def run_milling_drift_corrected(self, milling_current: float,  
        image_settings: ImageSettings, 
        ref_image: FibsemImage, 
        reduced_area: FibsemRectangle = None,
        asynch: bool = False
        ):
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
        _check_beam(BeamType.ION, self.system)
        status = self.connection.FIB.Beam.GetStatus()
        if status != Automation.FIB.Beam.Status.BeamOn:
            self.connection.FIB.Beam.On()
        self.connection.DrawBeam.LoadLayer(self.layer)
        logging.info("running ion beam milling now...")
        self.connection.DrawBeam.Start()
        self.connection.Progress.Show(
            "DrawBeam", "Layer 1 in progress", False, False, 0, 100
        )
        from fibsem import alignment
        while True:
            status = self.connection.DrawBeam.GetStatus()
            running = status[0] == DBStatus.ProjectLoadedExpositionInProgress
            if running:
                progress = 0
                if status[1] > 0:
                    progress = min(100, status[2] / status[1] * 100)
                printProgressBar(progress, 100)
                self.connection.Progress.SetPercents(progress)
                status = self.connection.DrawBeam.GetStatus()
                if status[0] == DBStatus.ProjectLoadedExpositionInProgress:
                    self.connection.DrawBeam.Pause()
                elif status[0] == DBStatus.ProjectLoadedExpositionIdle:
                    printProgressBar(100, 100, suffix="Finished")
                    self.connection.DrawBeam.Stop()
                    self.connection.DrawBeam.UnloadLayer()
                    break
                logging.info("Drift correction in progress...")
                image_settings.beam_type = BeamType.ION
                alignment.beam_shift_alignment(
                    self,
                    image_settings,
                    ref_image,
                    reduced_area,
                )
                time.sleep(1)
                status = self.connection.DrawBeam.GetStatus()
                if status[0] == DBStatus.ProjectLoadedExpositionPaused :
                    self.connection.DrawBeam.Resume()
                logging.info("Drift correction complete.")
                time.sleep(5)
            else:
                if status[0] == DBStatus.ProjectLoadedExpositionIdle:
                    printProgressBar(100, 100, suffix="Finished")
                    self.connection.DrawBeam.Stop()
                    self.connection.DrawBeam.UnloadLayer()
                break

        print()  # new line on complete
        self.connection.Progress.Hide()

    def finish_milling(self, imaging_current: float):
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        # """
        try:
            self.connection.FIB.Preset.Activate("30 keV; 150 pA")
            self.connection.DrawBeam.UnloadLayer()
            print("hello")
        except Exception as e:
            logging.debug(f"Error in finish_milling: {e}")
            pass
    
    def stop_milling(self):
        pass

    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings):
        pass

    def estimate_milling_time(self,patterns):
        
        # load and unload layer to check time
        self.connection.DrawBeam.LoadLayer(self.layer)
        est_time = self.connection.DrawBeam.EstimateTime() 
        self.connection.DrawBeam.UnloadLayer()

        return est_time

    def draw_rectangle(
        self,
        pattern_settings: FibsemRectangleSettings,
    ):
        """
        Draws a rectangle pattern using the current ion beam.

        Args:
            pattern_settings (FibsemRectangleSettings): the settings for the pattern to draw.

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
        paths = self.get_available_values(key="scan_direction")
        passes = pattern_settings.passes if pattern_settings.passes is not None else 1.0
        if pattern_settings.scan_direction in paths:
            path = pattern_settings.scan_direction
        else:
            path = "Flyback"
            logging.info(f"Scan direction {pattern_settings.scan_direction} not supported. Using Flyback instead.")
            logging.info(f"Supported scan directions are: Flyback, RLE, SpiralInsideOut, SpiralOutsideIn, ZigZag")
        self.connection.DrawBeam.ScanningPath = pattern_settings.scan_direction

        # TODO: replace with cross_section parameter
        if pattern_settings.cleaning_cross_section:
            self.layer.addRectanglePolish(
                CenterX=centre_x,
                CenterY=centre_y,
                Depth=depth,
                DepthUnit='m',
                Width=width,
                Height=height,
                Angle=rotation,
                ScanningPath=path,
                ExpositionFactor=passes
            )
        else:
            self.layer.addRectangleFilled(
                CenterX=centre_x,
                CenterY=centre_y,
                Depth=depth,
                DepthUnit='m',
                Width=width,
                Height=height,
                Angle=rotation,
                ScanningPath=path,
                ExpositionFactor=passes
            )

        pattern = self.layer
        
        return pattern

    def draw_line(self, pattern_settings: FibsemLineSettings):
        """
        Draws a line pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemLineSettings): A data class object specifying the pattern parameters,
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
            BeginX=start_x, BeginY=start_y, EndX=end_x, EndY=end_y, Depth=depth, DepthUnit='m',
        )

        pattern = self.layer
        return pattern

    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        """
        Draws a circle pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemCircleSettings): A data class object specifying the pattern parameters,
                including the centre point, radius and depth of the pattern.

        Returns:
            CirclePattern: A circle pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.

            
        """
        if pattern_settings.cleaning_cross_section:
            pattern = self.layer.addAnnulusPolish(
                CenterX=pattern_settings.centre_x,
                CenterY=pattern_settings.centre_y,
                RadiusA=pattern_settings.radius,
                RadiusB=0,
                Depth=pattern_settings.depth,
                DepthUnit='m',
            )
        else:
            pattern = self.layer.addAnnulusFilled(
                CenterX=pattern_settings.centre_x,
                CenterY=pattern_settings.centre_y,
                RadiusA=pattern_settings.radius,
                RadiusB=0,
                Depth=pattern_settings.depth,
                DepthUnit='m',
            )

        return pattern
    
    def draw_annulus(self,pattern_settings: FibsemCircleSettings):

        """Draws an annulus (donut) pattern on the current imaging view of the microscope.

        Args: 
            pattern_settings (FibsemCircleSettings): A data class object specifying the pattern parameters,
            including the centre point, outer radius and thickness of the annulus, and the depth of the pattern.

        Returns:
            annulus pattern object
        """
        outer_radius = pattern_settings.radius
        inner_radius = pattern_settings.radius - pattern_settings.thickness


        pattern = self.layer.addAnnulusFilled(
            CenterX=pattern_settings.centre_x,
            CenterY=pattern_settings.centre_y,
            RadiusA=outer_radius,
            RadiusB=inner_radius,
            Depth=pattern_settings.depth,
            DepthUnit='m',
        )

        return pattern
    
    def draw_bitmap_pattern(
        self,
        pattern_settings: FibsemBitmapSettings,
        path: str,
    ):
        return NotImplemented

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
        _check_sputter(self.system)
        gas = protocol["gas"]
        self.connection.FIB.Beam.On()
        lines = self.connection.GIS.Enum()
        for line in lines:
            if line.name == gas:
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
                writeFieldSize=protocol["hfw"],
                beamCurrent=protocol["beam_current"],
                spotSize=protocol["spot_size"],
                rate=3e-10, # Value for platinum
                dwellTime=protocol["dwell_time"],
            )
            self.layer = self.connection.DrawBeam.Layer("Layer1", layerSettings)
            self.connection.DrawBeam.LoadLayer(self.layer)

        except Exception as e:
            logging.debug(f"Error setting up sputter: {e}")
            import fibsem
            base_path = os.path.dirname(fibsem.__path__[0])
            layer_path = os.path.join(base_path,"fibsem", "config", "deposition.dbp")
            self.layer = self.connection.DrawBeam.Layer.fromDbp(layer_path)[0]
            # self.layer = self.connection.DrawBeam.LoadLayer(defaultLayerSettings[0])

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
        _check_sputter(self.system)
        # Open GIS valve to let the gas flow onto the sample
        self.connection.GIS.OpenValve(self.line)

        try:
            # Run predefined deposition process
            self.connection.DrawBeam.Start()
            self.connection.Progress.Show("DrawBeam", "Layer 1 in progress", False, False, 0, 100)
            logging.info("Sputtering with platinum started.")
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
        _check_sputter(self.system)
        # Move GIS out from chamber and turn off heating
        self.connection.GIS.MoveTo(self.line, Automation.GIS.Position.Home)
        self.connection.GIS.PrepareTemperature(self.line, False)
        self.connection.DrawBeam.UnloadLayer()
        logging.info("Platinum sputtering process completed.")

    # def setup_GIS(self,protocol) -> None:

    #     beam_type = protocol["beam_type"]

    #     if beam_type == "ION":


    #         layerSettings = self.connection.DrawBeam.LayerSettings.IDeposition(
    #             syncWriteField=True,
    #             writeFieldSize=protocol.get("hfw",0.0005),
    #             beamCurrent=protocol.get("beam_current",5e-10),
    #             spotSize=protocol.get("spot_size",5.0e-8),
    #             spacing=1.0,
    #             rate=3e-10, # Value for platinum
    #             dwellTime=protocol.get("dwell_time",1.0e-6),
    #             preset=None,
    #             parallel=False,
    #             material='Default Material',
    #             gisPrecursor=None,

    #         )

    #     else:


    #         layerSettings = self.connection.DrawBeam.LayerSettings.EDeposition(
    #             syncWriteField=True,
    #             writeFieldSize=protocol.get("hfw",0.0005),
    #             beamCurrent=protocol.get("beam_current",5e-10),
    #             spotSize=protocol.get("spot_size",5.0e-8),
    #             rate=3e-10, # Value for platinum
    #             spacing=1.0,
    #             dwellTime=protocol.get("dwell_time",1.0e-6),
    #             preset=None,
    #             parallel=False,
    #             material='Default Material',
    #             gisPrecursor=None,
    #         )
    #     self.gis_layer = self.connection.DrawBeam.Layer("Layer_GIS", layerSettings)


    #     logging.info(f"GIS Setup Complete, {beam_type} layer settings loaded")

    # def setup_GIS_pattern(self,protocol):

    #     hfw = protocol["hfw"]
    #     line_pattern_length = protocol["length"]


    #     start_x=-line_pattern_length/2 
    #     start_y=+line_pattern_length
    #     end_x=+line_pattern_length/2
    #     end_y=+line_pattern_length
    #     depth=2e-6

    #     self.gis_layer.addLine(
    #         BeginX=start_x,
    #         BeginY=start_y,
    #         EndX=end_x,
    #         EndY=end_y,
    #         Depth=3e-06,

    #     )

    #     self.connection.DrawBeam.LoadLayer(self.gis_layer)
    #     logging.info(f"GIS Pattern Setup Complete")

    # def run_GIS(self,protocol) -> None:


    #     gas_line = self.lines[protocol['gas']]

    #     try:

    #         self.connection.GIS.OpenValve(gas_line)

    #     except Exception as e:
    #         if e.args[0] == 'Error.OutgasRequired':
    #             logging.info("Outgassing required.")
    #             logging.info(f"Outgassing {protocol['gas']} Line")
    #             self.connection.GIS.Outgas(gas_line)
    #             self.connection.GIS.OpenValve(gas_line)

    #     valve_open = self.connection.GIS.GetValveStatus(gas_line)

    #     try:
    #         # Run predefined deposition process
    #         self.connection.DrawBeam.Start()
    #         self.connection.Progress.Show("DrawBeam", "Layer 1 in progress", False, False, 0, 100)
    #         logging.info("Sputtering started.")
    #         while True:
    #             status = self.connection.DrawBeam.GetStatus()
    #             running = status[0] == self.connection.DrawBeam.Status.ProjectLoadedExpositionInProgress
    #             if running:
    #                 progress = 0
    #                 if status[1] > 0:
    #                     progress = min(100, status[2] / status[1] * 100)
    #                 printProgressBar(progress, 100)
    #                 self.connection.Progress.SetPercents(progress)
    #                 time.sleep(1)
    #             else:
    #                 if status[0] == self.connection.DrawBeam.Status.ProjectLoadedExpositionIdle:
    #                     printProgressBar(100, 100, suffix='Finished')
    #                     print('')
    #                 break
    #     finally:
    #         # Close GIS Valve in both - success and failure
    #         if valve_open:
    #             self.connection.GIS.CloseValve(gas_line)

    #     self.connection.GIS.MoveTo(gas_line, Automation.GIS.Position.Home)
    #     # self.connection.GIS.PrepareTemperature(gas_line, False)
    #     self.connection.DrawBeam.UnloadLayer()
    #     logging.info("process completed.")


    # def GIS_available_lines(self) -> List[str]:
    #     """
    #     Returns a list of available GIS lines.
    #     Args:
    #         None
    #     Returns:
    #         A dictionary of available GIS lines.
    #     """
    #     _check_sputter(self.system)
    #     GIS_lines = self.connection.GIS.Enum()
    #     self.lines = {}
    #     line_names = []
    #     for line in GIS_lines:
    #         self.lines[line.name] = line
    #         line_names.append(line.name)

    #     return line_names

    # def GIS_position(self,line_name:str) -> str:
    #     _check_sputter(self.system)

    #     line = self.lines[line_name]

    #     position = self.connection.GIS.GetPosition(line)

    #     return position.name

    # def GIS_available_positions(self) -> List[str]:

    #     _check_sputter(self.system)
    #     self.GIS_positions = self.connection.GIS.Position

    #     return self.GIS_positions.__members__.keys()

    # def GIS_move_to(self,line_name,position) -> None:

    #     _check_sputter(self.system)

    #     line = self.lines[line_name]

    #     self.connection.GIS.MoveTo(line,self.GIS_positions[position])

    # def GIS_heat_up(self,line_name):

    #     _check_sputter(self.system)

    #     line = self.lines[line_name]

    #     self.connection.GIS.PrepareTemperature(line,True)

    #     self.connection.GIS.WaitForTemperatureReady(line)

    #     time.sleep(5)

    # def GIS_temp_ready(self,line_name):

    #     _check_sputter(self.system)

    #     line = self.lines[line_name]

    #     return self.connection.GIS.GetTemperatureReady(line)

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

        logging.info("restoring microscope state...")

        # restore electron beam
        _check_beam(BeamType.ELECTRON, self.system)
        logging.info("restoring electron beam settings...")
        self.connection.SEM.Optics.SetWD(
            microscope_state.electron_beam.working_distance
            * constants.METRE_TO_MILLIMETRE
        )

        self.connection.SEM.Beam.SetCurrent(
            microscope_state.electron_beam.beam_current * constants.SI_TO_PICO
        )

        self.connection.SEM.Optics.SetViewfield(
            microscope_state.electron_beam.hfw * constants.METRE_TO_MILLIMETRE
        )

        if microscope_state.electron_beam.shift is not None:
            print(microscope_state.electron_beam.shift.x, microscope_state.electron_beam.shift.y)
            self.connection.SEM.Optics.SetImageShift(microscope_state.electron_beam.shift.x, microscope_state.electron_beam.shift.y)
            time.sleep(1)
        if microscope_state.electron_beam.scan_rotation is not None:
            self.connection.SEM.Optics.SetImageRotation(microscope_state.electron_beam.scan_rotation)
        # microscope.beams.electron_beam.stigmator.value = (
        #     microscope_state.electron_beam.stigmation
        # )
        self.set_detector_settings(microscope_state.electron_detector, BeamType.ELECTRON)
        # restore ion beam
        _check_beam(BeamType.ION, self.system)
        logging.info(f"restoring ion beam settings...")

        self.connection.FIB.Optics.SetViewfield(
            microscope_state.ion_beam.hfw * constants.METRE_TO_MILLIMETRE
        )
        if microscope_state.ion_beam.shift is not None:
            self.connection.FIB.Optics.SetImageShift(microscope_state.electron_beam.shift.x, microscope_state.electron_beam.shift.y)
            time.sleep(1)
        if microscope_state.electron_beam.scan_rotation is not None:
            self.connection.FIB.Optics.SetImageRotation(microscope_state.electron_beam.scan_rotation)
        # microscope.beams.ion_beam.stigmator.value = microscope_state.ion_beam.stigmation
        self.set_detector_settings(microscope_state.ion_detector, BeamType.ION)

        self.move_stage_absolute(microscope_state.stage_position)
        logging.info("microscope state restored")
        return

    def get_available_values(self, key: str, beam_type: BeamType = None)-> list:
        """Get a list of available values for a given key.
        Keys: plasma_gas, current, detector_type
        """
        values = []
        if beam_type is BeamType.ION:
            if key == "plasma_gas":
                values = self.connection.GIS.Enum()

        if key == "current":
            if beam_type == BeamType.ELECTRON:
                values = [1.0e-12]
            if beam_type == BeamType.ION:
                values = [20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.6e-9, 28.0e-9, 120e-9]

        if key == "detector_type" and beam_type == BeamType.ELECTRON:
            values = self.connection.SEM.Detector.Enum()
            for i in range(len(values)):
                values[i-1] = values[i-1].name 
        if key == "detector_type" and beam_type == BeamType.ION:
            values = self.connection.FIB.Detector.Enum()
            for i in range(len(values)):
                values[i-1] = values[i-1].name
        
        if key == "detector_mode": 
            values = None 

        if key == "presets":
            return self._get_presets()

        if key == "scan_direction":
            values = ["Flyback", "RLE", "SpiralInsideOut", "SpiralOutsideIn", "ZigZag"]
            

        return values

   
    def _get(self, key: str, beam_type: BeamType = None) -> Union[float, str, None]:
        """Get a property of the microscope."""
        if beam_type is not None:
            beam = self.connection.SEM if beam_type == BeamType.ELECTRON else self.connection.FIB
            _check_beam(beam_type, self.system)
        
        # beam properties 
        if key == "on": 
            return beam.Beam.GetStatus()
        if key == "working_distance" and beam_type == BeamType.ELECTRON:
            return beam.Optics.GetWD() * constants.MILLIMETRE_TO_METRE
        if key == "current":
            if beam_type == BeamType.ELECTRON:
                return beam.Beam.GetCurrent() * constants.PICO_TO_SI
            else:
                return beam.Beam.ReadProbeCurrent() * constants.PICO_TO_SI
        if key == "voltage":
            return beam.Beam.GetVoltage() 
        if key == "hfw":
            return beam.Optics.GetViewfield() * constants.MILLIMETRE_TO_METRE
        if key == "resolution":
            if beam_type == BeamType.ELECTRON and self.last_image_eb is not None:
                return self.last_image_eb.metadata.image_settings.resolution
            elif beam_type == BeamType.ION and self.last_image_ib is not None:
                return self.last_image_ib.metadata.image_settings.resolution
        if key == "dwell_time":
            if beam_type == BeamType.ELECTRON and self.last_image_eb is not None:
                return self.last_image_eb.metadata.image_settings.dwell_time
            elif beam_type == BeamType.ION and self.last_image_ib is not None:
                return self.last_image_ib.metadata.image_settings.dwell_time   
        if key =="scan_rotation":
            return beam.Optics.GetImageRotation()   
        if key == "shift":
            values = beam.Optics.GetImageShift()
            shift = Point(values[0]*constants.MILLIMETRE_TO_METRE, values[1]*constants.MILLIMETRE_TO_METRE)
            return shift
        


        # system properties
        if key == "eucentric_height":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.eucentric_height
            elif beam_type is BeamType.ION:
                return self.system.ion.eucentric_height
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        if key == "column_tilt":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.column_tilt
            elif beam_type is BeamType.ION:
                return self.system.ion.column_tilt
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                return self.system.ion.plasma
            else:
                return False

        # stage properties
        if key == "stage_position":
            _check_stage(self.system)
            return self.get_stage_position()
        if key == "stage_calibrated":
            _check_stage(self.system)
            return self.connection.Stage.IsCalibrated()
        
        # chamber properties
        if key == "chamber_state":
            return self.connection.Chamber.GetStatus()
        if key == "chamber_pressure":
            return self.connection.Chamber.GetPressure(0)
        
        # detector properties
        if key == "detector_type":
            detector = beam.Detector.Get(Channel = 0) 
            if detector is not None:
                return detector.name
            else: 
                return None
        if key == "detector_contrast":
            if beam_type == BeamType.ELECTRON:
                contrast, brightness = beam.Detector.GetGainBlack(Detector= self.electron_detector_active)
            elif beam_type == BeamType.ION:
                contrast, brightness = beam.Detector.GetGainBlack(Detector= self.ion_detector_active)
            return contrast/100
        if key == "detector_brightness":
            if beam_type == BeamType.ELECTRON:
                contrast, brightness = beam.Detector.GetGainBlack(Detector= self.electron_detector_active)
            elif beam_type == BeamType.ION:
                contrast, brightness = beam.Detector.GetGainBlack(Detector= self.ion_detector_active)
            return brightness/100
        
        # manipulator properties
        if key == "manipulator_position":
            _check_manipulator(self.system)
            return self.connection.Nanomanipulator.GetPosition(0)
        if key == "manipulator_calibrated":
            _check_manipulator(self.system)
            return self.connection.Nanomanipulator.IsCalibrated(0)
        if key == "manipulator_state":
            _check_manipulator(self.system)
            return self.connection.Nanomanipulator.GetStatus(0)

        if key == "presets":
            return self._get_presets()


        # manufacturer properties
        if key == "manufacturer":
            return self.system.info.manufacturer
        if key == "model":
            return self.system.info.model
        if key == "software_version":
            return self.system.info.software_version
        if key == "serial_number":
            return self.system.info.serial_number
        if key == "hardware_version":
            return self.system.info.hardware_version
        
        if key == "column_tilt":
            # TODO: check if this is available
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.column_tilt
            elif beam_type is BeamType.ION:
                return self.system.ion.column_tilt
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")
        
        # logging.warning(f"Unknown key: {key} ({beam_type})")
        return None   

    def _set(self, key: str, value, beam_type: BeamType = None) -> None:
        """Set a property of the microscope."""
        if beam_type is not None:
            beam = self.connection.SEM if beam_type == BeamType.ELECTRON else self.connection.FIB
            _check_beam(beam_type, self.system)

        if key == "working_distance":
            if beam_type == BeamType.ELECTRON:
                beam.Optics.SetWD(value * constants.METRE_TO_MILLIMETRE)
                logging.info(f"Electron beam working distance set to {value} m.")
            else: 
                logging.info("Setting working distance for ion beam is not supported by Tescan API.")
            return
        if key == "current":
            if beam_type == BeamType.ELECTRON:
                beam.Beam.SetCurrent(value * constants.SI_TO_PICO)
                logging.info(f"Electron beam current set to {value} A.")
            else: 
                logging.warning("Setting current for ion beam is not supported by Tescan API, please use the native microscope interface.")
            return
        if key == "voltage":
            if beam_type == BeamType.ELECTRON:
                beam.Beam.SetVoltage(value)
                logging.info(f"Electron beam voltage set to {value} V.")
            else:
                logging.warning("Setting voltage for ion beam is not supported by Tescan API, please use the native microscope interface.")
            return
        if key == "hfw":
            beam.Optics.SetViewfield(value * constants.METRE_TO_MILLIMETRE)
            logging.info(f"{beam_type.name} HFW set to {value} m.")
            return
        if key == "scan_rotation":
            beam.Optics.SetImageRotation(value)
            logging.info(f"{beam_type.name} scan rotation set to {value} degrees.")
            return

        # beam control
        if key == "on":
            beam.Beam.On() if value else beam.Beam.Off()
            logging.info(f"{beam_type.name} beam turned {'on' if value else 'off'}.")
            return
        if key == "shift":
            point = Point(value.x*constants.METRE_TO_MILLIMETRE, value.y*constants.METRE_TO_MILLIMETRE)
            beam.Optics.SetImageShift(point.x, point.y)
            logging.info(f"{beam_type.name} beam shift set to {value}.")
            return
        
        # detector control
        if key == "detector_type":
            if beam_type == BeamType.ELECTRON:
                self.electron_detector_active = value
                beam.Detector.Set(Channel = 0, Detector = value)
                self.electron_detector_active = beam.Detector.Get(Channel = 0)
                logging.info(f"{beam_type} detector type set to {value}.")
                return
            elif beam_type == BeamType.ION:
                self.ion_detector_active = value
                beam.Detector.Set(Channel = 0, Detector = value)
                self.ion_detector_active = beam.Detector.Get(Channel = 0)
                logging.info(f"{beam_type} detector type set to {value}.")
                return
        if key in ["detector_brightness", "detector_contrast"]:
            _check_beam(beam_type, self.system)
            if key == "detector_brightness":
                if 0 <= value <= 1:
                    if beam_type == BeamType.ELECTRON:
                        og_contrast, og_brightness = beam.Detector.GetGainBlack(Detector= self.electron_detector_active)
                        beam.Detector.SetGainBlack(Detector= self.electron_detector_active, Gain = og_contrast, Black = value*100)
                        logging.info(f"{beam_type} detector brightness set to {value}.")
                    elif beam_type == BeamType.ION:
                        og_contrast, og_brightness = beam.Detector.GetGainBlack(Detector= self.ion_detector_active)
                        beam.Detector.SetGainBlack(Detector= self.ion_detector_active, Gain = og_contrast, Black = value*100)
                        logging.info(f"{beam_type} detector brightness set to {value}.")
                else:
                    logging.warning(f"Invalid brightness value: {value}, must be between 0 and 1.")
                return 
            if key == "detector_contrast":
                if 0 <= value <= 1:
                    if beam_type == BeamType.ELECTRON:
                        og_contrast, og_brightness = beam.Detector.GetGainBlack(Detector= self.electron_detector_active)
                        beam.Detector.SetGainBlack(Detector= self.electron_detector_active, Gain = value*100, Black = og_brightness)
                        logging.info(f"{beam_type} detector contrast set to {value}.")
                    elif beam_type == BeamType.ION:
                        og_contrast, og_brightness = beam.Detector.GetGainBlack(Detector= self.ion_detector_active)
                        beam.Detector.SetGainBlack(Detector= self.ion_detector_active, Gain = value*100, Black = og_brightness)
                        logging.info(f"{beam_type} detector contrast set to {value}.")
                else:
                    logging.warning(f"Invalid contrast value: {value}, must be between 0 and 1.")
                return 

        if key == "preset":
            beam.Preset.Activate(value)
            logging.info(f"Preset {value} activated for {beam_type}.")
            return
                    

        logging.warning(f"Unknown key: {key}, value: {value} ({beam_type})")
        return

    def check_available_values(self, key: str, beam_type: BeamType = None) -> bool:
        return False
    
    def home(self) -> None:
        logging.warning("No homing available, please use native UI.")
        return