import datetime
import logging
import os
import sys
import threading
import time
from copy import deepcopy
from queue import Queue
from typing import Dict, List, Union, Tuple

import numpy as np

import fibsem.constants as constants
from fibsem.microscope import (
    FibsemMicroscope,
    _check_beam,
    _check_manipulator,
    _check_sputter,
    _check_stage,
)

TESCAN_API_AVAILABLE = False

try:
    import tescanautomation
    from tescanautomation import Automation
    from tescanautomation.Common import Document, Bpp, Detector
    from tescanautomation.DrawBeam import IEtching, Status as DBStatus
    from tescanautomation.SEM import HVBeamStatus as SEMStatus

    sys.modules.pop("tescanautomation.GUI")
    sys.modules.pop("tescanautomation.pyside6gui")
    sys.modules.pop("tescanautomation.pyside6gui.imageViewer_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_utils")
    sys.modules.pop("tescanautomation.pyside6gui.rc_GUI")
    sys.modules.pop("tescanautomation.pyside6gui.workflow_private")
    sys.modules.pop("PySide6.QtCore")
    TESCAN_API_AVAILABLE = True
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
    PatterningState,
)

def _get_beam_settings_from_tescan_md(md: dict, beam_type: BeamType) -> BeamSettings:
    """Parse metadata from Tescan image header to get beam settings."""
    return BeamSettings(
        beam_type=beam_type,
        working_distance=float(md["WD"]),
        voltage=float(md["HV"]),
        beam_current=float(md["PredictedBeamCurrent"]),
        dwell_time=float(md["DwellTime"]),
        scan_rotation=float(md["ScanRotation"]) * constants.DEGREES_TO_RADIANS,
        stigmation=Point(x=float(md["StigmatorX"]), 
                            y=float(md["StigmatorY"])),
        shift=Point(x=float(md["ImageShiftX"]), 
                    y=float(md["ImageShiftY"])),
    )
    
def _get_detector_settings_from_tescan_md(md: dict) -> FibsemDetectorSettings:
    """Parse metadata from Tescan image header to get detector settings."""
    return FibsemDetectorSettings(
        type=md["Detector0"],
        brightness=float(md["Detector0Gain"]) / 100.0,
        contrast=float(md["Detector0Offset"]) / 100.0,
    )


def _get_pixel_size_from_tescan_md(md: dict) -> Point:
    """Parse metadata from Tescan image header to get pixel size."""
    pixelsize = Point(float(md["MAIN"]["PixelSizeX"]), 
                    float(md["MAIN"]["PixelSizeY"]))
    return pixelsize

def _get_microscope_state_from_tescan_md(md: dict, image_shape: Tuple[int, int]) -> MicroscopeState:
    """Parse metadata from Tescan image header to create a MicroscopeState object."""
    ddict = {}
    SUPPORTED_KEYS = ["MAIN", "FIB", "SEM"]

    for k in md:
        if k in SUPPORTED_KEYS:
            ddict[k] = dict(md[k])

    if "FIB" in ddict:
        beam_type = BeamType.ION
        k = "FIB"
    if "SEM" in ddict:
        beam_type = BeamType.ELECTRON
        k = "SEM"


    # stage position
    stage_position = FibsemStagePosition(
        x=float(ddict[k]["StageX"]),
        y=float(ddict[k]["StageY"]),
        z=float(ddict[k]["StageZ"]),
        r=float(ddict[k]["StageRotation"]) * constants.DEGREES_TO_RADIANS,
        t=float(ddict[k]["StageTilt"]) * constants.DEGREES_TO_RADIANS,
        coordinate_system="RAW",
    )

    # fov must be calc manually from pixelsize * resolution
    pixelsize = _get_pixel_size_from_tescan_md(ddict)
    resolution = image_shape[1], image_shape[0]
    hfw = pixelsize.x * resolution[0]
   
    # default values
    electron_beam = BeamSettings(beam_type=BeamType.ELECTRON)
    ion_beam = BeamSettings(beam_type=BeamType.ION)
    electron_detector = FibsemDetectorSettings()
    ion_detector = FibsemDetectorSettings()

    # beam settings 
    if beam_type is BeamType.ION:
        ion_detector = _get_detector_settings_from_tescan_md(ddict[k])
        ion_beam = _get_beam_settings_from_tescan_md(ddict[k], BeamType.ION)
        ion_beam.hfw = hfw
        ion_beam.resolution = resolution
    if beam_type is BeamType.ELECTRON:
        electron_detector = _get_detector_settings_from_tescan_md(ddict[k])
        electron_beam = _get_beam_settings_from_tescan_md(ddict[k], BeamType.ELECTRON)
        electron_beam.hfw = hfw
        electron_beam.resolution = resolution

    # acquisition timestamp
    acquisition_time = ddict["MAIN"]["Date"] + " " + ddict["MAIN"]["Time"]
    timestamp = datetime.datetime.strptime(acquisition_time, "%Y-%m-%d %H:%M:%S").timestamp()

    ms = MicroscopeState(
        timestamp=timestamp,
        stage_position=stage_position,
        electron_beam=electron_beam,
        ion_beam=ion_beam,
        electron_detector=electron_detector,
        ion_detector=ion_detector,
    )
    return ms

def fromTescanImage(image: Document, image_settings: ImageSettings = None) -> FibsemImage:
    """Create a FibsemImage object from a Tescan image document."""
    image_data = np.array(image.Image)
    pixelsize = _get_pixel_size_from_tescan_md(image.Header)
    ms = _get_microscope_state_from_tescan_md(image.Header, 
                                              image_shape=image_data.shape)
    if image_settings is None:
        image_settings = ImageSettings()

    md = FibsemImageMetadata(
        image_settings=image_settings,
        microscope_state=ms,
        pixel_size=pixelsize,
    )

    return FibsemImage(data=image_data, metadata=md)

FibsemImage.fromTescanImage = fromTescanImage

DrawBeamStatusToPatterningState = {
    DBStatus.ProjectNotLoaded: PatterningState.IDLE,
    DBStatus.ProjectLoadedExpositionIdle: PatterningState.IDLE,
    DBStatus.ProjectLoadedExpositionInProgress: PatterningState.RUNNING,
    DBStatus.ProjectLoadedExpositionPaused: PatterningState.PAUSED,
    DBStatus.Unknown: PatterningState.ERROR,
}

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


SEM_LIMITS: Dict[str, Tuple] = {
    "hfw":  (1.0e-6, 2580.0e-6),
}
FIB_LIMITS: Dict[str, Tuple] = {
    "hfw":  (1.0e-6, 450.0e-6),
}

LIMITS = {
    BeamType.ELECTRON: SEM_LIMITS,
    BeamType.ION: FIB_LIMITS,
}


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
        if not TESCAN_API_AVAILABLE:
            raise ImportError("The TESCAN Automation API is not available. Please see the user guide for installation instructions.")
        
        if ip_address is None:
            ip_address = system_settings.info.ip_address
        
        # create microscope client
        self.connection: Automation 

        # initialise system settings
        self.system: SystemSettings = system_settings

        self.milling_channel: BeamType = BeamType.ION
        
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

        self._default_detector_names = {BeamType.ELECTRON: "E-T", BeamType.ION: "SE"}
        self._active_detector: Dict[BeamType, Detector] = {}    

        # TODO: use what the user specified in the configuration file
        # set up detectors
        self.set("detector_type", self._default_detector_names[BeamType.ELECTRON], BeamType.ELECTRON)
        self.set("detector_type", self._default_detector_names[BeamType.ION], BeamType.ION) 
        image = self.connection.SEM.Scan.AcquireImageFromChannel(0, 1, 1, 100) # TODO: find a better way to get system info

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

        logging.info(f"acquiring new {image_settings.beam_type.name} image.")

        image = self.acquire_image_v2(image_settings)

        if image_settings.beam_type == BeamType.ELECTRON:
            self.last_image_eb = image
        if image_settings.beam_type == BeamType.ION:
            self.last_image_ib = image

        image.metadata.user = self.user
        image.metadata.experiment = self.experiment 
        image.metadata.system = self.system

        return image

    def acquire_image_v2(self, image_settings: ImageSettings):
        """
        Acquires an image with the given settings and returns a FibsemImage object.
        """

        # prepare the beam (turn on, stop scanning)
        self._prepare_beam(image_settings.beam_type)
        beam = self._get_beam(image_settings.beam_type)

        # set the active detector (maybe redundant?)
        beam.Detector.Set(
            Channel=0, 
            Detector=self._active_detector[image_settings.beam_type], 
            Bpp=Bpp.Grayscale_8_bit
        )

        dwell_time = image_settings.dwell_time * constants.SI_TO_NANO

        # resolution
        image_width, image_height = image_settings.resolution

        self.set("hfw", image_settings.hfw, image_settings.beam_type)       
        
        if image_settings.reduced_area is not None:
            left =  int(image_settings.reduced_area.left * image_width)
            top = int(image_settings.reduced_area.top * image_height) 
            width = int(image_settings.reduced_area.width * image_width)
            height = int(image_settings.reduced_area.height * image_height)
            image = beam.Scan.AcquireROIFromChannel(
                Channel= 0,
                Width= image_width,
                Height= image_height,
                Left= left,
                Top= top,
                Right= left + width -1 ,
                Bottom= top + height - 1,
                DwellTime= dwell_time
            )
        else:
            image = beam.Scan.AcquireImageFromChannel(
                Channel=0, 
                Width=image_width, 
                Height=image_height, 
                DwellTime=dwell_time
            )

        fibsem_image = FibsemImage.fromTescanImage(image, image_settings)
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
        self._get_beam(beam_type=beam_type).Detector.AutoSignal(Detector=self._active_detector[beam_type])
    
    def auto_focus(self, beam_type: BeamType) -> None:
        _check_beam(beam_type, self.system)
        if beam_type == BeamType.ELECTRON:
            logging.info("Running autofocus on electron beam.")
            self.connection.SEM.AutoWDFine(self._active_detector[beam_type])
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
        # TODO: migrate to super function
        electron_beam = BeamSettings(BeamType.ELECTRON)
        if self.system.electron.enabled is True:
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
        
        ion_beam = BeamSettings(BeamType.ION)
        if self.system.ion.enabled is True:
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
        tilt = self.get("column_tilt", beam_type)
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

        return False
        # manipulator_positions = cfg.load_tescan_manipulator_calibration()

        # if not manipulator_positions["calibrated"]:
        #     logging.warning("Manipulator positions not calibrated, cannot get state")
        #     return False

        # retracted_position_x = manipulator_positions["parking"]["x"]*constants.METRE_TO_MILLIMETRE
        # retracted_position_y = manipulator_positions["parking"]["y"]*constants.METRE_TO_MILLIMETRE
        # retracted_position_z = manipulator_positions["parking"]["z"]*constants.METRE_TO_MILLIMETRE

        # current_position = self.get_manipulator_position()

        # current_position_array = [current_position.x*constants.METRE_TO_MILLIMETRE, current_position.y*constants.METRE_TO_MILLIMETRE, current_position.z*constants.METRE_TO_MILLIMETRE]

        # check_compare = np.isclose(current_position_array, [retracted_position_x, retracted_position_y, retracted_position_z], atol=0.1)

        # return True if False in check_compare else False
            

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
            mill_settings (FibsemMillingSettings): Milling settings.

        Note:
            This method sets up the microscope imaging and patterning for milling using the ion beam.
            It sets the active view and device to the ion beam, the default beam type to the ion beam,
            the specified application file, and the specified patterning mode.
            It also clears any existing patterns and sets the horizontal field width to the desired value.
            The method does not start the milling process.
        """
        if mill_settings.milling_channel is not BeamType.ION:
            raise ValueError("Only FIB milling is currently supported.")
        
        _check_beam(mill_settings.milling_channel, self.system)
        self._prepare_beam(mill_settings.milling_channel)

        # spot_size =   # application_file
        # rate = ## in application file called Volume per Dose (m3/C)
        # dwell_time =   # in seconds ## in application file
        # parallel_mode = 
        self.milling_channel = mill_settings.milling_channel

        self.set("preset", mill_settings.preset, BeamType.ION)  # QUERY: do we need to set this here as it is also set in IEtching?

        layer_settings = IEtching(
            syncWriteField=False,
            writeFieldSize=mill_settings.hfw,
            beamCurrent=self.get("current", self.milling_channel),
            spotSize=mill_settings.spot_size,
            rate=mill_settings.rate,
            dwellTime=mill_settings.dwell_time,
            parallel=bool(mill_settings.patterning_mode == "Parallel"),
            preset=mill_settings.preset,
            spacing=mill_settings.spacing,
        )

        # TODO: change the layer name to milling stage name
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
        self._prepare_beam(self.milling_channel)

        self.connection.DrawBeam.LoadLayer(self.layer)
        logging.info("running ion beam milling now...")
        self.connection.DrawBeam.Start()

        if asynch:
            return # up to the user to monitor the milling process/progress

        self.connection.Progress.Show(
            Title="DrawBeam Milling (OpenFIBSEM)", 
            Text="Layer 1 in progress", 
            HideButton=True, 
            Marquee=False, 
            ProgressMin=0, ProgressMax=100
        )

        err = None
        try:
            while True:
                status = self.connection.DrawBeam.GetStatus() # status, total, elapsed
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
        except Exception as err:
            logging.error(f"Error in run_milling: {err}")
            pass
        finally:
            print()  # new line on complete
            self.connection.Progress.Hide()
            if err:
                self.connection.DrawBeam.Stop()
                self.connection.DrawBeam.UnloadLayer()

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

    def finish_milling(self, imaging_current: float = None, imaging_voltage: float = None):
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        # """
        try:
            default_preset = "30 keV; 150 pA"
            self.connection.FIB.Preset.Activate(default_preset) # TODO: restore the default preset?
            self.connection.DrawBeam.UnloadLayer()
            logging.debug(f"Finished milling, restored preset to {default_preset}")
        except Exception as e:
            logging.debug(f"Error in finish_milling: {e}")
            pass

    def stop_milling(self):

        # TODO: improve thread safety to stop from another thread
        try:
            thread_connection = Automation(self.system.info.ip_address, port=8300)
            if thread_connection.DrawBeam.GetStatus()[0] == DBStatus.ProjectLoadedExpositionInProgress:
                logging.info("Milling is in progress, stopping now...")
                thread_connection.DrawBeam.Stop()
        except Exception as e:
            logging.error(f"Error in stop_milling: {e}")
        finally:
            del thread_connection

    def pause_milling(self):
        self.connection.DrawBeam.Pause()

    def resume_milling(self):
        self.connection.DrawBeam.Resume()

    def get_milling_state(self):
        state = self.connection.DrawBeam.GetStatus()[0]
        return DrawBeamStatusToPatterningState[state]

    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings):
        pass

    def estimate_milling_time(self, patterns: List[str] = None):
        
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
            and scanned in the direction specified in the scan_direction attribute of pattern_settings.

            The created pattern can be added to the patterning queue and executed using the layer methods in Automation.
        """
        # centre_x = pattern_settings.centre_x
        # centre_y = pattern_settings.centre_y
        # depth = pattern_settings.depth
        # width = pattern_settings.width
        # height = pattern_settings.height
        # rotation = pattern_settings.rotation * constants.RADIANS_TO_DEGREES # CHECK UNITS (TESCAN Takes Degrees)
        # passes = pattern_settings.passes if pattern_settings.passes is not None else 1.0 # not the same concept, remove for now
        scan_directions = self.get_available_values(key="scan_direction")
        if pattern_settings.scan_direction in scan_directions:
            scanning_path = pattern_settings.scan_direction
        else:
            scanning_path = "Flyback"
            logging.info(f"Scan direction {pattern_settings.scan_direction} not supported. Using Flyback instead.")
            logging.info(f"Supported scan directions are: {scan_directions}")
        self.connection.DrawBeam.ScanningPath = scanning_path

        # TODO: replace with cross_section parameter
        if pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            add_pattern_fn = self.layer.addRectanglePolish
        else:
            add_pattern_fn = self.layer.addRectangleFilled

        add_pattern_fn(
                CenterX=pattern_settings.centre_x,
                CenterY= pattern_settings.centre_y,
                Depth=pattern_settings.depth,
                DepthUnit='m',
                Width=pattern_settings.width,
                Height=pattern_settings.height,
                Angle=pattern_settings.rotation * constants.RADIANS_TO_DEGREES,
                ScanningPath=scanning_path,
                # ExpositionFactor=passes
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
        if microscope_state.electron_beam.working_distance is not None:
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

    def _get_beam(self, beam_type: BeamType) -> Union[Automation.SEM, Automation.FIB]:
        """Get the beam object for the given beam type."""
        if not isinstance(beam_type, BeamType):
            raise ValueError(f"Invalid beam type: {beam_type}")

        if beam_type is BeamType.ELECTRON:
            return self.connection.SEM
        if beam_type is BeamType.ION:
            return self.connection.FIB

    def _prepare_beam(self, beam_type: BeamType):
        """Prepare the beam for imaging, milling, or other operations."""
        beam = self._get_beam(beam_type)
        
        # check the beam is on
        status = beam.Beam.GetStatus()
        if status != beam.Beam.Status.BeamOn:
            beam.Beam.On()

        # stop the scanning before we start scanning or before automatic procedures,
        beam.Scan.Stop()

    def _get_presets(self, beam_type: BeamType) -> List[str]:
        presets = self._get_beam(beam_type=beam_type).Preset.Enum()	
        return sorted(presets)

    def _get_available_detectors(self, beam_type: BeamType) -> List[str]:
        """Get a list of available detectors for the given beam type."""
        detectors = []
        if beam_type == BeamType.ELECTRON:
            detectors = self.connection.SEM.Detector.Enum()
        elif beam_type == BeamType.ION:
            detectors = self.connection.FIB.Detector.Enum()
        return detectors

    def _get_detector(self, detector_type: Union[Detector, str], beam_type: BeamType) -> str:
        """Get the detector object for the given detector type and beam type."""
        if isinstance(detector_type, Detector):
            detector_type = detector_type.name
        
        available_detectors = self._get_available_detectors(beam_type)
        detector: Detector
        for detector in available_detectors:
            if detector_type == detector.name:
                logging.debug(f"Found detector {detector.name}, index {detector.index}")
                return detector
        return None

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

        if key == "detector_type":
            detectors = self._get_available_detectors(beam_type=beam_type)
            values = [detector.name for detector in detectors]
        
        if key == "detector_mode": 
            values = None  # TODO: this should be empty list?

        if key == "presets":
            return self._get_presets(beam_type=beam_type)

        if key == "scan_direction":
            values = ["Flyback", "RLE", "SpiralInsideOut", "SpiralOutsideIn", "ZigZag"]
            

        return values

   
    def _get(self, key: str, beam_type: BeamType = None) -> Union[float, str, None]:
        """Get a property of the microscope."""
        if beam_type is not None:
            beam = self._get_beam(beam_type)
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
        if key == "stigmation":
            if beam_type == BeamType.ELECTRON and self.last_image_eb is not None:
                return self.last_image_eb.metadata.microscope_state.electron_beam.stigmation
            elif beam_type == BeamType.ION and self.last_image_ib is not None:
                return self.last_image_ib.metadata.microscope_state.ion_beam.stigmation         
        if key =="scan_rotation":
            return beam.Optics.GetImageRotation()
        if key == "shift":
            values = beam.Optics.GetImageShift()
            shift = Point(values[0]*constants.MILLIMETRE_TO_METRE, 
                          values[1]*constants.MILLIMETRE_TO_METRE)
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
            if detector is None: # QUERY: can this be None?
                return None
            return detector.name
         
        if key in ["detector_contrast", "detector_brightness"]:
            contrast, brightness = beam.Detector.GetGainBlack(
                Detector=self._active_detector[beam_type]
            )

            if key == "detector_contrast":
                return contrast / 100
            if key == "detector_brightness":
                return brightness / 100
        
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
            return self._get_presets(beam_type=beam_type)

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
        
        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None   

    def _set(self, key: str, value, beam_type: BeamType = None) -> None:
        """Set a property of the microscope."""
        if beam_type is not None:
            beam = self._get_beam(beam_type)
            _check_beam(beam_type, self.system)
            self._prepare_beam(beam_type)

        if key == "working_distance":
            if beam_type is BeamType.ION:
                logging.warning(f"Setting working distance directly for {beam_type} is not supported by Tescan API")
                return
            if beam_type is BeamType.ELECTRON:
                beam.Optics.SetWD(value * constants.METRE_TO_MILLIMETRE)
                logging.info(f"Electron beam working distance set to {value} m.")
            return
        if key == "current":
            if beam_type is BeamType.ION:
                logging.warning(f"Setting current directly for {beam_type} is not supported by Tescan API, please use presets instead.") 
                return
            if beam_type is BeamType.ELECTRON:
                beam.Beam.SetCurrent(value * constants.SI_TO_PICO)
                logging.info(f"Electron beam current set to {value} A.")
            return
        if key == "voltage":
            if beam_type is BeamType.ION:
                logging.warning(f"Setting voltage directly for {beam_type} is not supported by Tescan API, please use presets instead.") 
                return
            if beam_type is BeamType.ELECTRON:
                beam.Beam.SetVoltage(value)
                logging.info(f"Electron beam voltage set to {value} V.")
            return

        if key == "hfw":
            limits = LIMITS[beam_type]["hfw"]
            value = np.clip(value, limits[0], limits[1])
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
            detector = self._get_detector(value, beam_type)
            if detector is None:
                logging.warning(f"Detector {value} not found for {beam_type}.")
                return
            beam.Detector.Set(Channel = 0, Detector = detector)
            self._active_detector[beam_type] = detector
            logging.debug(f"{beam_type.name} detector type set to {value}.")
            return

        if key == "detector_mode":
            logging.debug("Setting detector mode not supported by Tescan API.")
            return

        if key in ["detector_brightness", "detector_contrast"]:
            
            # check if value is between 0 and 1
            if not (0 <= value <= 1):
                logging.warning(f"Invalid value for {beam_type} {key}: {value}. Must be between 0 and 1.")
                return
            
            # get active detector 
            active_detector =  self._active_detector[beam_type]
            if active_detector is None:
                logging.warning(f"No active detector for {beam_type}. Please set detector type first.")
                return
            
            # get current gain and black level
            contrast, brightness = beam.Detector.GetGainBlack(Detector= active_detector)
            if key == "detector_contrast":
                contrast = value * 100
            if key == "detector_brightness":
                brightness = value * 100

            # set new gain and black level
            beam.Detector.SetGainBlack(Detector= active_detector, 
                                        Gain = contrast, 
                                        Black = brightness)
            logging.info(f"{beam_type.name} {key} set to {value}.")
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
    


    # def fromTescanFile(
    #     cls,
    #     image_path: str,
    #     metadata_path: str,
    #     beam_type: BeamType,
    # ) -> "FibsemImage":
    #     with tff.TiffFile(image_path) as tiff_image:
    #         data = tiff_image.asarray()

    #     stage = 0
    #     dictionary = {"MAIN": {}, "SEM": {}, "FIB": {}}
    #     with open(metadata_path, "r") as file:
    #         for line in file:
    #             if line.startswith("["):
    #                 stage += 1
    #                 continue

    #             line = line.strip()
    #             if not line:
    #                 continue  # Skip empty lines

    #             key, value = line.split("=")
    #             key = key.strip()
    #             value = value.strip()
    #             if stage == 1:
    #                 dictionary["MAIN"][key] = value
    #             if stage == 2 and beam_type.name == "ELECTRON":
    #                 dictionary["SEM"][key] = value
    #             if stage == 2 and beam_type.name == "ION":
    #                 dictionary["FIB"][key] = value

    #     if beam_type.name == "ELECTRON":
    #         image_settings = ImageSettings(
    #             resolution=[data.shape[0], data.shape[1]],
    #             dwell_time=float(dictionary["SEM"]["DwellTime"]),
    #             hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
    #             beam_type=BeamType.ELECTRON,
    #             filename=Path(image_path).stem,
    #             path=Path(image_path).parent,
    #         )
    #         pixel_size = Point(
    #             float(dictionary["MAIN"]["PixelSizeX"]),
    #             float(dictionary["MAIN"]["PixelSizeY"]),
    #         )
    #         microscope_state = MicroscopeState(
    #             timestamp=datetime.strptime(
    #                 dictionary["MAIN"]["Date"] + " " + dictionary["MAIN"]["Time"],
    #                 "%Y-%m-%d %H:%M:%S",
    #             ),
    #             electron_beam=BeamSettings(
    #                 beam_type=BeamType.ELECTRON,
    #                 working_distance=float(dictionary["SEM"]["WD"]),
    #                 beam_current=float(dictionary["SEM"]["PredictedBeamCurrent"]),
    #                 voltage=float(dictionary["SEM"]["TubeVoltage"]),
    #                 hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
    #                 resolution=[data.shape[0], data.shape[1]],
    #                 dwell_time=float(dictionary["SEM"]["DwellTime"]),
    #                 shift=Point(
    #                     float(dictionary["SEM"]["ImageShiftX"]),
    #                     float(dictionary["SEM"]["ImageShiftY"]),
    #                 ),
    #                 stigmation=Point(
    #                     float(dictionary["SEM"]["StigmatorX"]),
    #                     float(dictionary["SEM"]["StigmatorY"]),
    #                 ),
    #             ),
    #             ion_beam=BeamSettings(beam_type=BeamType.ION),
    #         )
    #         detector_settings = FibsemDetectorSettings(
    #             type=dictionary["SEM"]["Detector"],
    #             brightness=float(dictionary["SEM"]["Detector0Offset"]),
    #             contrast=float(dictionary["SEM"]["Detector0Gain"]),
    #         )

    #     if beam_type.name == "ION":
    #         image_settings = ImageSettings(
    #             resolution=[data.shape[0], data.shape[1]],
    #             dwell_time=float(dictionary["FIB"]["DwellTime"]),
    #             hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
    #             beam_type=BeamType.ELECTRON,
    #             filename=Path(image_path).stem,
    #             path=Path(image_path).parent,
    #         )
    #         pixel_size = Point(
    #             float(dictionary["MAIN"]["PixelSizeX"]),
    #             float(dictionary["MAIN"]["PixelSizeY"]),
    #         )
    #         microscope_state = MicroscopeState(
    #             timestamp=datetime.strptime(
    #                 dictionary["MAIN"]["Date"] + " " + dictionary["MAIN"]["Time"],
    #                 "%Y-%m-%d %H:%M:%S",
    #             ),
    #             electron_beam=BeamSettings(beam_type=BeamType.ELECTRON),
    #             ion_beam=BeamSettings(
    #                 beam_type=BeamType.ION,
    #                 working_distance=float(dictionary["FIB"]["WD"]),
    #                 beam_current=float(dictionary["FIB"]["PredictedBeamCurrent"]),
    #                 hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
    #                 resolution=[data.shape[0], data.shape[1]],
    #                 dwell_time=float(dictionary["FIB"]["DwellTime"]),
    #                 shift=Point(
    #                     float(dictionary["FIB"]["ImageShiftX"]),
    #                     float(dictionary["FIB"]["ImageShiftY"]),
    #                 ),
    #                 stigmation=Point(
    #                     float(dictionary["FIB"]["StigmatorX"]),
    #                     float(dictionary["FIB"]["StigmatorY"]),
    #                 ),
    #             ),
    #         )
    #         detector_settings = FibsemDetectorSettings(
    #             type=dictionary["FIB"]["Detector"],
    #             brightness=float(dictionary["FIB"]["Detector0Offset"]) / 100,
    #             contrast=float(dictionary["FIB"]["Detector0Gain"]) / 100,
    #         )

    #     metadata = FibsemImageMetadata(
    #         image_settings=image_settings,
    #         pixel_size=pixel_size,
    #         microscope_state=microscope_state,
    #         # detector_settings=detector_settings,
    #         version=METADATA_VERSION,
    #     )
    #     return FibsemImage(data=data, metadata=metadata)