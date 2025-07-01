from __future__ import annotations
import glob
import logging
import os
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Dict, List, Optional, Tuple, Union
from collections.abc import Iterator

import numpy as np
from skimage.transform import resize

from fibsem.microscope import (
    FibsemMicroscope,
    ThermoMicroscope,
    _check_beam,
    _check_manipulator,
    _check_manipulator_movement,
    _check_sputter,
    _check_stage,
    _check_stage_movement,
)
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
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
    FibsemPatternSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    MillingState,
    Point,
    SystemSettings,
)

######################## SIMULATOR ########################

SIMULATOR_KNOWN_UNKNOWN_KEYS = ["preset"]

# simulator constants
SIMULATOR_PLASMA_GASES = ["Oxygen", "Argon", "Nitrogen", "Xenon"]
SIMULATOR_SCAN_DIRECTIONS = ["BottomToTop", "LeftToRight", "RightToLeft", "TopToBottom"]
SIMULATOR_APPLICATION_FILES =  ["Si", "Si-multipass", "Si-ccs", "autolamella", "cryo_Pt_dep"]
SIMULATOR_BEAM_CURRENTS = {
    BeamType.ELECTRON: [1.0e-12, 3.0e-12, 10e-12, 30e-12, 0.1e-9, 0.3e-9, 1e-9, 4e-9, 15e-9, 60e-9],
    BeamType.ION: {
        "Xenon": [1.0e-12, 3.0e-12, 10e-12, 30e-12, 0.1e-9, 0.3e-9, 1.00005e-9, 4e-9, 15e-9, 60e-9],
        "Argon": [1.0e-12, 6.0e-12, 20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.4e-9, 28.0e-9, 120.0e-9],
        None: [1.0e-12, 3.0e-12, 20e-12, 41e-12, 90e-12, 0.2e-9, 0.4e-9, 1.000005e-9, 2.0e-9, 4.0e-9, 15e-9], # None = Gallium
    }}

# hack, do this properly @patrick

@dataclass
class DemoMicroscopeClient:
    connected: bool = False

    def connect(self, ip_address: str, port: int = 8080):
        logging.debug(f"Connecting to microscope at {ip_address}:{port}")
        self.connected = True
        logging.debug(f"Connected to microscope at {ip_address}:{port}")

    def disconnect(self):
        self.connected = False


@dataclass
class BeamSystem:
    on: bool
    blanked: bool
    beam: BeamSettings
    detector: FibsemDetectorSettings  
    scanning_mode: str
    scanning_mode_value: Union[None, Point, FibsemRectangle] = None

@dataclass
class ChamberSystem:
    state: str
    pressure: float

@dataclass
class StageSystem:
    is_homed: bool
    is_linked: bool
    position: FibsemStagePosition

@dataclass
class ManipulatorSystem:
    inserted: bool
    position: FibsemManipulatorPosition

@dataclass
class GasInjectionSystem:
    gas: str
    inserted: bool = False
    heated: bool = False
    opened: bool = False
    position: str = None

    def insert(self):
        self.inserted = True
        logging.debug("GIS inserted")

    def retract(self):
        self.inserted = False
        logging.debug("GIS retracted")

    def turn_heater_on(self):
        self.heated = True
        logging.debug("GIS heater on")
    
    def turn_heater_off(self):
        self.heated = False
        logging.debug("GIS heater off")

    def open(self):
        self.opened = True
        logging.debug("GIS opened")

    def close(self):
        self.opened = False
        logging.debug("GIS closed")

@dataclass
class MillingSystem:
    state: MillingState = MillingState.IDLE
    patterns: List[FibsemPatternSettings] = field(default_factory=list)
    patterning_mode: str = "Serial"
    default_beam_type: BeamType = BeamType.ION
    default_application_file: str = "Si"
    application_files: List[str] = field(default_factory=lambda: SIMULATOR_APPLICATION_FILES)

@dataclass
class ImagingSystem:
    active_view: int = BeamType.ELECTRON.value
    active_device: int = BeamType.ELECTRON.value
    last_image: Dict[BeamType, Optional[FibsemImage]] = field(default_factory=dict)
    image_iterators: Dict[BeamType, Iterator[str]] = field(default_factory=dict)  # Image filename iterators

class DemoMicroscope(FibsemMicroscope):
    """Simulator microscope client based on TFS microscopes"""

    def __init__(self, system_settings: SystemSettings):

        # initialise system
        self.connection = DemoMicroscopeClient()
        self.system = system_settings    

        self.chamber = ChamberSystem(state="Pumped", pressure=1e-6)
        self.stage_system = StageSystem(
            is_homed=True,
            is_linked=True,
            position=FibsemStagePosition(
                x=0, y=0, z=0, r=0, t=0, coordinate_system="RAW"
            ),
        )

        self.manipulator_system = ManipulatorSystem(
            inserted=False,
            position=FibsemManipulatorPosition(
                x=0, y=0, z=0, r=0, t=0, coordinate_system="RAW"
            ),
        )

        self.gis_system = GasInjectionSystem(gas="Pt dep")

        self.electron_system = BeamSystem(
            on=True,
            blanked=False,
            beam=BeamSettings(
                beam_type=BeamType.ELECTRON,
                working_distance=4.0e-3,
                beam_current=1e-12,
                voltage=2000,
                hfw=150e-6,
                resolution=[1536, 1024],
                dwell_time=1e-6,
                stigmation=Point(0, 0),
                shift=Point(0, 0),
                scan_rotation=0,
            ),
            detector=FibsemDetectorSettings(
                type="ETD",
                mode="SecondaryElectrons",
                brightness=0.5,
                contrast=0.5,
            ),
            scanning_mode = "full_frame"
        )
            
        self.ion_system = BeamSystem(
            on=True,
            blanked=False,
            beam=BeamSettings(
                beam_type=BeamType.ION,
                working_distance=16.5e-3,
                beam_current=20e-12, 
                voltage=30000,
                hfw=150e-6,
                resolution=[1536, 1024],
                dwell_time=1e-6,
                stigmation=Point(0, 0),
                shift=Point(0, 0),
                scan_rotation=0,
                ), 
            detector=FibsemDetectorSettings(
                type="ETD",
                mode="SecondaryElectrons",
                brightness=0.5,
                contrast=0.5,
            ),
            scanning_mode="full_frame",
            scanning_mode_value = None,
        )
        self.stage_is_compustage: bool = False
        self.milling_system = MillingSystem(patterns=[])
        self.imaging_system = ImagingSystem()

        # setup image iterators
        try:
            self._setup_image_iterators()
        except ValueError as e:
            logging.error("Failed to set up sim image iterators: %s", str(e))
            
        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

        self._last_imaging_settings: ImageSettings = ImageSettings()
        self.milling_channel: BeamType.ION = BeamType.ION
        logging.debug({"msg": "create_microscope_client", "system_settings": system_settings.to_dict()})

    def connect_to_microscope(self, ip_address: str, port: int = 8080) -> None:
        """Connect to the microscope server.
        Args:
            ip_address: The IP address of the microscope server.
            port: The port number of the microscope server.
        """
        # connect to microscope
        self.connection.connect(ip_address=ip_address, port=port)

        # system information
        self.system.info.model="DemoMicroscope"
        self.system.info.serial_number="123456"
        self.system.info.software_version="0.1"
        self.system.info.hardware_version="v0.23"
        self.system.info.ip_address=ip_address
        
        # reset beam shifts
        self.reset_beam_shifts()

        # user logging
        info = self.system.info
        logging.info(f"Microscope client connected to {info.model} with serial number {info.serial_number} and software version {info.software_version}")       

        # logging
        logging.debug({"msg": "connect_to_microscope", "ip_address": ip_address, "port": port, "system_info": info.to_dict() })

        return

    def disconnect(self) -> None:
        """Disconnect from the microscope server."""
        self.connection.disconnect()
        logging.info("Disconnected from Demo Microscope")

    def set_channel(self, beam_type: BeamType) -> None:
        self.imaging_system.active_view = beam_type.value
        self.imaging_system.active_device = beam_type.value

    def acquire_image(self, image_settings: Optional[ImageSettings] = None, beam_type: Optional[BeamType] = None) -> FibsemImage:
        """
        Acquire a new image with the specified settings or current settings for the given beam type.

        Args:
            image_settings (ImageSettings, optional): The settings for the new image. 
                Takes precedence if both parameters are provided.
            beam_type (BeamType, optional): The beam type to use with current settings. 
                Used only if image_settings is not provided.

        Returns:
            FibsemImage: A new FibsemImage representing the acquired image.
            
        Raises:
            ValueError: If neither image_settings nor beam_type is provided.
            
        Examples:
            # Acquire with specific settings
            settings = ImageSettings(beam_type=BeamType.ELECTRON, hfw=1e-6, resolution=(1024, 1024))
            image = microscope.acquire_image(image_settings=settings)
            
            # Acquire with current settings for a specific beam type
            image = microscope.acquire_image(beam_type=BeamType.ION)
            
            # If both provided, image_settings takes precedence
            image = microscope.acquire_image(image_settings=settings, beam_type=BeamType.ION)  # Uses settings
        """
        
        # Validate parameters - at least one must be provided
        if image_settings is None and beam_type is None:
            raise ValueError("Must provide either image_settings (to acquire with specific settings) or beam_type (to acquire with current microscope settings for that beam type).")
        
        # Determine which beam type and settings to use (image_settings takes precedence)
        if image_settings is not None:
            # Use provided image settings
            effective_beam_type = image_settings.beam_type
            effective_image_settings = image_settings
        else:
            # Use current settings for the specified beam type
            effective_beam_type = beam_type
            effective_image_settings = self.get_imaging_settings(beam_type=beam_type)

        logging.info(f"acquiring new {effective_beam_type.name} image.")

        # get state for image metadata
        microscope_state = self.get_microscope_state(beam_type=effective_beam_type)

        # construct image (random noise)
        image = FibsemImage.generate_blank_image(
            resolution=effective_image_settings.resolution,
            hfw=effective_image_settings.hfw,
            random=True
        )

        # generate the next image from the sequence iterator
        if self.use_image_sequence:
            image.data  = self._generate_next_image(beam_type=effective_beam_type, 
                                                    output_shape=image.data.shape, 
                                                    dtype=image.data.dtype)

        # add additional metadata
        image.metadata.image_settings = effective_image_settings
        image.metadata.microscope_state = microscope_state
        image.metadata.system = self.system
        image.metadata.experiment = self.experiment
        image.metadata.user = self.user
        
        # crop the image data if reduced area is set
        if effective_image_settings.reduced_area is not None:
            rect = effective_image_settings.reduced_area
            width = int(rect.width * image.data.shape[1])
            height = int(rect.height * image.data.shape[0])

            x0 = int(rect.left * image.data.shape[1])
            y0 = int(rect.top * image.data.shape[0])
            x1, y1 = x0 + width, y0 + height
            image.data =  image.data[y0:y1, x0:x1]

        # store last image, and imaging settings (only if image_settings was provided)
        self.imaging_system.last_image[effective_beam_type] = image
        if image_settings is not None:
            self._last_imaging_settings = image_settings

        logging.debug({"msg": "acquire_image", "metadata": image.metadata.to_dict()})

        return image

    def _generate_next_image(
        self,
        beam_type: BeamType,
        output_shape: Tuple[int, int],
        dtype: np.dtype = np.uint8,
    ) -> np.ndarray:
        """Generate the next image in the sequence for the specified beam type, 
            formatted for the current acquisition settings.
        Args:
            beam_type: The type of beam (electron or ion).
            output_shape: The shape of the output image (height, width).
            dtype: The data type of the output image.
        Returns:
            np.ndarray: The generated image data.
        """
        try:
            # get the next filename from the imaging system
            image_iterator = self.imaging_system.image_iterators.get(beam_type, None)
            if image_iterator is None:
                raise ValueError(f"No image iterator found for beam type {beam_type.name}")

            filename = next(image_iterator)
            
            # check if file still exists
            if not os.path.exists(filename):
                logging.warning(f"Image file not found: {filename}, falling back to random noise")
                return np.random.randint(0, 256, output_shape, dtype=dtype)
            
            # load and process the image
            logging.debug(f"Generating image from {filename} for beam type {beam_type.name}")
            img = FibsemImage.load(filename)
            
            # resize the image data to the specified resolution
            image_data = resize(
                img.data, 
                output_shape=output_shape, 
                anti_aliasing=True, 
                preserve_range=True
            )
            return image_data.astype(dtype)
        except StopIteration as e:
            logging.debug(f"Image sequence for {beam_type.name} exhausted, falling back to random noise: {e}")
            return np.random.randint(0, 256, output_shape, dtype=dtype)

        except (FileNotFoundError, OSError, ValueError) as e:
            logging.warning(f"Failed to load image for {beam_type.name}: {e}, falling back to random noise")
            return np.random.randint(0, 256, output_shape, dtype=dtype)
        except Exception as e:
            logging.error(f"Unexpected error loading image for {beam_type.name}: {e}, falling back to random noise")
            return np.random.randint(0, 256, output_shape, dtype=dtype)

    def _setup_image_iterators(self) -> None:
        """Setup image iterators for simulator image sequences.
        
        Initializes image sequence iterators from configured SEM and FIB data paths.
        Falls back to random noise generation if no simulator configuration is provided.
        """
        self.use_image_sequence = False
        
        if self.system.sim is None:
            logging.debug("No simulator configuration found, using random noise generation")
            return
            
        sem_data_path = self.system.sim.get("sem", None)
        fib_data_path = self.system.sim.get("fib", None)
        
        if sem_data_path is None or fib_data_path is None:
            logging.info("SEM or FIB data path not configured in simulator settings, using random noise generation")
            return

        if not os.path.exists(sem_data_path) or not os.path.exists(fib_data_path):
            raise ValueError(f"SEM data path {sem_data_path} or FIB data path {fib_data_path} does not exist.")

        # find all .tif files in the directories
        self._sem_filenames = sorted(glob.glob(os.path.join(sem_data_path, "*.tif*")))
        self._fib_filenames = sorted(glob.glob(os.path.join(fib_data_path, "*.tif*")))

        # validate that files were found
        if len(self._sem_filenames) == 0:
            raise ValueError(f"No .tif files found in SEM data path: {sem_data_path}")
        if len(self._fib_filenames) == 0:
            raise ValueError(f"No .tif files found in FIB data path: {fib_data_path}")

        # create cycling iterators for continuous image sequences
        use_cycle = self.system.sim.get("use_cycle", False)
        if use_cycle:
            self.imaging_system.image_iterators[BeamType.ELECTRON] = cycle(self._sem_filenames)
            self.imaging_system.image_iterators[BeamType.ION] = cycle(self._fib_filenames)
        else:
            self.imaging_system.image_iterators[BeamType.ELECTRON] = iter(self._sem_filenames)
            self.imaging_system.image_iterators[BeamType.ION] = iter(self._fib_filenames)

        self.use_image_sequence = True
        logging.info(f"Image iterators initialized: {len(self._sem_filenames)} SEM images, {len(self._fib_filenames)} FIB images")

    def last_image(self, beam_type: BeamType) -> Optional[FibsemImage]:
        """Get the last acquired image of the specified beam type.
        Args:
            beam_type: The type of beam (electron or ion).
        Returns:
            FibsemImage: The last acquired image.
        """
        image = self.imaging_system.last_image.get(beam_type)
        logging.debug({"msg": "last_image", "beam_type": beam_type.name, "metadata": image.metadata.to_dict()})
        return image

    def acquire_chamber_image(self) -> FibsemImage:
        """Acquire an image of the chamber inside."""
        image = FibsemImage(
            data=np.random.randint(low=0, high=256, 
                size=(1024, 1536), 
                dtype=np.uint8),
                metadata=None)
        logging.debug({"msg": "acquire_chamber_image"})
        return image

    def _acquisition_worker(self, beam_type: BeamType):
        """Worker thread for image acquisition."""

        # TODO: add lock

        self.set_channel(beam_type)

        try:
            while True:
                if self._stop_acquisition_event.is_set():
                    break

                # "acquire" image
                image = self.acquire_image(beam_type=beam_type)

                # simulate acquisition time
                dwell_time = image.metadata.image_settings.dwell_time
                resolution = image.metadata.image_settings.resolution
                estimated_time = dwell_time * resolution[0] * resolution[1]
                time.sleep(estimated_time)

                # emit the acquired image
                if beam_type is BeamType.ELECTRON:
                    self.sem_acquisition_signal.emit(image)
                if beam_type is BeamType.ION:
                    self.fib_acquisition_signal.emit(image)

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")

    def autocontrast(self, beam_type: BeamType, reduced_area: FibsemRectangle = None) -> None:
        if reduced_area is not None:
            self.set_reduced_area_scanning_mode(reduced_area, beam_type)
        # TODO: implement auto-contrast
        if reduced_area:
            self.set_full_frame_scanning_mode(beam_type)
        logging.debug({"msg": "autocontrast", "beam_type": beam_type.name})

    def auto_focus(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:        
        if reduced_area is not None:
            self.set_reduced_area_scanning_mode(reduced_area, beam_type)
        # TODO: implement auto-focus
        if reduced_area:
            self.set_full_frame_scanning_mode(beam_type)
        logging.debug({"msg": "auto_focus", "beam_type": beam_type.name})
        
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:

        logging.debug({"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name})         

        if beam_type == BeamType.ELECTRON:
            self.electron_system.beam.shift += Point(float(dx), float(dy))
        elif beam_type == BeamType.ION:
            self.ion_system.beam.shift += Point(float(dx), float(dy))

    def get_stage_orientation(self, stage_position: Optional[FibsemStagePosition] = None) -> str:
        return ThermoMicroscope.get_stage_orientation(self, stage_position)

    def _safe_rotation_movement(self, stage_position: FibsemStagePosition) -> None:
        return ThermoMicroscope._safe_rotation_movement(self, stage_position)

    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the specified position using safe strategy"""
        return ThermoMicroscope.safe_absolute_stage_movement(self, stage_position)

    def project_stable_move(self, dx:float, dy:float, beam_type:BeamType, base_position:FibsemStagePosition) -> FibsemStagePosition:
        return ThermoMicroscope.project_stable_move(self, dx, dy, beam_type, base_position)

    def move_stage_absolute(self, position: FibsemStagePosition) -> None:
        """Move the stage to the specified position."""
        _check_stage_movement(self.system, position)
        
        # only assign if not None
        if position.x is not None:
            self.stage_system.position.x = position.x
        if position.y is not None:
            self.stage_system.position.y = position.y
        if position.z is not None:
            self.stage_system.position.z = position.z
        if position.r is not None:
            self.stage_system.position.r = position.r
        if position.t is not None:
            self.stage_system.position.t = position.t
        
        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

        return self.get_stage_position()

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage by the specified amount."""
    
        self.stage_system.position += position

        logging.debug({"msg": "move_stage_relative", "position": position.to_dict()})

        return self.get_stage_position()

    def stable_move(self, dx: float, dy:float, beam_type: BeamType, static_wd: bool=False) -> FibsemStagePosition:
        return ThermoMicroscope.stable_move(self, dx, dy, beam_type, static_wd)


    def vertical_move(self, dy: float, dx:float = 0.0, static_wd: bool=True) -> FibsemStagePosition:
        """Move the stage vertically by the specified amount."""
        return ThermoMicroscope.vertical_move(self, dy, dx, static_wd)

    def _y_corrected_stage_movement(self, expected_y: float, beam_type: BeamType) -> FibsemStagePosition:
        """
        Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        Args:
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
            beam_type (BeamType): beam type to move in
            static_wd (bool, optional): whether to fix the working distance. Defaults to False.
        """
        return ThermoMicroscope._y_corrected_stage_movement(self, expected_y=expected_y, beam_type=beam_type)

    def insert_manipulator(self, name: str = "PARK") -> FibsemManipulatorPosition:
        """Insert the manipulator to the specified position."""
        _check_manipulator(self.system)

        logging.info(f"Inserting manipulator to {name}...")
        self.move_manipulator_absolute(FibsemManipulatorPosition(x=0, y=0, z=180e-6, r=0, t=0))
        self.manipulator_system.inserted = True
        logging.debug({"msg": "insert_manipulator", "name": name})

        return self.get_manipulator_position()

    def retract_manipulator(self):
        """Retract the manipulator."""
        _check_manipulator(self.system)
        logging.info("Retracting manipulator...")
        self.move_manipulator_absolute(FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0))
        self.manipulator_system.inserted = False
        logging.debug({"msg": "retract_manipulator"})

    def move_manipulator_relative(self, position: FibsemManipulatorPosition) -> FibsemManipulatorPosition:
        _check_manipulator_movement(self.system, position)
        logging.info(f"Moving manipulator: {position} (Relative)")
        self.manipulator_system.position += position
        logging.debug({"msg": "move_manipulator_relative", "position": position.to_dict()})
        return self.get_manipulator_position()
    
    def move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> FibsemManipulatorPosition:
        _check_manipulator(self.system)
        logging.info(f"Moving manipulator: {position} (Absolute)")
        self.manipulator_system.position = position
        logging.debug({"msg": "move_manipulator_absolute", "position": position.to_dict()})
        return self.get_manipulator_position()
              
    def move_manipulator_corrected(self, dx: float, dy: float, beam_type: BeamType) -> FibsemManipulatorPosition:
        _check_manipulator(self.system)
        logging.info(f"Moving manipulator: dx={dx:.2e}, dy={dy:.2e}, beam_type = {beam_type.name} (Corrected)")
        self.manipulator_system.position.x += dx
        self.manipulator_system.position.y += dy
        logging.debug({"msg": "move_manipulator_corrected", "dx": dx, "dy": dy, "beam_type": beam_type.name})
        return self.get_manipulator_position()

    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str = None) -> FibsemManipulatorPosition:
        _check_manipulator(self.system)
        if name is None:
            name = "EUCENTRIC"

        position = self._get_saved_manipulator_position(name)
        
        logging.info(f"Moving manipulator: {offset} to {name}")
        self.move_manipulator_absolute(position + offset)
        logging.debug({"msg": "move_manipulator_to_position_offset", "offset": offset.to_dict(), "name": name})
        return self.get_manipulator_position()

    def _get_saved_manipulator_position(self, name: str = "PARK") -> FibsemManipulatorPosition:
        _check_manipulator(self.system)

        if name not in ["PARK", "EUCENTRIC"]:
            raise ValueError(f"Unknown manipulator position: {name}")
        if name == "PARK":
            return FibsemManipulatorPosition(x=0, y=0, z=180e-6, r=0, t=0)
        if name == "EUCENTRIC":
            return FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0)

    def setup_milling(self, mill_settings: FibsemMillingSettings):
        """Setup the milling parameters."""

        _check_beam(mill_settings.milling_channel, self.system)
        self.milling_system.default_application_file = mill_settings.application_file
        self.milling_channel = mill_settings.milling_channel
        self.set_milling_settings(mill_settings=mill_settings)
        self.clear_patterns()
    
        logging.debug({"msg": "setup_milling", "mill_settings": mill_settings.to_dict()})

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False) -> None:
        """Run milling with the specified current and voltage."""
        _check_beam(BeamType.ION, self.system)

        MILLING_SLEEP_TIME = 1

        # start milling
        start_time = time.time()
        estimated_time = self.estimate_milling_time()
        remaining_time = estimated_time
        self.milling_system.state = MillingState.RUNNING

        if asynch:
            return # up to the caller to handle

        while remaining_time > 0 or self.get_milling_state() in ACTIVE_MILLING_STATES:
            logging.debug(f"Running milling: {remaining_time} s remaining.")
            if self.get_milling_state() == MillingState.PAUSED:
                logging.info("Milling paused.")
                time.sleep(MILLING_SLEEP_TIME)
                continue
            if self.get_milling_state() == MillingState.IDLE:
                logging.info("Milling stopped.")
                break
            time.sleep(MILLING_SLEEP_TIME)
            remaining_time -= MILLING_SLEEP_TIME

            # update milling progress via signal
            self.milling_progress_signal.emit({"progress": {
                    "state": "update", 
                    "start_time": start_time,
                    "milling_state": self.get_milling_state(),
                    "estimated_time": estimated_time, 
                    "remaining_time": remaining_time}
                    })

            if remaining_time <= 0: # milling complete
                self.milling_system.state = MillingState.IDLE

        # stop milling and clear patterns
        self.milling_system.state = MillingState.IDLE
        self.clear_patterns()
        logging.debug({"msg": "run_milling", "milling_current": milling_current, "milling_voltage": milling_voltage, "asynch": asynch})

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        """Finish milling by restoring the imaging current and voltage."""
        logging.info(f"Finishing milling: {imaging_current:.2e}")
        self.set("current", imaging_current, self.milling_channel)
        self.set("voltage", imaging_voltage, self.milling_channel)
        self.clear_patterns()

    def clear_patterns(self) -> None:
        self.milling_system.patterns = []

    def stop_milling(self) -> None:
        self.milling_system.state = MillingState.IDLE

    def pause_milling(self) -> None:
        self.milling_system.state = MillingState.PAUSED

    def resume_milling(self) -> None:
        self.milling_system.state = MillingState.RUNNING

    def get_milling_state(self) -> MillingState:
        return self.milling_system.state

    def estimate_milling_time(self) -> float:
        """Estimate the milling time for the specified patterns."""
        PATTERN_SLEEP_TIME = 5
        return PATTERN_SLEEP_TIME * len(self.milling_system.patterns)

    def set_default_application_file(self, application_file: str, strict: bool = True) -> str:
        application_file = ThermoMicroscope.get_application_file(self, application_file, strict)
        self.milling_system.default_application_file = application_file
        return application_file

    def set_patterning_mode(self, patterning_mode: str) -> None:
        """Set the patterning mode for milling."""
        if patterning_mode not in ["Serial", "Parallel"]:
            raise ValueError(f"Invalid patterning mode: {patterning_mode}. Must be 'Serial' or 'Parallel'.")
        self.milling_system.patterning_mode = patterning_mode
        logging.debug({"msg": "set_patterning_mode", "patterning_mode": patterning_mode})

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings) -> None:
        logging.debug({"msg": "draw_rectangle", "pattern_settings": pattern_settings.to_dict()})
        if pattern_settings.time != 0:
            logging.info(f"Setting pattern time to {pattern_settings.time}.")
        self.milling_system.patterns.append(pattern_settings)

    def draw_line(self, pattern_settings: FibsemLineSettings) -> None:
        logging.debug({"msg": "draw_line", "pattern_settings": pattern_settings.to_dict()})
        self.milling_system.patterns.append(pattern_settings)
    
    def draw_circle(self, pattern_settings: FibsemCircleSettings) -> None:
        logging.debug({"msg": "draw_circle", "pattern_settings": pattern_settings.to_dict()})
        self.milling_system.patterns.append(pattern_settings)
    
    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings, path: str) -> None:
        logging.debug({"msg": "draw_bitmap_pattern", "pattern_settings": pattern_settings.to_dict(), "path": path})
        self.milling_system.patterns.append(pattern_settings)

    def setup_sputter(self, protocol: dict) -> None:
        _check_sputter(self.system)
        logging.info(f"Setting up sputter: {protocol}")

    def draw_sputter_pattern(self, hfw: float, line_pattern_length: float, sputter_time: float):
        logging.debug({"msg": "draw_sputter_pattern", "hfw": hfw, "line_pattern_length": line_pattern_length, "sputter_time": sputter_time})

    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        """Run non-specific cryo deposition protocol.

        # TODO: universalise this for demo, tescan
        """

        use_multichem = self.is_available("multichem")
        port = gis_settings.port
        gas = gis_settings.gas
        duration = gis_settings.duration
        insert_position = gis_settings.insert_position

        logging.info({"msg": "inserting gis", "settings": gis_settings.to_dict()})

        gis = self.gis_system

        # insert gis / multichem
        logging.info(f"Inserting Gas Injection System at {insert_position}")
        gis.insert()
    
        logging.info(f"Turning on heater for {gas}")
        # turn on heater
        gis.turn_heater_on()
        time.sleep(3) # wait for the heat
        # TODO: get state feedback, wait for heater to be at temp

        # run deposition
        logging.info(f"Running deposition for {duration} seconds")
        # gis.open()
        time.sleep(duration) 
        gis.close()

        # turn off heater
        logging.info(f"Turning off heater for {gas}")
        gis.turn_heater_off()

        # retract gis / multichem
        logging.info("Retracting Gas Injection System")
        gis.retract()

        return
            
    def run_sputter(self, **kwargs):
        _check_sputter(self.system)
        logging.info(f"Running sputter: {kwargs}")

    def finish_sputter(self, **kwargs):
        _check_sputter(self.system)
        logging.info(f"Finishing sputter: {kwargs}")

    def get_available_values(self, key: str, beam_type: Optional[BeamType] = None) -> List[Union[str, int, float]]:
        """Get the available values for a given key."""
        values = []
        if key == "current":


            # return values based on beam type, and plasma gas
            if beam_type is BeamType.ION:
                plasma_gas = self.get("plasma_gas", beam_type)
                values = SIMULATOR_BEAM_CURRENTS[beam_type][plasma_gas]
            else:
                values = SIMULATOR_BEAM_CURRENTS[beam_type]

        if key == "voltage":
            if beam_type is BeamType.ELECTRON:
                # SEM: [1000, 2000, 3000, 5000, 10000, 20000, 30000]
                values = [2000, 5000, 10000, 20000, 30000]
            elif beam_type is BeamType.ION:
                values = [500, 1000, 2000, 8000, 16000, 30000]
                # FIB: [500, 1000, 2000, 8000, 1600, 30000]

        if key == "application_file":
            values = self.milling_system.application_files

        if key == "detector_type":
            values = ["ETD", "TLD", "EDS"]
        if key == "detector_mode":
            values = ["SecondaryElectrons", "BackscatteredElectrons", "EDS"]

        if key == "scan_direction":
            values = SIMULATOR_SCAN_DIRECTIONS

        if key == "plasma_gas":
            values = SIMULATOR_PLASMA_GASES

        if key == "gis_ports":
            values = ["Pt Dep", "Pt Dep Cryo2"]

        return values

    def _get(self, key, beam_type: BeamType = None) -> Union[float, int, bool, str, list]:
        """Get a value from the microscope."""
        # get beam
        if beam_type is not None:
            beam_system = self.electron_system if beam_type is BeamType.ELECTRON else self.ion_system
            beam, detector = beam_system.beam, beam_system.detector

        # TODO: change this so value is returned, so we can log the return value

        # beam properties
        if key == "on": 
            return beam_system.on
        if key == "blanked":
            return beam_system.blanked
        if key == "voltage":
            return beam.voltage
        if key == "current":
            return beam.beam_current
        if key == "working_distance":
            return beam.working_distance
        if key == "hfw":
            return beam.hfw
        if key == "resolution":
            return beam.resolution
        if key == "dwell_time":
            return beam.dwell_time
        if key == "stigmation":
            return Point(beam.stigmation.x, beam.stigmation.y)
        if key == "shift":
            return Point(beam.shift.x, beam.shift.y)
        if key == "scan_rotation":
            return beam.scan_rotation
        
        # system properties
        if key == "beam_enabled":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.enabled
            elif beam_type is BeamType.ION:
                return self.system.ion.enabled
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")
            
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

        if key == "plasma_gas":
            if beam_type is BeamType.ION and self.system.ion.plasma:
                return self.system.ion.plasma_gas # might need to check if this is available?
            else:
                return None
    
        # stage 
        if key == "stage_position":
            return self.stage_system.position
        if key == "stage_homed":
            return self.stage_system.is_homed
        if key == "stage_linked":
            return self.stage_system.is_linked
        
        # detector properties
        if key == "detector_type":
            return detector.type
        if key == "detector_mode":
            return detector.mode
        if key == "detector_brightness":
            return detector.brightness
        if key == "detector_contrast":
            return detector.contrast

        # manipulator properties
        if key == "manipulator_position":
            return self.manipulator_system.position
        if key == "manipulator_state":  
            return self.manipulator_system.inserted     
        
        # manufacturer properties
        if key == "manufacturer":
            return self.system.info.manufacturer
        if key == "model":
            return self.system.info.model
        if key == "software_version":
            return self.system.info.software_version
        if key == "serial_number":
            return "Unknown"
        if key == "hardware_version":
            return self.system.info.hardware_version

        # chamber properties
        if key == "chamber_state":
            return self.chamber.state
        if key == "chamber_pressure":
            return self.chamber.pressure

        # scanning mode
        if key == "scanning_mode":
            return beam_system.scanning_mode

        if key in SIMULATOR_KNOWN_UNKNOWN_KEYS:
            logging.debug(f"Skipping unknown key: {key} for {beam_type}")
            return None

        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def _set(self, key: str, value, beam_type: BeamType = None) -> None:
        """Set a property of the microscope."""
        
        # get beam
        if beam_type is not None:
            beam_system = self.electron_system if beam_type is BeamType.ELECTRON else self.ion_system            
            beam = beam_system.beam
            detector = beam_system.detector
            _check_beam(beam_type, self.system)

        # voltage
        if key == "voltage":
            beam.voltage = value
            return
        # current
        if key == "current":
            beam.beam_current = value
            return        
        
        if key == "working_distance":
            beam.working_distance = value
            return
        
        if key == "stigmation":
            beam.stigmation = value
            return
        if key == "shift":
            beam.shift = value
            return
        if key == "scan_rotation":
            beam.scan_rotation = value
            return
        if key == "hfw":
            beam.hfw = value
            return
        if key == "resolution":
            beam.resolution = value
            return
        if key == "dwell_time":
            beam.dwell_time = value
            return

        # beam control
        if key == "on":
            beam_system.on = value
            return

        if key == "blanked":
            beam_system.blanked = value
            return

        # detector
        if key == "detector_type":
            detector.type = value
            return
        if key == "detector_mode":
            detector.mode = value
            return 
        if key == "detector_contrast":
            detector.contrast = value
            return
        if key == "detector_brightness":
            detector.brightness = value
            return
        
        # system properties
        if key == "beam_enabled":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.beam.enabled = value
                return 
            elif beam_type is BeamType.ION:
                self.system.ion.beam.enabled = value
                return
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")
            return

        if key == "eucentric_height":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.eucentric_height = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.eucentric_height = value
                return 
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        if key =="column_tilt":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.column_tilt = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.column_tilt = value
                return 
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                self.system.ion.plasma = value
                return

        if beam_type is BeamType.ION:
            if key == "plasma_gas":
                if not self.system.ion.plasma:
                    logging.debug("Plasma gas cannot be set on this microscope.")
                    return
                if not self.check_available_values("plasma_gas", value, beam_type):
                    logging.warning(f"Plasma gas {value} not available. Available values: {self.get_available_values('plasma_gas', beam_type)}")
                    return 
                logging.info(f"Setting plasma gas to {value}... this may take some time...")
                self.system.ion.plasma_gas = value
                logging.info(f"Plasma gas set to {value}.")

                return

        if key == "spot_mode":
            # value: Point, image pixels
            beam_system.scanning_mode = "spot"
            beam_system.scanning_mode_value = value
            return

        if key == "reduced_area":
            beam_system.scanning_mode = "reduced_area"
            beam_system.scanning_mode_value = value
            return

        if key == "full_frame":
            beam_system.scanning_mode = "full_frame"
            beam_system.scanning_mode_value = value
            return

        # imaging system
        if key == "active_view":
            self.imaging_system.active_view = value.value
            return
        if key == "active_device":
            self.imaging_system.active_device = value.value
            return

        # milling
        if key == "patterning_mode":
            self.milling_system.patterning_mode = value
            return
        if key == "application_file":
            self.milling_system.default_application_file = value
            return
        if key == "milling_channel":
            self.milling_channel = value
            return
        if key == "default_patterning_beam_type":
            self.milling_system.default_beam_type = value
            return

        # stage properties
        if key == "stage_home":
            logging.info("Homing stage...")
            self.stage_system.is_homed = True
            logging.info("Stage homed.")
            return
        
        if key == "stage_link":
            if self.stage_is_compustage:
                logging.debug("Compustage does not support linking.")
                return
            logging.info("Linking stage...")
            self.stage_system.is_linked = True
            logging.info("Stage linked.")
            return

        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info("Pumping chamber...")
                self.chamber.state = "Pumped"
                self.chamber.pressure = 1e-6 # 1 uTorr
                logging.info("Chamber pumped.")
            else:
                logging.info(f"Invalid value for pump_chamber: {value}")
            return
        if key == "vent_chamber":
            if value:
                logging.info("Venting chamber...")
                self.chamber.state = "Vented"
                self.chamber.pressure = 1e5
                logging.info("Chamber vented.")
            else:
                logging.info(f"Invalid value for vent_chamber: {value}")
            return

        if key in SIMULATOR_KNOWN_UNKNOWN_KEYS:
            logging.debug(f"Skipping unknown key: {key} for {beam_type}")
            return

        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def check_available_values(self, key: str, value, beam_type: BeamType = None) -> bool:
        logging.info(f"Checking if {key}={value} is available ({beam_type})")

        if key == "plasma_gas":
            return value in self.get_available_values(key, beam_type)
        
        return False
    
    def home(self):
        _check_stage(self.system)
        logging.info("Homing Stage")
        self.stage_system.is_homed = True
        logging.info("Stage homed.")
        return