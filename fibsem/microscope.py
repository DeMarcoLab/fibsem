
import copy
import datetime
import logging
import os
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from psygnal import Signal

THERMO_API_AVAILABLE = False
MINIMUM_AUTOSCRIPT_VERSION_4_7 = parse_version("4.7")

class AutoScriptException(Exception):
    pass


try:
    sys.path.append(r'C:\Program Files\Thermo Scientific AutoScript')
    sys.path.append(r'C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages')
    sys.path.append(r'C:\Program Files\Python36\envs\AutoScript')
    sys.path.append(r'C:\Program Files\Python36\envs\AutoScript\Lib\site-packages')
    import autoscript_sdb_microscope_client
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

    version = autoscript_sdb_microscope_client.build_information.INFO_VERSIONSHORT
    try:
        AUTOSCRIPT_VERSION = parse_version(version)
    except InvalidVersion:
        raise AutoScriptException(f"Failed to parse AutoScript version '{version}'")

    if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
        raise AutoScriptException(
            f"AutoScript {version} found. Please update your AutoScript version to 4.7 or higher."
        )
    
    # special case for Monash development environment
    if os.environ.get("COMPUTERNAME", "hostname") == "MU00190108":
        print("Overwriting autoscript version to 4.7, for Monash dev install")
        AUTOSCRIPT_VERSION = MINIMUM_AUTOSCRIPT_VERSION_4_7
        
    from autoscript_sdb_microscope_client._dynamic_object_proxies import (
        CirclePattern,
        CleaningCrossSectionPattern,
        LinePattern,
        RectanglePattern,
        RegularCrossSectionPattern,
    )
    from autoscript_sdb_microscope_client.enumerations import (
        CoordinateSystem,
        ManipulatorCoordinateSystem,
        ManipulatorSavedPosition,
        ManipulatorState,
        MultiChemInsertPosition,
        PatterningState,
        RegularCrossSectionScanMethod,
    )
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        BitmapPatternDefinition,
        GrabFrameSettings,
        Limits,
        Limits2d,
        ManipulatorPosition,
        MoveSettings,
        Rectangle,
        StagePosition,
    )
    THERMO_API_AVAILABLE = True
except AutoScriptException as e:
    logging.warning("Failed to load AutoScript (ThermoFisher): %s", str(e))
    pass
except ImportError as e:
    logging.debug("AutoScript (ThermoFisher) not found: %s", str(e))
    pass
except Exception:
    logging.error("Failed to load AutoScript (ThermoFisher) due to unexpected error", exc_info=True)
    pass


import fibsem.constants as constants
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


class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""
    milling_progress_signal = Signal(dict)
    _last_imaging_settings: ImageSettings
    system: SystemSettings
    _patterns: List
    stage_is_compustage: bool = False

    # live acquisition
    sem_acquisition_signal = Signal(FibsemImage)
    fib_acquisition_signal = Signal(FibsemImage)
    _stop_acquisition_event = threading.Event()
    _acquisition_thread: threading.Thread = None

    @abstractmethod
    def connect_to_microscope(self, ip_address: str, port: int) -> None:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def acquire_image(self, image_settings:ImageSettings) -> FibsemImage:
        pass

    @abstractmethod
    def last_image(self, beam_type: BeamType) -> FibsemImage:
        pass

    @property
    def is_acquiring(self) -> bool:
        """Check if the microscope is currently acquiring an image."""
        return self._acquisition_thread and self._acquisition_thread.is_alive()

    def start_acquisition(self, beam_type: BeamType) -> None:
        """Start the image acquisition process.
        Args:
            beam_type: The beam type to start acquisition for.
        """
        if self.is_acquiring:
            logging.warning("Acquisition thread is already running.")
            return

        # reset stop event if needed
        self._stop_acquisition_event.clear()

        # start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_worker,
            args=(beam_type,),
            daemon=True
        )
        self._acquisition_thread.start()

    def stop_acquisition(self) -> None:
        """Stop the image acquisition process."""
        if self._stop_acquisition_event and not self._stop_acquisition_event.is_set():
            self._stop_acquisition_event.set()
            if self._acquisition_thread:
                self._acquisition_thread.join(timeout=2)
            # Disconnect signal handler
            # self.sem_acquisition_signal.disconnect()
            # self.fib_acquisition_signal.disconnect()

    def _acquisition_worker(self, beam_type: BeamType) -> None:
        """The worker function for the acquisition thread. 
        Acquires images from the microscope, and emits them as signals."""
        pass

    @abstractmethod
    def acquire_chamber_image(self) -> FibsemImage:
        pass

    @abstractmethod
    def autocontrast(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        pass

    @abstractmethod
    def auto_focus(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        pass

    def reset_beam_shifts(self) -> None:
        """Set the beam shift to zero for the electron and ion beams."""
        self.set_beam_shift(Point(0, 0), BeamType.ELECTRON)
        self.set_beam_shift(Point(0, 0), BeamType.ION)

    @abstractmethod
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> Point:
        pass

    def get_stage_position(self) -> FibsemStagePosition:
        """
        Get the current stage position.

        This method retrieves the current stage position from the microscope and returns it as
        a FibsemStagePosition object. FibsemStage Position is in the RAW coordinate frame

        Returns:
            FibsemStagePosition: The current stage position.
        """

        stage_position = self.get("stage_position")
        logging.debug({"msg": "get_stage_position", "pos": stage_position.to_dict()})
        return deepcopy(stage_position)

    @abstractmethod
    def move_stage_absolute(self, position: FibsemStagePosition) -> None:
        pass

    @abstractmethod
    def move_stage_relative(self, position: FibsemStagePosition) -> None:
        pass

    @abstractmethod
    def stable_move(self,dx: float, dy: float, beam_type: BeamType) -> FibsemStagePosition:
        pass

    @abstractmethod
    def vertical_move(self, dy: float, dx: float = 0, static_wd: bool = True) -> None:
        pass

    @abstractmethod
    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        pass

    def move_flat_to_beam(self, beam_type: BeamType, _safe:bool = True) -> None:
        """Move the sample surface flat to the electron or ion beam."""

        _check_stage(self.system, rotation= True, tilt=True)
        stage_settings = self.system.stage
        shuttle_pre_tilt = stage_settings.shuttle_pre_tilt

        if beam_type is BeamType.ELECTRON:
            rotation = np.deg2rad(stage_settings.rotation_reference)
            tilt = np.deg2rad(shuttle_pre_tilt)

        if beam_type is BeamType.ION:
            rotation = np.deg2rad(stage_settings.rotation_180)
            tilt = np.deg2rad(self.system.ion.column_tilt - shuttle_pre_tilt)

        # new style
        # omap = {BeamType.ELECTRON: "SEM", BeamType.ION: "FIB"}
        # pos = self.get_orientation(omap[beam_type])
        # rotation, tilt = pos.r, pos.t

        # compustage is tilted by 180 degrees for flat to beam, because we image the backside fo the grid,
        # therefore, we need to offset the tilt by 180 degrees
        if self.stage_is_compustage and beam_type is BeamType.ION:
            tilt = -np.pi + tilt
            
        # updated safe rotation move
        logging.info(f"moving flat to {beam_type.name}")
        stage_position = FibsemStagePosition(r=rotation, t=tilt, coordinate_system="Raw")

        logging.debug({"msg": "move_flat_to_beam", "stage_position": stage_position.to_dict(), "beam_type": beam_type.name})

        if _safe:
            self.safe_absolute_stage_movement(stage_position)
        else:
            self.move_stage_absolute(stage_position)

    @abstractmethod
    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        pass

    def get_manipulator_state(self) -> bool:
        """Get the manipulator state (Inserted = True, Retracted = False)"""
        # TODO: convert to enum
        return self.get("manipulator_state")

    def get_manipulator_position(self) -> FibsemManipulatorPosition:
        """Get the manipulator position."""
        return self.get("manipulator_position")

    @abstractmethod
    def insert_manipulator(self, name: str) -> None:
        pass

    @abstractmethod
    def retract_manipulator(self):
        pass

    @abstractmethod
    def move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
        pass

    @abstractmethod
    def move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
        pass

    @abstractmethod
    def move_manipulator_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
        pass

    @abstractmethod
    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str) -> None:
        pass

    @abstractmethod
    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        pass

    @abstractmethod
    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        pass

    @abstractmethod
    def run_milling(self, milling_current: float, asynch: bool) -> None:
        pass

    @abstractmethod
    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        pass

    @abstractmethod
    def stop_milling(self) -> None:
        return 

    @abstractmethod
    def pause_milling(self) -> None:
        return

    @abstractmethod
    def resume_milling(self) -> None:
        return

    @abstractmethod
    def get_milling_state(self) -> MillingState:
        pass 

    @abstractmethod
    def estimate_milling_time(self) -> float:
        pass

    @abstractmethod
    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        pass

    @abstractmethod
    def draw_line(self, pattern_settings: FibsemLineSettings):
        pass

    @abstractmethod
    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        pass

    @abstractmethod
    def draw_bitmap_pattern(
        self,
        pattern_settings: FibsemBitmapSettings,
        path: str,
    ):
        pass

    @abstractmethod
    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        pass

    @abstractmethod
    def setup_sputter(self, *args, **kwargs):
        pass

    @abstractmethod
    def draw_sputter_pattern(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def run_sputter(self, *args, **kwargs):
        pass

    @abstractmethod
    def finish_sputter(self):
        pass

    @abstractmethod
    def get_available_values(self, key: str, beam_type: Optional[BeamType] = None) -> List[Union[str, float, int]]:
        pass

    # TODO: use a decorator instead?
    def get(self, key: str, beam_type: Optional[BeamType] = None) -> Union[float, int, bool, str, list, tuple, Point]:
        """Get wrapper for logging."""
        logging.debug(f"Getting {key} ({beam_type})")
        value = self._get(key, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug({"msg": "get", "key": key, "beam_type": beam_name, "value": value})
        return value

    def set(self, key: str, value: Union[str, float, int, tuple, list, Point], beam_type: Optional[BeamType] = None) -> None:
        """Set wrapper for logging"""
        logging.debug(f"Setting {key} to {value} ({beam_type})")
        self._set(key, value, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug({"msg": "set", "key": key, "beam_type": beam_name, "value": value})

    @abstractmethod
    def _get(self, key: str, beam_type: Optional[BeamType] = None) -> Union[float, int, bool, str, list]:
        pass

    @abstractmethod
    def _set(self, key: str, value: Union[str, float, int, list, tuple, Point], beam_type: Optional[BeamType] = None) -> None:
        pass

    # TODO: i dont think this is needed, you set the beam settings and detector settings separately
    # you can't set image settings, only when acquiring an image
    def get_imaging_settings(self, beam_type: BeamType) -> ImageSettings:
        """Get the current imaging settings for the specified beam type."""
        # TODO: finish this with the other imaging settings... @patrick
        logging.debug(f"Getting {beam_type.name} imaging settings...")
        image_settings = ImageSettings(
            beam_type=beam_type,
            resolution=self.get_resolution(beam_type),
            dwell_time=self.get_dwell_time(beam_type),
            hfw=self.get_field_of_view(beam_type),
            path=self._last_imaging_settings.path,
            filename=self._last_imaging_settings.filename,
        )
        logging.debug({"msg": "get_imaging_settings", "image_settings": image_settings.to_dict(), "beam_type": beam_type.name})
        return image_settings

    def set_imaging_settings(self, image_settings: ImageSettings) -> None:
        """Set the imaging settings for the specified beam type."""
        logging.debug(f"Setting {image_settings.beam_type.name} imaging settings...")
        self.set_resolution(image_settings.resolution, image_settings.beam_type)
        self.set_dwell_time(image_settings.dwell_time, image_settings.beam_type)
        self.set_field_of_view(image_settings.hfw, image_settings.beam_type)
        # self.set("frame_integration", image_settings.frame_integration, image_settings.beam_type)
        # self.set("line_integration", image_settings.line_integration, image_settings.beam_type)
        # self.set("scan_interlacing", image_settings.scan_interlacing, image_settings.beam_type)
        # self.set("drift_correction", image_settings.drift_correction, image_settings.beam_type)

        # TODO: implement the rest of these settings... @patrick
        logging.debug({"msg": "set_imaging_settings", "image_settings": image_settings.to_dict(), "beam_type": image_settings.beam_type.name})

        return 

    def get_beam_settings(self, beam_type: BeamType) -> BeamSettings:
        """Get the current beam settings for the specified beam type.
        """

        logging.debug(f"Getting {beam_type.name} beam settings...")
        beam_settings = BeamSettings(
            beam_type=beam_type,
            working_distance=self.get_working_distance(beam_type),
            beam_current=self.get_beam_current(beam_type),
            voltage=self.get_beam_voltage(beam_type),
            hfw=self.get_field_of_view(beam_type),
            resolution=self.get_resolution(beam_type),  
            dwell_time=self.get_dwell_time(beam_type),
            stigmation=self.get_stigmation(beam_type),
            shift=self.get_beam_shift(beam_type),
            scan_rotation=self.get_scan_rotation(beam_type),
            preset=self.get("preset", beam_type),
        )
        logging.debug({"msg": "get_beam_settings", "beam_settings": beam_settings.to_dict(), "beam_type": beam_type.name})

        return beam_settings

    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        """Set the beam settings for the specified beam type"""
        logging.debug(f"Setting {beam_settings.beam_type.name} beam settings...")
        self.set_working_distance(beam_settings.working_distance, beam_settings.beam_type)
        self.set_beam_current(beam_settings.beam_current, beam_settings.beam_type)
        self.set_beam_voltage(beam_settings.voltage, beam_settings.beam_type)
        self.set_field_of_view(beam_settings.hfw, beam_settings.beam_type)
        self.set_resolution(beam_settings.resolution, beam_settings.beam_type)
        self.set_dwell_time(beam_settings.dwell_time, beam_settings.beam_type)
        self.set_stigmation(beam_settings.stigmation, beam_settings.beam_type)
        self.set_beam_shift(beam_settings.shift, beam_settings.beam_type)
        self.set_scan_rotation(beam_settings.scan_rotation, beam_settings.beam_type)
        self.set("preset", beam_settings.preset, beam_settings.beam_type)

        logging.debug({"msg": "set_beam_settings", "beam_settings": beam_settings.to_dict(), "beam_type": beam_settings.beam_type.name})
        return

    def get_beam_system_settings(self, beam_type: BeamType) -> BeamSystemSettings:
        """Get the current beam system settings for the specified beam type.
        """
        logging.debug(f"Getting {beam_type.name} beam system settings...")
        beam_system_settings = BeamSystemSettings(
            beam_type=beam_type,
            enabled=self.get("beam_enabled", beam_type),
            beam=self.get_beam_settings(beam_type),
            detector=self.get_detector_settings(beam_type),
            eucentric_height=self.get("eucentric_height", beam_type),
            column_tilt=self.get("column_tilt", beam_type),
            plasma=self.get("plasma", beam_type),
            plasma_gas=self.get("plasma_gas", beam_type),
        )

        logging.debug({"msg": "get_beam_system_settings", "settings": beam_system_settings.to_dict(), "beam_type": beam_type.name})
        return beam_system_settings

    def set_beam_system_settings(self, settings: BeamSystemSettings) -> None:
        """Set the beam system settings for the specified beam type.
        """
        beam_type = settings.beam_type
        logging.debug(f"Setting {settings.beam_type.name} beam system settings...")
        self.set("beam_enabled", settings.enabled, beam_type)
        self.set_beam_settings(settings.beam)
        self.set_detector_settings(settings.detector, beam_type)
        self.set("eucentric_height", settings.eucentric_height, beam_type)
        self.set("column_tilt", settings.column_tilt, beam_type)

        if beam_type is BeamType.ION:
            self.set("plasma_gas", settings.plasma_gas, beam_type)
            self.set("plasma", settings.plasma, beam_type)

        logging.debug( {"msg": "set_beam_system_settings", "settings": settings.to_dict(), "beam_type": beam_type.name})
    
        return

    def get_detector_settings(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemDetectorSettings:
        """Get the current detector settings for the specified beam type.
        """
        logging.debug(f"Getting {beam_type.name} detector settings...")
        detector_settings = FibsemDetectorSettings(
            type=self.get_detector_type(beam_type),
            mode=self.get_detector_mode(beam_type),
            brightness=self.get_detector_brightness(beam_type),
            contrast=self.get_detector_contrast(beam_type),
        )
        logging.debug({"msg": "get_detector_settings", "detector_settings": detector_settings.to_dict(), "beam_type": beam_type.name})
        return detector_settings
    
    def set_detector_settings(self, detector_settings: FibsemDetectorSettings, beam_type: BeamType = BeamType.ELECTRON) -> None:
        """Set the detector settings for the specified beam type"""
        logging.debug(f"Setting {beam_type.name} detector settings...")
        self.set_detector_type(detector_settings.type, beam_type)
        self.set_detector_mode(detector_settings.mode, beam_type)
        self.set_detector_brightness(detector_settings.brightness, beam_type)
        self.set_detector_contrast(detector_settings.contrast, beam_type)
        logging.debug({"msg": "set_detector_settings", "detector_settings": detector_settings.to_dict(), "beam_type": beam_type.name})

        return

    def get_microscope_state(self, beam_type: Optional[BeamType] = None) -> MicroscopeState:
        """Get the current microscope state."""

        # default values
        electron_beam, electron_detector = None, None
        ion_beam, ion_detector = None, None
        stage_position = None
        get_electron_state = beam_type in [BeamType.ELECTRON, None]
        get_ion_state = beam_type in [BeamType.ION, None]

        # get the state of the electron beam
        if self.is_available("electron_beam") and get_electron_state:
            electron_beam = self.get_beam_settings(beam_type=BeamType.ELECTRON)
            electron_detector = self.get_detector_settings(beam_type=BeamType.ELECTRON)
 
        # get the state of the ion beam        
        if self.is_available("ion_beam") and get_ion_state:
            ion_beam = self.get_beam_settings(beam_type=BeamType.ION)
            ion_detector = self.get_detector_settings(beam_type=BeamType.ION)

        # get the state of the stage
        if self.is_available("stage"):
            stage_position = self.get_stage_position()       

        current_microscope_state = MicroscopeState(
            timestamp=datetime.datetime.timestamp(datetime.datetime.now()),
            stage_position=stage_position,                                  # get absolute stage coordinates (RAW)
            electron_beam=electron_beam,                                    # electron beam state
            ion_beam=ion_beam,                                              # ion beam state
            electron_detector=electron_detector,                            # electron beam detector state
            ion_detector=ion_detector,                                      # ion beam detector state
        )

        logging.debug({"msg": "get_microscope_state", "state": current_microscope_state.to_dict()})

        return deepcopy(current_microscope_state)

    def set_microscope_state(self, microscope_state: MicroscopeState) -> None:
        """Reset the microscope state to the provided state."""
            
        if self.is_available("electron_beam"):
            if microscope_state.electron_beam is not None:
                self.set_beam_settings(microscope_state.electron_beam)
            if microscope_state.electron_detector is not None:
                self.set_detector_settings(microscope_state.electron_detector, BeamType.ELECTRON)
        if self.is_available("ion_beam"):
            if microscope_state.ion_beam is not None:
                self.set_beam_settings(microscope_state.ion_beam)
            if microscope_state.ion_detector is not None:
                self.set_detector_settings(microscope_state.ion_detector, BeamType.ION)
        if self.is_available("stage") and microscope_state.stage_position is not None:
            self.safe_absolute_stage_movement(microscope_state.stage_position)

        logging.debug({"msg": "set_microscope_state", "state": microscope_state.to_dict()})

        return

    def set_milling_settings(self, mill_settings: FibsemMillingSettings) -> None:
        self.set("active_view", mill_settings.milling_channel, mill_settings.milling_channel)
        self.set("active_device", mill_settings.milling_channel, mill_settings.milling_channel)
        self.set("default_patterning_beam_type", mill_settings.milling_channel, mill_settings.milling_channel)
        self.set("application_file", mill_settings.application_file, mill_settings.milling_channel)
        self.set("patterning_mode", mill_settings.patterning_mode, mill_settings.milling_channel)
        self.set("hfw", mill_settings.hfw, mill_settings.milling_channel)
        self.set("current", mill_settings.milling_current, mill_settings.milling_channel)
        self.set("voltage", mill_settings.milling_voltage, mill_settings.milling_channel)

    def is_available(self, system: str) -> bool:

        if system == "electron_beam":
            return self.system.electron.enabled
        elif system == "ion_beam":
            return self.system.ion.enabled
        elif system == "ion_plasma":
            return self.system.ion.plasma
        elif system == "stage":
            return self.system.stage.enabled
        elif system == "stage_rotation":
            return self.system.stage.rotation
        elif system == "stage_tilt":
            return self.system.stage.tilt
        elif system == "manipulator":
            return self.system.manipulator.enabled
        elif system == "manipulator_rotation":
            return self.system.manipulator.rotation
        elif system == "manipulator_tilt":
            return self.system.manipulator.tilt
        elif system == "gis":
            return self.system.gis.enabled
        elif system == "gis_multichem":
            return self.system.gis.multichem
        elif system == "gis_sputter_coater":
            return self.system.gis.sputter_coater
        else:
            return False

    def set_available(self, system: str, value: bool) -> None:

        if system == "electron_beam":
            self.system.electron.enabled = value
        elif system == "ion_beam":
            self.system.ion.enabled = value
        elif system == "ion_plasma":
            self.system.ion.plasma = value
        elif system == "stage":
            self.system.stage.enabled = value
        elif system == "stage_rotation":
            self.system.stage.rotation = value
        elif system == "stage_tilt":
            self.system.stage.tilt = value
        elif system == "manipulator":
            self.system.manipulator.enabled = value
        elif system == "manipulator_rotation":
            self.system.manipulator.rotation = value
        elif system == "manipulator_tilt":
            self.system.manipulator.tilt = value
        elif system == "gis":
            self.system.gis.enabled = value
        elif system == "gis_multichem":
            self.system.gis.multichem = value
        elif system == "gis_sputter_coater":
            self.system.gis.sputter_coater = value

    def apply_configuration(self, system_settings: Optional[SystemSettings] = None) -> None:
        """Apply the system settings to the microscope."""

        logging.info("Applying Microscope Configuration...")

        if system_settings is None:
            system_settings = self.system
            logging.info("Using current system settings.")

        # apply the system settings
        if self.is_available("electron_beam"):
            self.set_beam_system_settings(system_settings.electron)
        if self.is_available("ion_beam"):
            self.set_beam_system_settings(system_settings.ion)

        if self.is_available("stage"):
            self.system.stage = system_settings.stage

        if self.is_available("manipulator"):
            self.system.manipulator = system_settings.manipulator

        if self.is_available("gis"):
            self.system.gis = system_settings.gis

        # dont update info -> read only
        logging.info("Microscope configuration applied.")
        logging.debug({"msg": "apply_configuration", "system_settings": system_settings.to_dict()})

    @abstractmethod
    def check_available_values(self, key:str, values, beam_type: Optional[BeamType] = None) -> bool:
        pass

    def home(self) -> bool:
        """Home the stage."""
        self.set("stage_home", True)
        return self.get("stage_homed")

    def link_stage(self) -> bool:
        """Link the stage to the working distance"""
        self.set("stage_link", True)
        return self.get("stage_linked")

    def pump(self) -> str:
        """"Pump the chamber."""
        self.set("pump_chamber", True)
        return self.get("chamber_state")

    def vent(self) -> str:
        """Vent the chamber."""
        self.set("vent_chamber", True)
        return self.get("chamber_state")
    
    def turn_on(self, beam_type: BeamType) -> bool:
        """Turn on the specified beam type."""
        self.set("on", True, beam_type)
        return self.get("on", beam_type)

    def turn_off(self, beam_type: BeamType) -> bool:
        "Turn off the specified beam type."
        self.set("on", False, beam_type)
        return self.get("on", beam_type)
    
    def is_on(self, beam_type: BeamType) -> bool:
        """Check if the specified beam type is on."""
        return self.get("on", beam_type)
    
    def blank(self, beam_type: BeamType) -> bool:
        """Blank the specified beam type."""
        self.set("blanked", True, beam_type)
        return self.get("blanked", beam_type)
    
    def unblank(self, beam_type: BeamType) -> bool:
        """Unblank the specified beam type."""
        self.set("blanked", False, beam_type)
        return self.get("blanked", beam_type)
    
    def is_blanked(self, beam_type: BeamType) -> bool:
        """Check if the specified beam type is blanked."""
        return self.get("blanked", beam_type)
    
    def get_available_beams(self) -> List[BeamType]:
        """Get the available beams for the microscope."""
        available_beams = []
        if self.is_available("electron_beam"):
            available_beams.append(BeamType.ELECTRON)
        if self.is_available("ion_beam"):
            available_beams.append(BeamType.ION)
        return available_beams

    def set_spot_scanning_mode(self, point: Point, beam_type: BeamType) -> None:
        """Set the spot scanning mode for the specified beam type."""
        self.set("spot_mode", point, beam_type)
        return

    def set_reduced_area_scanning_mode(self, reduced_area: FibsemRectangle, beam_type: BeamType) -> None:
        """Set the reduced area scanning mode for the specified beam type."""
        self.set("reduced_area", reduced_area, beam_type)
        return

    def set_full_frame_scanning_mode(self, beam_type: BeamType) -> None:
        """Set the full frame scanning mode for the specified beam type."""
        self.set("full_frame", None, beam_type)
        return

    def get_beam_current(self, beam_type: BeamType) -> float:
        """Get the beam current for the specified beam type."""
        return self.get("current", beam_type)

    def set_beam_current(self, current: float, beam_type: BeamType) -> float:
        """Set the beam current for the specified beam type."""
        self.set("current", current, beam_type)
        return self.get("current", beam_type)

    def get_beam_voltage(self, beam_type: BeamType) -> float:
        """Get the beam voltage for the specified beam type."""
        return self.get("voltage", beam_type)

    def set_beam_voltage(self, voltage: float, beam_type: BeamType) -> float:
        """Set the beam voltage for the specified beam type."""
        self.set("voltage", voltage, beam_type)
        return self.get("voltage", beam_type)

    def set_resolution(self, resolution: Tuple[int, int], beam_type: BeamType) -> List[int]:
        """Set the resolution for the specified beam type."""
        self.set("resolution", resolution, beam_type)
        return self.get("resolution", beam_type)

    def get_resolution(self, beam_type: BeamType) -> Tuple[int, int]:
        """Get the resolution for the specified beam type."""
        return self.get("resolution", beam_type)

    def get_field_of_view(self, beam_type: BeamType) -> float:
        """Get the field of view for the specified beam type."""
        return self.get("hfw", beam_type)

    def set_field_of_view(self, hfw: float, beam_type: BeamType) -> float:
        """Set the field of view for the specified beam type."""
        self.set("hfw", hfw, beam_type)
        return self.get("hfw", beam_type)

    def get_working_distance(self, beam_type: BeamType) -> float:
        """Get the working distance for the specified beam type."""
        return self.get("working_distance", beam_type)

    def set_working_distance(self, wd: float, beam_type: BeamType) -> float:
        """Set the working distance for the specified beam type."""
        self.set("working_distance", wd, beam_type)
        return self.get("working_distance", beam_type)

    def get_dwell_time(self, beam_type: BeamType) -> float:
        """Get the dwell time for the specified beam type."""
        return self.get("dwell_time", beam_type)

    def set_dwell_time(self, dwell_time: float, beam_type: BeamType) -> float:
        """Set the dwell time for the specified beam type."""
        self.set("dwell_time", dwell_time, beam_type)
        return self.get("dwell_time", beam_type)

    def get_stigmation(self, beam_type: BeamType) -> Point:
        """Get the stigmation for the specified beam type."""
        return self.get("stigmation", beam_type)

    def set_stigmation(self, stigmation: Point, beam_type: BeamType) -> Point:
        """Set the stigmation for the specified beam type."""
        self.set("stigmation", stigmation, beam_type)
        return self.get("stigmation", beam_type)

    def get_beam_shift(self, beam_type: BeamType) -> Point:
        """Get the beam shift for the specified beam type."""
        return self.get("shift", beam_type)

    def set_beam_shift(self, shift: Point, beam_type: BeamType) -> Point:
        """Set the beam shift for the specified beam type."""
        self.set("shift", shift, beam_type)
        return self.get("shift", beam_type)

    def get_scan_rotation(self, beam_type: BeamType) -> float:
        """Get the scan rotation for the specified beam type."""
        return self.get("scan_rotation", beam_type)

    def set_scan_rotation(self, rotation: float, beam_type: BeamType) -> float:
        """Set the scan rotation for the specified beam type."""
        self.set("scan_rotation", rotation, beam_type)
        return self.get("scan_rotation", beam_type)

    def get_detector_type(self, beam_type: BeamType) -> str:
        """Get the detector type for the specified beam type."""
        return self.get("detector_type", beam_type)

    def set_detector_type(self, detector_type: str, beam_type: BeamType) -> str:
        """Set the detector type for the specified beam type."""
        self.set("detector_type", detector_type, beam_type)
        return self.get("detector_type", beam_type)

    def get_detector_mode(self, beam_type: BeamType) -> str:
        """Get the detector mode for the specified beam type."""
        return self.get("detector_mode", beam_type)

    def set_detector_mode(self, mode: str, beam_type: BeamType) -> str:
        """Set the detector mode for the specified beam type."""
        self.set("detector_mode", mode, beam_type)
        return self.get("detector_mode", beam_type)

    def get_detector_contrast(self, beam_type: BeamType) -> float:
        """Get the detector contrast for the specified beam type."""
        return self.get("detector_contrast", beam_type)

    def set_detector_contrast(self, contrast: float, beam_type: BeamType) -> float:
        """Set the detector contrast for the specified beam type."""
        self.set("detector_contrast", contrast, beam_type)
        return self.get("detector_contrast", beam_type)

    def get_detector_brightness(self, beam_type: BeamType) -> float:
        """Get the detector brightness for the specified beam type."""
        return self.get("detector_brightness", beam_type)

    def set_detector_brightness(self, brightness: float, beam_type: BeamType) -> float:
        """Set the detector brightness for the specified beam type."""
        self.set("detector_brightness", brightness, beam_type)
        return self.get("detector_brightness", beam_type)

    def _get_compucentric_rotation_offset(self) -> FibsemStagePosition:
        return FibsemStagePosition(x=0, y=0) # assume no offset to rotation centre

    def _get_compucentric_rotation_position(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Get the compucentric rotation position for the given stage position. 
        Assumes 180deg rotation. TFS only"""

        # get the compucentric rotation offset
        offset = self._get_compucentric_rotation_offset()

        # convert the raw stage position to specimen coordinates
        specimen_position = deepcopy(position)
        specimen_position.x += offset.x
        specimen_position.y += offset.y

        # apply "compucentric" rotation offset (invert x,y)
        target_position = deepcopy(specimen_position)
        target_position.r += np.radians(180)
        target_position.x = -specimen_position.x
        target_position.y = -specimen_position.y

        # convert the target position to raw coordinates
        target_position.x -= offset.x
        target_position.y -= offset.y

        return target_position

    def get_target_position(self, stage_position: FibsemStagePosition, target_orientation: str) -> FibsemStagePosition:
        """Convert the stage position to the target position for the given orientation."""

        currrent_orientation = self.get_stage_orientation(stage_position)
        logging.info(f"Getting target position for {target_orientation} from {currrent_orientation}")

        if currrent_orientation == target_orientation:
            return stage_position

        if currrent_orientation == "UNKNOWN":
            raise ValueError("Unknown orientation. Cannot convert stage position.")

        orientation = self.get_orientation(target_orientation)

        if currrent_orientation in ["SEM", "MILLING"] and target_orientation == "FIB":
            # Convert from SEM/MILLING to FIB
            target_position = self._get_compucentric_rotation_position(stage_position)
            target_position.r = orientation.r
            target_position.t = orientation.t

        elif currrent_orientation == "FIB" and target_orientation in ["SEM", "MILLING"]:
            # Convert from FIB to SEM/MILLING
            target_position = self._get_compucentric_rotation_position(stage_position)
            target_position.r = orientation.r
            target_position.t = orientation.t
        elif currrent_orientation == "SEM" and target_orientation == "MILLING":
            # Convert from SEM to MILLING
            target_position = stage_position
            target_position.r = orientation.r
            target_position.t = orientation.t
        elif currrent_orientation == "MILLING" and target_orientation == "SEM":
            # Convert from MILLING to SEM
            target_position = stage_position
            target_position.r = orientation.r
            target_position.t = orientation.t
        else:
            raise ValueError(f"Cannot convert from {currrent_orientation} to {target_orientation}")

        return target_position

    def get_stage_orientation(self, stage_position: Optional[FibsemStagePosition] = None) -> str:
        """Get the orientation of the stage position."""
        return NotImplemented

    def get_orientation(self, orientation: str) -> FibsemStagePosition:
        """Get the orientation (r,t) for the given orientation string."""

        stage_settings = self.system.stage
        shuttle_pre_tilt = stage_settings.shuttle_pre_tilt  # deg
        milling_angle = stage_settings.milling_angle        # deg

        # needs to be dynmaically updated as it can change.
        from fibsem.transformations import get_stage_tilt_from_milling_angle
        milling_stage_tilt = get_stage_tilt_from_milling_angle(self, np.radians(milling_angle))

        self.orientations = {
            "SEM": FibsemStagePosition(
                r=np.radians(stage_settings.rotation_reference),
                t=np.radians(shuttle_pre_tilt),
            ),
            "FIB": FibsemStagePosition(
                r=np.radians(stage_settings.rotation_180),
                t=np.radians(self.system.ion.column_tilt - shuttle_pre_tilt),
            ),
            "MILLING": FibsemStagePosition(
                r=np.radians(stage_settings.rotation_reference),
                t=milling_stage_tilt
            ),
        }

        if self.stage_is_compustage:
            self.orientations["FIB"].t -= np.radians(180)

        if orientation not in self.orientations:
            raise ValueError(f"Orientation {orientation} not supported.")

        return self.orientations[orientation]

    def get_current_milling_angle(self) -> float:
        """Get the current milling angle in degrees based on the current stage tilt."""

        from fibsem.transformations import convert_stage_tilt_to_milling_angle

        # NOTE: this is only valid for sem orientation
        if self.get_stage_orientation() == "FIB":
            return 90  # stage-tilt + pre-tilt + 90 - column-tilt

        # Calculate the milling angle from the stage tilt
        milling_angle = convert_stage_tilt_to_milling_angle(
            stage_tilt=self.get_stage_position().t, 
            pretilt=np.radians(self.system.stage.shuttle_pre_tilt), 
            column_tilt=np.radians(self.system.ion.column_tilt)
        )
        return np.degrees(milling_angle)


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

        autocontrast(self, beam_type: BeamType) -> None: 
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

        vertical_move(self,  dy: float, dx: float = 0, static_wd: bool = True) -> None:
            Move the stage vertically to correct eucentric point
        
        get_manipulator_position(self) -> FibsemManipulatorPosition:
            Get the current manipulator position.
        
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

        _y_corrected_stage_movement(self, expected_y: float, beam_type: BeamType = BeamType.ELECTRON) -> FibsemStagePosition:
            Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).
    """

    def __init__(self, system_settings: Optional[SystemSettings] = None):
        if not THERMO_API_AVAILABLE:
            raise Exception("Autoscript (ThermoFisher) not installed. Please see the user guide for installation instructions.")            

        # create microscope client 
        self.connection = SdbMicroscopeClient()

        # initialise system settings
        self.system: SystemSettings = system_settings
        self._patterns: List = []

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

        # logging
        logging.debug({"msg": "create_microscope_client", "system_settings": system_settings.to_dict()})

    def reconnect(self):
        """Attempt to reconnect to the microscope client."""
        if not hasattr(self, "system"):
            raise Exception("Please connect to the microscope first")

        self.disconnect()
        self.connect_to_microscope(self.system.info.ip_address)

    def disconnect(self):
        """Disconnect from the microscope client."""
        self.connection.disconnect()
        del self.connection
        self.connection = None

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
        if self.connection is None:
            self.connection = SdbMicroscopeClient()

        # TODO: get the port
        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        self.connection.connect(host=ip_address, port=port)
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")

        # system information
        self.system.info.model = self.connection.service.system.name
        self.system.info.serial_number = self.connection.service.system.serial_number
        self.system.info.hardware_version = self.connection.service.system.version
        self.system.info.software_version = self.connection.service.autoscript.client.version
        info = self.system.info
        logging.info(f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}.")

        # autoscript information
        logging.info(f"Autoscript Client: {self.connection.service.autoscript.client.version}")
        logging.info(f"Autoscript Server: {self.connection.service.autoscript.server.version}")

        self.reset_beam_shifts()

        # assign stage
        if self.connection.specimen.compustage.is_installed:
            self.stage = self.connection.specimen.compustage
            self.stage_is_compustage = True
            self._default_stage_coordinate_system = CoordinateSystem.SPECIMEN
        elif self.connection.specimen.stage.is_installed:
            self.stage = self.connection.specimen.stage
            self.stage_is_compustage = False
            self._default_stage_coordinate_system = CoordinateSystem.RAW
        else:
            self.stage = None
            self.stage_is_compustage = False
            logging.warning("No stage is installed on the microscope.")

        # set default coordinate system
        self.stage.set_default_coordinate_system(self._default_stage_coordinate_system)
        # TODO: set default move settings, is this dependent on the stage type?
        self._default_application_file = "Si"

        self._last_imaging_settings: ImageSettings = ImageSettings()
        self.milling_channel: BeamType = BeamType.ION

    def set_channel(self, channel: BeamType) -> None:
        """
        Set the active channel for the microscope.

        Args:
            channel (BeamType): The beam type to set as the active channel.
        """
        # TODO: create mapping for the other channels/devices
        self.connection.imaging.set_active_view(channel.value)
        self.connection.imaging.set_active_device(channel.value)
        logging.debug(f"Set active channel to {channel.name}")
        
    def acquire_image(self, image_settings: Optional[ImageSettings] = None, beam_type: Optional[BeamType] = None) -> FibsemImage:
        """
        Acquire a new image with the specified settings.

            Args:
            image_settings (ImageSettings): The settings for the new image.
            beam_type (BeamType, optional): The beam type to use with current settings.
                Used only if image_settings is not provided.

        Returns:
            FibsemImage: A new FibsemImage object representing the acquired image.
        """
        if beam_type is not None:
            return self.acquire_image3(image_settings=None, beam_type=beam_type)

        # set reduced area settings
        if image_settings.reduced_area is not None:
            rect = image_settings.reduced_area
            reduced_area = Rectangle(rect.left, rect.top, rect.width, rect.height)
            logging.debug(f"Set reduced are: {reduced_area} for beam type {image_settings.beam_type}")
        else:
            reduced_area = None
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        # set the imaging hfw
        self.set_field_of_view(hfw=image_settings.hfw, beam_type=image_settings.beam_type)

        logging.info(f"acquiring new {image_settings.beam_type.name} image.")
        self.set_channel(image_settings.beam_type)

        # set the imaging frame settings
        frame_settings = GrabFrameSettings(
            resolution=f"{image_settings.resolution[0]}x{image_settings.resolution[1]}",
            dwell_time=image_settings.dwell_time,
            reduced_area=reduced_area,
            line_integration=image_settings.line_integration,
            scan_interlacing=image_settings.scan_interlacing,
            frame_integration=image_settings.frame_integration,
            drift_correction=image_settings.drift_correction,
        )

        image = self.connection.imaging.grab_frame(frame_settings)

        # restore to full frame imaging
        if image_settings.reduced_area is not None:
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        # get the microscope state (for metadata)
        # TODO: convert to using fromAdornedImage, we dont need to full state
        # we should just get the 'state' of the image beam, e.g. stage, beam, detector for electron
        # therefore we don't trigger the view to switch
        state = self.get_microscope_state(beam_type=image_settings.beam_type)

        fibsem_image = FibsemImage.fromAdornedImage(
            copy.deepcopy(image), 
            copy.deepcopy(image_settings), 
            copy.deepcopy(state),
        )

        # set additional metadata
        fibsem_image.metadata.user = self.user
        fibsem_image.metadata.experiment = self.experiment
        fibsem_image.metadata.system = self.system

        # store last imaging settings
        self._last_imaging_settings = image_settings

        logging.debug({"msg": "acquire_image", "metadata": fibsem_image.metadata.to_dict()})

        return fibsem_image

    def _acquire_image2(self, beam_type: BeamType, frame_settings: Optional['GrabFrameSettings'] = None) -> FibsemImage:
        """
        Acquire an image with the specified beam type and frame settings, and return it as a FibsemImage.
        NOTE: this method is used for the acquisition worker thread, don't use it directly.

        Args:
            beam_type: The beam type to use for acquisition.
            frame_settings: The frame settings for the acquisition (Optional).

        Returns:
            FibsemImage: The acquired image.
        """
        # set the active view and device
        self.set_channel(channel=beam_type)
        
        # acquire the frame
        adorned_image: AdornedImage = self.connection.imaging.grab_frame(settings=frame_settings)

        # get the required metadata, convert to FibsemImage
        state = self.get_microscope_state(beam_type=beam_type)
        image_settings = self.get_imaging_settings(beam_type=beam_type)

        image = FibsemImage.fromAdornedImage(
            copy.deepcopy(adorned_image), 
            copy.deepcopy(image_settings), 
            copy.deepcopy(state),
        )

        # set additional metadata
        image.metadata.user = self.user
        image.metadata.experiment = self.experiment
        image.metadata.system = self.system

        return image

    def acquire_image3(self, image_settings: Optional[ImageSettings] = None, beam_type: Optional[BeamType] = None) -> FibsemImage:
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
            image = microscope.acquire_image3(image_settings=settings)

            # Acquire with current settings for a specific beam type
            image = microscope.acquire_image3(beam_type=BeamType.ION)

            # If both provided, image_settings takes precedence
            image = microscope.acquire_image3(image_settings=settings, beam_type=BeamType.ION)  # Uses settings
        """

        # Validate parameters - at least one must be provided
        if image_settings is None and beam_type is None:
            raise ValueError(
                "Must provide either image_settings (to acquire with specific settings) or beam_type (to acquire with current microscope settings for that beam type)."
            )

        if image_settings is not None:
            # Use provided image settings (takes precedence)
            effective_beam_type = image_settings.beam_type
            effective_image_settings = image_settings

            # apply specified image settings, create frame settings
            self._apply_image_settings(image_settings)
            frame_settings = self._create_frame_settings(image_settings)
        else:
            # Use current settings for the specified beam type
            effective_beam_type = beam_type
            effective_image_settings = self.get_imaging_settings(beam_type=beam_type)
            frame_settings = None

        logging.info(f"acquiring new {effective_beam_type.name} image.")

        self.set_channel(effective_beam_type)
        adorned_image: AdornedImage = self.connection.imaging.grab_frame(frame_settings)

        # QUERY: is this required, reduced area is only set for the grab_frame?
        # Restore full frame if reduced area was used (same as acquire_image)
        if image_settings is not None and image_settings.reduced_area is not None:
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        logging.info(f"acquiring new {effective_beam_type.name} image.")

        # Create FibsemImage with metadata (common for both paths)
        state = self.get_microscope_state(beam_type=effective_beam_type)
        fibsem_image = FibsemImage.fromAdornedImage(
            copy.deepcopy(adorned_image),
            copy.deepcopy(effective_image_settings),
            copy.deepcopy(state),
        )

        # Set additional metadata
        self._set_additional_metadata(fibsem_image)

        # Store last imaging settings if image_settings was provided
        if image_settings is not None:
            self._last_imaging_settings = image_settings

        logging.debug(
            {"msg": "acquire_image", "metadata": fibsem_image.metadata.to_dict()}
        )

        return fibsem_image

    def _apply_image_settings(self, image_settings: ImageSettings) -> None:
        """Apply imaging settings to the microscope."""
        # Set reduced area or full frame
        if image_settings.reduced_area is not None:
            logging.debug(
                f"Set reduced area: {image_settings.reduced_area} for beam type {image_settings.beam_type}"
            )
        else:
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        # Set the imaging hfw
        self.set_field_of_view(
            hfw=image_settings.hfw, beam_type=image_settings.beam_type
        )

    def _create_frame_settings(
        self, image_settings: ImageSettings
    ) -> "GrabFrameSettings":
        """Create GrabFrameSettings from ImageSettings."""
        reduced_area = None
        if image_settings.reduced_area is not None:
            rect = image_settings.reduced_area
            reduced_area = Rectangle(rect.left, rect.top, rect.width, rect.height)

        return GrabFrameSettings(
            resolution=f"{image_settings.resolution[0]}x{image_settings.resolution[1]}",
            dwell_time=image_settings.dwell_time,
            reduced_area=reduced_area,
            line_integration=image_settings.line_integration,
            scan_interlacing=image_settings.scan_interlacing,
            frame_integration=image_settings.frame_integration,
            drift_correction=image_settings.drift_correction,
        )

    def _set_additional_metadata(self, fibsem_image: FibsemImage) -> None:
        """Set additional metadata for the FibsemImage."""
        fibsem_image.metadata.user = self.user
        fibsem_image.metadata.experiment = self.experiment
        fibsem_image.metadata.system = self.system

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
        _check_beam(beam_type = beam_type, settings = self.system)

        # set active view and device
        self.set_channel(beam_type)

        # get the last image
        image = self.connection.imaging.get_image()
        image = AdornedImage(data=image.data.astype(np.uint8), metadata=image.metadata)

        # get the microscope state (for metadata)
        state = self.get_microscope_state(beam_type=beam_type)

        # create the fibsem image
        fibsem_image = FibsemImage.fromAdornedImage(adorned=image,
                                                    image_settings=None,
                                                    state=state,
                                                    beam_type=beam_type)

        # set additional metadata
        fibsem_image.metadata.user = self.user
        fibsem_image.metadata.experiment = self.experiment
        fibsem_image.metadata.system = self.system

        logging.debug({"msg": "acquire_image", "metadata": fibsem_image.metadata.to_dict()})

        return fibsem_image

    def acquire_chamber_image(self) -> FibsemImage:
        """Acquire an image of the chamber inside."""
        self.connection.imaging.set_active_view(4)
        self.connection.imaging.set_active_device(3)
        image = self.connection.imaging.get_image()
        logging.debug({"msg": "acquire_chamber_image"})
        return FibsemImage(data=image.data, metadata=None)

    def _acquisition_worker(self, beam_type: BeamType):
        """Worker thread for image acquisition."""

        # TODO: add lock

        self.set_channel(channel=beam_type)

        try:
            while True:
                if self._stop_acquisition_event.is_set():
                    break

                # acquire image using current beam settings
                image = self.acquire_image(beam_type=beam_type, image_settings=None)

                # emit the acquired image
                if beam_type is BeamType.ELECTRON:
                    self.sem_acquisition_signal.emit(image)
                if beam_type is BeamType.ION:
                    self.fib_acquisition_signal.emit(image)

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")

    def autocontrast(self, beam_type: BeamType, reduced_area: FibsemRectangle = None) -> None:
        """
        Automatically adjust the microscope image contrast for the specified beam type.

        Args:
            beam_type (BeamType) The imaging beam type for which to adjust the contrast.
        """
        logging.debug(f"Running autocontrast on {beam_type.name}.")
        self.set_channel(beam_type)
        if reduced_area is not None:
            self.set_reduced_area_scanning_mode(reduced_area, beam_type)

        self.connection.auto_functions.run_auto_cb()
        if reduced_area is not None:
            self.set_full_frame_scanning_mode(beam_type)

        logging.debug({"msg": "autocontrast", "beam_type": beam_type.name})

    def auto_focus(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:
        """Automatically focus the specified beam type.

        Args:
            beam_type (BeamType): The imaging beam type for which to focus.
        """
        logging.debug(f"Running auto-focus on {beam_type.name}.")
        self.set_channel(beam_type)
        if reduced_area is not None:
            self.set_reduced_area_scanning_mode(reduced_area, beam_type)

        # run the auto focus
        self.connection.auto_functions.run_auto_focus()

        # restore the full frame scanning mode
        if reduced_area is not None:
            self.set_full_frame_scanning_mode(beam_type)
        logging.debug({"msg": "auto_focus", "beam_type": beam_type.name})

    def beam_shift(self, dx: float, dy: float, beam_type: BeamType = BeamType.ION) -> Point:
        """
        Adjusts the beam shift based on relative values that are provided.

        Args:
            dx: the relative x term
            dy: the relative y term
            beam_type: the beam to shift
        Return:
            Point: the current beam shift of the requested beam_type, as this can now be clipped.
        """
        _check_beam(beam_type, self.system)

        # beam shift limits
        beam= self._get_beam(beam_type=beam_type)
        limits: Limits2d = beam.beam_shift.limits

        # check if requested shift is outside limits
        current_shift = self.get_beam_shift(beam_type=beam_type)
        new_shift = Point(x=current_shift.x + dx, y=current_shift.y + dy)
        if new_shift.x < limits.limits_x.min or new_shift.x > limits.limits_x.max:
            logging.warning(f"Beam shift x value {new_shift.x} is out of bounds: {limits.limits_x}")
        if new_shift.y < limits.limits_y.min or new_shift.y > limits.limits_y.max:
            logging.warning(f"Beam shift y value {new_shift.y} is out of bounds: {limits.limits_y}")

        # clip the requested shift to the limits
        new_shift.x = np.clip(new_shift.x, limits.limits_x.min, limits.limits_x.max)
        new_shift.y = np.clip(new_shift.y, limits.limits_y.min, limits.limits_y.max)
        self.set_beam_shift(shift=new_shift, beam_type=beam_type)

        logging.debug({"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name})

        return self.get_beam_shift(beam_type=beam_type)

    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """
        Move the stage to the specified coordinates.

        Args:
            position: The raw stage position to move to.

        Returns:
            FibsemStagePosition: The stage position after movement.
        """
        _check_stage_movement(self.system, position)

        # get current working distance, to be restored later
        wd = self.get_working_distance(BeamType.ELECTRON)

        # convert to autoscript position
        autoscript_position = position.to_autoscript_position(compustage=self.stage_is_compustage)

        logging.info(f"Moving stage to {position}.")
        self.stage.absolute_move(autoscript_position, MoveSettings(rotate_compucentric=True)) # TODO: This needs at least an optional safe move to prevent collision?

        # restore working distance to adjust for microscope compenstation
        if not self.stage_is_compustage:
            self.set_working_distance(wd, BeamType.ELECTRON)

        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

        return self.get_stage_position()

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """
        Move the stage by the specified relative move.

        Args:
            position: the relative stage position to move by.

        Returns:
            None
        """

        _check_stage_movement(self.system, position)

        logging.info(f"Moving stage by {position}.")

        # convert to autoscript position
        thermo_position = position.to_autoscript_position(self.stage_is_compustage)

        # move stage
        self.stage.relative_move(thermo_position)

        logging.debug({"msg": "move_stage_relative", "position": position.to_dict()})

        return self.get_stage_position()

    # TODO: migrate from stable_move vocab to sample_stage
    def stable_move(self, dx: float, dy: float, beam_type: BeamType, static_wd: bool = False) -> FibsemStagePosition:
        """
        Calculate the corrected stage movements based on the beam_type stage tilt, shuttle pre-tilt, 
        and then move the stage relatively.

        Args:
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
            beam_type (BeamType): beam type to move in
            static_wd (bool, optional): whether to fix the working distance to the eucentric heights. Defaults to False.
        """

        _check_stage(self.system)

        wd = self.get_working_distance(beam_type=BeamType.ELECTRON)

        scan_rotation = self.get_scan_rotation(beam_type=beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # calculate stable movement
        yz_move = self._y_corrected_stage_movement(
            expected_y=dy,
            beam_type=beam_type,
        )
        stage_position = FibsemStagePosition(x=dx, y=yz_move.y, z=yz_move.z, 
                                             r=0, t=0, coordinate_system="RAW")

        # move stage
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        if static_wd:
            wd = self.system.electron.eucentric_height
        
        if not self.stage_is_compustage:
            self.set_working_distance(wd, BeamType.ELECTRON)

        # logging
        logging.debug({"msg": "stable_move", "dx": dx, "dy": dy, 
                "beam_type": beam_type.name, "static_wd": static_wd,
                "working_distance": wd, "scan_rotation": scan_rotation, 
                "position": stage_position.to_dict()})

        return self.get_stage_position()

    def vertical_move(
        self,
        dy: float,
        dx: float = 0.0,
        static_wd: bool = True,
    ) -> FibsemStagePosition:
        """ Move the stage vertically to correct coincidence point

        Args:
            dy (float): distance along the y-axis (image coordinates)
            dx (float, optional): distance along the x-axis (image coordinates). Defaults to 0.0.
            static_wd (bool, optional): whether to fix the working distance. Defaults to True.

        """
        # confirm stage is enabled
        _check_stage(self.system)

        # get current working distance, to be restored later
        wd = self.get_working_distance(beam_type=BeamType.ELECTRON)

        # adjust for scan rotation
        scan_rotation = self.get_scan_rotation(beam_type=BeamType.ION)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # TODO: ARCTIS Do we need to reverse the direction of the movement because of the inverted stage tilt?
        if self.stage_is_compustage:
            stage_tilt = self.get_stage_position().t
            if stage_tilt >= np.deg2rad(-90):
                dy *= -1.0

        # TODO: implement perspective correction
        PERSPECTIVE_CORRECTION = 0.9
        z_move = dy
        if True: #use_perspective: 
            z_move = dy / np.cos(np.deg2rad(90 - self.system.ion.column_tilt)) * PERSPECTIVE_CORRECTION  # TODO: MAGIC NUMBER, 90 - fib tilt

        # manually calculate the dx, dy, dz 
        theta = self.get_stage_position().t # rad
        dy = z_move * np.sin(theta)
        dz = z_move / np.cos(theta)
        stage_position = FibsemStagePosition(x=dx, y=dy, z=dz, coordinate_system="RAW")
        logging.info(f"Vertical movement: {stage_position}")
        self.move_stage_relative(stage_position) # NOTE: this seems to be a bit less than previous... -> perspective correction?

        # restore working distance to adjust for microscope compenstation
        if static_wd and not self.stage_is_compustage:
            self.set_working_distance(wd=self.system.electron.eucentric_height, beam_type=BeamType.ELECTRON)
            self.set_working_distance(wd=self.system.ion.eucentric_height, beam_type=BeamType.ION)
        else:
            self.set_working_distance(wd=wd, beam_type=BeamType.ELECTRON)

        # logging
        logging.debug({"msg": "vertical_move", "dy": dy, "dx": dx, 
                "static_wd": static_wd, "wd": wd, 
                "scan_rotation": scan_rotation, 
                "position": stage_position.to_dict()})


        return self.get_stage_position()

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

        # all angles in radians
        sem_column_tilt = np.deg2rad(self.system.electron.column_tilt)
        fib_column_tilt = np.deg2rad(self.system.ion.column_tilt)

        stage_pretilt = np.deg2rad(self.system.stage.shuttle_pre_tilt)

        stage_rotation_flat_to_eb = np.deg2rad(
            self.system.stage.rotation_reference
        ) % (2 * np.pi)
        stage_rotation_flat_to_ion = np.deg2rad(
            self.system.stage.rotation_180
        ) % (2 * np.pi)

        # current stage position
        current_stage_position = self.get_stage_position()
        stage_rotation = current_stage_position.r % (2 * np.pi)
        stage_tilt = current_stage_position.t

        # TODO: @patrick investigate if these calculations need to be adjusted for compustage...
        # the compustage does not have pre-tilt, cannot rotate, but tilts 180 deg. 
        # Therefore, the rotation will always be 0, pre-tilt will always be 0
        # therefore, I think it should always be treated as a flat stage, that is oriented towards the ion beam (in rotation)?
        # need hardware to confirm this
        # QUESTION: is the compustage always flat to the ion beam? or is it flat to the electron beam?
        # QUESTION: what is the tilt coordinate system (where is 0 degrees, where is 90 degrees, where is 180 degrees)?
        # QUESTION: what does flip do? Is it 180 degrees rotation or tilt? This will affect move_flat_to_beam        
        # ASSUMPTION: (naive) tilt=0 -> flat to electron beam, tilt=52 -> flat to ion

        # new info:
        # rotation always will be zero -> PRETILT_SIGN = 1
        # because we want to image the back of the grid, we need to flip the stage by 180 degrees
        # flat to electron, tilt = -180
        # flat to ion, tilt = -128
        # we may also need to flip the PRETILT_SIGN?

        if self.stage_is_compustage:

            if stage_tilt < 0:
                expected_y *= -1.0

            stage_tilt += np.pi
        # QUERY: for compustage, can we just return the expected y? there is no pre-tilt?

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation # TODO: migrate to orientation
        from fibsem import movement
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_ion, atol=5):
            PRETILT_SIGN = -1.0

        # corrected_pretilt_angle = PRETILT_SIGN * stage_tilt_flat_to_electron
        corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + sem_column_tilt) # electron angle = 0, ion = 52

        # perspective tilt adjustment (difference between perspective view and sample coordinate system)
        if beam_type == BeamType.ELECTRON:
            perspective_tilt_adjustment = -corrected_pretilt_angle
        elif beam_type == BeamType.ION:
            perspective_tilt_adjustment = (-corrected_pretilt_angle - fib_column_tilt)

        # the amount the sample has to move in the y-axis
        y_sample_move = expected_y  / np.cos(stage_tilt + perspective_tilt_adjustment)

        # the amount the stage has to move in each axis
        y_move = y_sample_move * np.cos(corrected_pretilt_angle)
        z_move = -y_sample_move * np.sin(corrected_pretilt_angle) #TODO: investigate this

        return FibsemStagePosition(x=0, y=y_move, z=z_move)
    
    # TODO: update this to an enum
    def get_stage_orientation(self, stage_position: Optional[FibsemStagePosition] = None) -> str:

        # current stage position
        if stage_position is None:
            stage_position = self.get_stage_position()
        stage_rotation = stage_position.r % (2 * np.pi)
        stage_tilt = stage_position.t

        from fibsem import movement
        # TODO: also check xyz ranges?

        sem = self.get_orientation("SEM")
        fib = self.get_orientation("FIB")
        milling = self.get_orientation("MILLING")

        is_sem_rotation = movement.rotation_angle_is_smaller(stage_rotation, sem.r, atol=5) # query: do we need rotation_angle_is_smaller, since we % 2pi the rotation?
        is_fib_rotation = movement.rotation_angle_is_smaller(stage_rotation, fib.r, atol=5)

        is_sem_tilt = np.isclose(stage_tilt, sem.t, atol=0.1)
        is_fib_tilt = np.isclose(stage_tilt, fib.t, atol=0.1)
        is_milling_tilt = np.isclose(stage_tilt, milling.t, atol=0.1)

        if is_sem_rotation and is_sem_tilt:
            return "SEM"
        if is_sem_rotation and is_milling_tilt:
            return "MILLING"
        if is_fib_rotation and is_fib_tilt:
            return "FIB"

        return "UNKNOWN"

    def _safe_rotation_movement(
        self, stage_position: FibsemStagePosition
    ):
        """Tilt the stage flat when performing a large rotation to prevent collision.

        Args:
            stage_position (StagePosition): desired stage position.
        """
        current_position = self.get_stage_position()

        # tilt flat for large rotations to prevent collisions
        from fibsem import movement
        if movement.rotation_angle_is_larger(stage_position.r, current_position.r):

            self.move_stage_absolute(FibsemStagePosition(t=0))
            logging.info("tilting to flat for large rotation.")

        return

    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system

        """
        # safe movements are not required on the compustage, because it doesn't rotate
        if not self.stage_is_compustage:

            # tilt flat for large rotations to prevent collisions
            self._safe_rotation_movement(stage_position)

            # move to compucentric rotation
            self.move_stage_absolute(FibsemStagePosition(r=stage_position.r, coordinate_system="RAW")) # TODO: support compucentric rotation directly

        logging.debug(f"safe moving to {stage_position}")
        self.move_stage_absolute(stage_position)

        logging.debug("safe movement complete.")

        return

    def project_stable_move(self, 
        dx:float, dy:float, 
        beam_type:BeamType, 
        base_position:FibsemStagePosition) -> FibsemStagePosition:
        
        scan_rotation = self.get_scan_rotation(beam_type=beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0
        
        # stable-move-projection
        point_yz = self._y_corrected_stage_movement(dy, beam_type)
        dy, dz = point_yz.y, point_yz.z

        # calculate the corrected move to reach that point from base-state?
        new_position = deepcopy(base_position)
        new_position.x += dx
        new_position.y += dy
        new_position.z += dz

        return new_position
    
    def insert_manipulator(self, name: str = "PARK"):
        """Insert the manipulator to the specified position"""

        if not self.is_available("manipulator"):
            raise ValueError("Manipulator not available.")
         
        if name not in ["PARK", "EUCENTRIC"]:
            raise ValueError(f"insert position {name} not supported.")
        if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
            raise NotImplementedError("Manipulator saved positions not supported in this version. Please upgrade to 4.7 or higher")
        
        # get the saved position name
        saved_position = ManipulatorSavedPosition.PARK if name == "PARK" else ManipulatorSavedPosition.EUCENTRIC

        # get the insert position
        insert_position = self.connection.specimen.manipulator.get_saved_position(
            saved_position, ManipulatorCoordinateSystem.RAW
        )
        # insert the manipulator
        logging.info("inserting manipulator to {saved_position}: {insert_position}.")
        self.connection.specimen.manipulator.insert(insert_position)
        logging.info("insert manipulator complete.")

        # return the manipulator position
        manipulator_position = self.get_manipulator_position()
        logging.debug({"msg": "insert_manipulator", "name": name, "position": manipulator_position.to_dict()})                      
        return manipulator_position

    def retract_manipulator(self):
        """Retract the manipulator"""        

        if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
            raise NotImplementedError("Manipulator saved positions not supported in this version. Please upgrade to 4.7 or higher")

        if not self.is_available("manipulator"):
            raise NotImplementedError("Manipulator not available.")

        # Retract the needle, preserving the correct parking postiion
        needle = self.connection.specimen.manipulator
        park_position = needle.get_saved_position(
            ManipulatorSavedPosition.PARK, ManipulatorCoordinateSystem.RAW
        )

        logging.info(f"retracting needle to {park_position}")
        needle.absolute_move(park_position)
        time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
        logging.info("retracting needle...")
        needle.retract()
        logging.info("retract needle complete")
    
    def move_manipulator_relative(self, position: FibsemManipulatorPosition):
        _check_manipulator_movement(self.system, position)

        logging.info(f"moving manipulator by {position}")

        # convert to autoscript position
        autoscript_position = position.to_autoscript_position()
        # move manipulator relative
        self.connection.specimen.manipulator.relative_move(autoscript_position)
        logging.debug({"msg": "move_manipulator_relative", "position": position.to_dict()})

    def move_manipulator_absolute(self, position: FibsemManipulatorPosition):
        """Move the manipulator to the specified coordinates."""

        _check_manipulator_movement(self.system, position)
        logging.info(f"moving manipulator to {position}")
        
        # convert to autoscript
        autoscript_position = position.to_autoscript_position()
        
        # move manipulator
        self.connection.specimen.manipulator.absolute_move(autoscript_position)
        logging.debug({"msg": "move_manipulator_absolute", "position": position.to_dict()})

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
        Then move the needle relatively. Manipulator movement axis is based on stage tilt, so we need to adjust for that 
        with corrected movements, depending on the stage tilt and imaging perspective.

        BeamType.ELECTRON:  move in x, y (raw coordinates)
        BeamType.ION:       move in x, z (raw coordinates)

        Args:
            microscope (FibsemMicroscope) 
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image corodinates)
            beam_type (BeamType, optional): the beam type to move in. Defaults to BeamType.ELECTRON.
        """
        _check_manipulator(self.system)
        stage_tilt = self.get_stage_position().t

        # xy
        if beam_type is BeamType.ELECTRON:
            x_move = self._x_corrected_needle_movement(expected_x=dx)
            yz_move = self._y_corrected_needle_movement(dy, stage_tilt=stage_tilt)

        # xz,
        if beam_type is BeamType.ION:

            x_move = self._x_corrected_needle_movement(expected_x=dx)
            yz_move = self._z_corrected_needle_movement(expected_z=dy, stage_tilt=stage_tilt)

        # explicitly set the coordinate system
        self.connection.specimen.manipulator.set_default_coordinate_system(
            ManipulatorCoordinateSystem.STAGE
        )
        manipulator_position = FibsemManipulatorPosition(x=x_move.x, y=yz_move.y, 
                                                    z=yz_move.z, 
                                                    r = 0.0 ,coordinate_system="STAGE")
        
        # move manipulator
        self.move_manipulator_relative(manipulator_position)

        return self.get_manipulator_position()
    
    def move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str = None) -> None:
        """Move the manipulator to the specified coordinates, offset by the provided offset."""
        _check_manipulator_movement(self.system, offset)
        
        saved_position = self._get_saved_manipulator_position(name)

        # calculate corrected manipulator movement
        stage_tilt = self.get_stage_position().t
        yz_move = self._z_corrected_needle_movement(offset.z, stage_tilt)

        # adjust for offset
        saved_position.x += offset.x
        saved_position.y += yz_move.y + offset.y
        saved_position.z += yz_move.z  # RAW, up = negative, STAGE: down = negative
        saved_position.r = None  # rotation is not supported

        logging.debug({"msg": "move_manipulator_to_position_offset", 
                       "name": name, "offset": offset.to_dict(), 
                       "saved_position": saved_position.to_dict()})

        # move manipulator absolute
        self.move_manipulator_absolute(saved_position)
        

    def _get_saved_manipulator_position(self, name: str = "PARK") -> FibsemManipulatorPosition:
        
        if name not in ["PARK", "EUCENTRIC"]:
            raise ValueError(f"saved position {name} not supported.")
        if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
            raise NotImplementedError("Manipulator saved positions not supported in this version. Please upgrade to 4.7 or higher")
        
        named_position = ManipulatorSavedPosition.PARK if name == "PARK" else ManipulatorSavedPosition.EUCENTRIC
        autoscript_position = self.connection.specimen.manipulator.get_saved_position(
                named_position, ManipulatorCoordinateSystem.STAGE # TODO: why is this STAGE not RAW?
            )

        # convert to FibsemManipulatorPosition
        manipulator_position = FibsemManipulatorPosition.from_autoscript_position(autoscript_position)        
        
        logging.debug({"msg": "get_saved_manipulator_position", "name": name, "position": manipulator_position.to_dict()})

        return manipulator_position 

    def setup_milling(
        self,
        mill_settings: FibsemMillingSettings,
    ):
        """
        Configure the microscope for milling using the ion beam.

        Args:
            mill_settings (FibsemMillingSettings): Milling settings.
        """
        self.milling_channel = mill_settings.milling_channel
        _check_beam(self.milling_channel, self.system)
        self.set_channel(self.milling_channel)
        self.connection.patterning.set_default_beam_type(self.milling_channel.value)
        self.connection.patterning.set_default_application_file(mill_settings.application_file)
        self._default_application_file = mill_settings.application_file
        self.connection.patterning.mode = mill_settings.patterning_mode
        self.clear_patterns()  # clear any existing patterns
        self.set_field_of_view(hfw=mill_settings.hfw, beam_type=self.milling_channel)
        self.set_beam_current(current=mill_settings.milling_current, beam_type=self.milling_channel)
        self.set_beam_voltage(voltage=mill_settings.milling_voltage, beam_type=self.milling_channel)

        # TODO: migrate to _set_milling_settings():
        # self.milling_channel = mill_settings.milling_channel
        # self._default_application_file = mill_settings.application_file
        # _check_beam(self.milling_channel, self.system)
        # self.set_milling_settings(mill_settings)
        # self.clear_patterns()
    
        logging.debug({"msg": "setup_milling", "mill_settings": mill_settings.to_dict()})

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False):
        """
        Run ion beam milling using the specified milling current.

        Args:
            milling_current (float): The current to use for milling in amps.
            milling_voltage (float): The voltage to use for milling in volts.
            asynch (bool, optional): If True, the milling will be run asynchronously. 
                                     Defaults to False, in which case it will run synchronously.
        """
        if not self.is_available("ion_beam"):
            raise ValueError("Ion beam not available.")
        
        try:
            # change to milling current, voltage # TODO: do this in a more standard way (there are other settings)
            if self.get_beam_voltage(beam_type=self.milling_channel) != milling_voltage:
                self.set_beam_voltage(voltage=milling_voltage, beam_type=self.milling_channel)
            if self.get_beam_current(beam_type=self.milling_channel) != milling_current:
                self.set_beam_current(current=milling_current, beam_type=self.milling_channel)
        except Exception as e:
            logging.warning(f"Failed to set voltage or current: {e}, voltage={milling_voltage}, current={milling_current}")

        # run milling (asynchronously)
        self.set_channel(channel=self.milling_channel)  # the ion beam view
        logging.info(f"running ion beam milling now... asynchronous={asynch}")
        self.start_milling()

        start_time = time.time()
        estimated_time = self.estimate_milling_time()
        remaining_time = estimated_time
        
        if asynch:
            return # return immediately, up to the caller to handle the milling process
        
        MILLING_SLEEP_TIME = 1
        while self.get_milling_state() is MillingState.IDLE: # giving time to start 
            time.sleep(0.5)
        while self.get_milling_state() in ACTIVE_MILLING_STATES:
            # logging.info(f"Patterning State: {self.connection.patterning.state}")
            # TODO: add drift correction support here... generically
            if self.get_milling_state() is MillingState.RUNNING:
                remaining_time -= MILLING_SLEEP_TIME # TODO: investigate if this is a good estimate
            time.sleep(MILLING_SLEEP_TIME)
            # TODO: refresh the remaining time by getting the milling time from the patterning API as user can change the patterns on xtUI

            # update milling progress via signal
            self.milling_progress_signal.emit({"progress": {
                    "state": "update", 
                    "start_time": start_time,
                    "milling_state": self.get_milling_state(),
                    "estimated_time": estimated_time, 
                    "remaining_time": remaining_time}
                    })

        # milling complete
        self.clear_patterns()
                                    
        logging.debug({"msg": "run_milling", "milling_current": milling_current, "milling_voltage": milling_voltage, "asynch": asynch})

    def finish_milling(self, imaging_current: float, imaging_voltage: float):
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        """
        _check_beam(self.milling_channel, self.system)
        self.clear_patterns()
        self.set_beam_current(current=imaging_current, beam_type=self.milling_channel)
        self.set_beam_voltage(voltage=imaging_voltage, beam_type=self.milling_channel)
        self.set("patterning_mode", value="Serial")
         # TODO: store initial imaging settings in setup_milling, restore here, rather than hybrid

        logging.debug({"msg": "finish_milling", "imaging_current": imaging_current, "imaging_voltage": imaging_voltage})

    def start_milling(self) -> None:
        """Start the milling process."""
        if self.get_milling_state() is MillingState.IDLE:
            self.connection.patterning.start()
            logging.info("Starting milling...")

    def stop_milling(self) -> None:
        """Stop the milling process."""
        if self.get_milling_state() in ACTIVE_MILLING_STATES:
            logging.info("Stopping milling...")
            self.connection.patterning.stop()
            logging.info("Milling stopped.")

    def pause_milling(self) -> None:
        """Pause the milling process."""
        if self.get_milling_state() == MillingState.RUNNING:
            logging.info("Pausing milling...")
            self.connection.patterning.pause()
            logging.info("Milling paused.")

    def resume_milling(self) -> None:
        """Resume the milling process."""
        if self.get_milling_state() == MillingState.PAUSED:
            logging.info("Resuming milling...")
            self.connection.patterning.resume()
            logging.info("Milling resumed.")
    
    def get_milling_state(self) -> MillingState:
        """Get the current milling state."""
        self.set_channel(channel=self.milling_channel)
        return MillingState[self.connection.patterning.state.upper()]
    
    def clear_patterns(self):
        """Clear all currently drawn milling patterns."""
        self.connection.patterning.clear_patterns()
        self._patterns = []

    def estimate_milling_time(self) -> float:
        """Calculates the estimated milling time for a list of patterns."""
        total_time = 0
        for pattern in self._patterns:
            total_time += pattern.time

        return total_time

    def get_application_file(self, application_file: str, strict: bool = True) -> str:
        """Get a valid application file for the patterning API.
        The api requires setting a valid application file before creating patterns.
        Args:
            application_file (str): The name of the application file to set as default.
            strict (bool): If True, raises an error if the application file is not available.
                If False, tries to find the closest match to the application file.
                Defaults to True.
        Returns:
                str: The name of the application file that was set as default.
        Raises:
            ValueError: If the application file is not available.
        """

        # check if the application file is valid
        application_files = self.get_available_values("application_file")
        if application_file not in application_files:
            if strict:
                raise ValueError(f"Application file {application_file} not available. Available files: {application_files}")
            from difflib import get_close_matches
            closest_match = get_close_matches(application_file, application_files, n=1)
            if not closest_match:
                raise ValueError(f"Application file {application_file} not available. Available files: {application_files}")
            application_file = str(closest_match[0])

        return application_file

    def set_default_application_file(self, application_file: str, strict: bool = True) -> str:
        """Sets the default application file for the patterning API.
        The api requires setting a valid application file before creating patterns.
        Args:
            application_file (str): The name of the application file to set as default.
        """
        application_file = self.get_application_file(application_file, strict=strict)
        self.connection.patterning.set_default_application_file(application_file)
        logging.debug({"msg": "set_default_application_file", "application_file": self._default_application_file})
        return application_file

    def set_patterning_mode(self, mode: str):
        """Sets the patterning mode for the patterning API.
        The api requires setting a valid patterning mode before creating patterns.
        Args:
            mode (str): The patterning mode to set. Can be "Serial" or "Parallel".
        """
        if mode not in ["Serial", "Parallel"]:
            raise ValueError(f"Patterning mode {mode} not supported. Supported modes: Serial, Parallel")
        
        self.connection.patterning.mode = mode
        logging.debug({"msg": "set_patterning_mode", "mode": mode})
        return mode

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
            AutoscriptError: if an error occurs while creating the pattern.
        """
        
        # get patterning api
        patterning_api = self.connection.patterning
        if pattern_settings.cross_section is CrossSectionPattern.RegularCrossSection:
            create_pattern_function = patterning_api.create_regular_cross_section
            self.connection.patterning.mode = "Serial" # parallel mode not supported for regular cross section
            self.connection.patterning.set_default_application_file("Si-multipass")
        elif pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            create_pattern_function = patterning_api.create_cleaning_cross_section
            self.connection.patterning.mode = "Serial" # parallel mode not supported for cleaning cross section
            self.connection.patterning.set_default_application_file("Si-ccs")
        else:
            create_pattern_function = patterning_api.create_rectangle
            
        # create pattern
        pattern = create_pattern_function(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )

        if not np.isclose(pattern_settings.time, 0.0):
            logging.debug(f"Setting pattern time to {pattern_settings.time}.")
            pattern.time = pattern_settings.time

        # set pattern rotation
        pattern.rotation = pattern_settings.rotation

        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        # set scan direction
        available_scan_directions = self.get_available_values("scan_direction")        
    
        if pattern_settings.scan_direction in available_scan_directions:
            pattern.scan_direction = pattern_settings.scan_direction
        else:
            pattern.scan_direction = "TopToBottom"
            logging.warning(f"Scan direction {pattern_settings.scan_direction} not supported. Using TopToBottom instead.")
            logging.warning(f"Supported scan directions are: {available_scan_directions}")        
        
        # set passes       
        if pattern_settings.passes: # not zero
            if isinstance(pattern, RegularCrossSectionPattern):
                pattern.multi_scan_pass_count = pattern_settings.passes
                pattern.scan_method = 1 # multi scan
            else:
                pattern.dwell_time = pattern.dwell_time * (pattern.pass_count / pattern_settings.passes)
                
                # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
                # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
                # NB: the current must be set before doing this, otherwise it will be out of range

        # restore default application file
        self.connection.patterning.set_default_application_file(self._default_application_file)

        logging.debug({"msg": "draw_rectangle", "pattern_settings": pattern_settings.to_dict()})

        self._patterns.append(pattern)

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
        logging.debug({"msg": "draw_line", "pattern_settings": pattern_settings.to_dict()})
        self._patterns.append(pattern)
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

        Raises:
            autoscript.exceptions.InvalidArgumentException: if any of the pattern parameters are invalid.
        """

        outer_diameter = 2 * pattern_settings.radius
        inner_diameter = 0
        if  pattern_settings.thickness != 0:       
            inner_diameter = outer_diameter - 2*pattern_settings.thickness

        self.connection.patterning.set_default_application_file("Si")
        pattern = self.connection.patterning.create_circle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            outer_diameter = outer_diameter,
            inner_diameter = inner_diameter,
            depth=pattern_settings.depth,
        )
        pattern.application_file = "Si"
        pattern.overlap_r = 0.8
        pattern.overlap_t = 0.8
        self.connection.patterning.set_default_application_file(self._default_application_file)

        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        logging.debug({"msg": "draw_circle", "pattern_settings": pattern_settings.to_dict()})
        self._patterns.append(pattern)
        return pattern
    
    def draw_bitmap_pattern(
        self,
        pattern_settings: FibsemBitmapSettings,
        path: str,
    ):

        bitmap_pattern = BitmapPatternDefinition.load(path)

        pattern = self.connection.patterning.create_bitmap(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
            bitmap_pattern_definition=bitmap_pattern,
        )

        logging.debug({"msg": "draw_bitmap_pattern", "pattern_settings": pattern_settings.to_dict(), "path": path})
        self._patterns.append(pattern)
        return pattern

    def get_gis(self, port: str = None):
        use_multichem = self.is_available("gis_multichem")
        
        if use_multichem:
            gis = self.connection.gas.get_multichem()
        else:
            gis = self.connection.gas.get_gis_port(port)
        logging.debug({"msg": "get_gis", "use_multichem": use_multichem, "port": port})
        self.gis = gis
        return self.gis

    def insert_gis(self, insert_position: str = None) -> None:

        if insert_position:
            logging.info(f"Inserting Multichem GIS to {insert_position}")
            self.gis.insert(insert_position)
        else:
            logging.info("Inserting Gas Injection System")
            self.gis.insert()

        logging.debug({"msg": "insert_gis", "insert_position": insert_position})


    def retract_gis(self):
        """Retract the gis"""
        self.gis.retract()
        logging.debug({"msg": "retract_gis", "use_multichem": self.is_available("gis_multichem")})

    def gis_turn_heater_on(self, gas: str = None) -> None:
        """Turn the heater on and wait for it to get to temperature"""
        logging.info(f"Turning on heater for {gas}")
        if gas is not None:
            self.gis.turn_heater_on(gas)
        else:
            self.gis.turn_heater_on()
        
        logging.info("Waiting for heater to get to temperature...")
        time.sleep(3) # we need to wait a bit

        wait_time = 0
        max_wait_time = 15
        target_temp = 300 # validate this somehow?
        while True:
            if gas is not None:
                temp = self.gis.get_temperature(gas) # multi-chem requires gas name
            else:
                temp = self.gis.get_temperature()
            logging.info(f"Waiting for heater: {temp}K, target={target_temp}, wait_time={wait_time}/{max_wait_time} sec")

            if temp >= target_temp:
                break

            time.sleep(1) # wait for the heat

            wait_time += 1
            if wait_time > max_wait_time:
                raise TimeoutError("Gas Injection Failed to heat within time...")
        
        logging.debug({"msg": "gis_turn_heater_on", "temp": temp, "target_temp": target_temp, 
                                "wait_time": wait_time, "max_wait_time": max_wait_time})

        return 


    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        """Run non-specific cryo deposition protocol.

        # TODO: universalise this for demo, tescan
        """

        use_multichem = self.is_available("gis_multichem")
        port = gis_settings.port
        gas = gis_settings.gas
        duration = gis_settings.duration
        insert_position = gis_settings.insert_position

        logging.debug({"msg": "cryo_depositon_v2", "settings": gis_settings.to_dict()})
        
        # get gis subsystem
        self.get_gis(port)

        # insert gis / multichem
        logging.info(f"Inserting Gas Injection System at {insert_position}")
        if use_multichem is False:
            insert_position = None
        self.insert_gis(insert_position)

        # turn heater on
        gas = gas if use_multichem else None
        self.gis_turn_heater_on(gas)
        
        # run deposition
        logging.info(f"Running deposition for {duration} seconds")
        self.gis.open()
        time.sleep(duration) 
        # TODO: provide more feedback to user
        self.gis.close()

        # turn off heater
        logging.info(f"Turning off heater for {gas}")
        self.gis.turn_heater_off()

        # retract gis / multichem
        logging.info("Retracting Gas Injection System")
        self.retract_gis()
            
        return
        

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
        _check_sputter(self.system)
        self.original_active_view = self.connection.imaging.get_active_view()
        self.set_channel(BeamType.ELECTRON)
        self.connection.patterning.clear_patterns()
        self.connection.patterning.set_default_application_file(protocol["application_file"])
        self.connection.patterning.set_default_beam_type(BeamType.ELECTRON.value)
        self.multichem = self.connection.gas.get_multichem()
        self.multichem.insert(protocol["position"])
        self.multichem.turn_heater_on(protocol["gas"])  # "Pt cryo")
        time.sleep(3)
        
        logging.debug({"msg": "setup_sputter", "protocol": protocol})
        

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
        
        logging.debug({"msg": "draw_sputter_pattern", "hfw": hfw, "line_pattern_length": line_pattern_length, "sputter_time": sputter_time})

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
        _check_sputter(self.system)
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
        _check_sputter(self.system)
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

    def get_available_values(self, key: str, beam_type: Optional[BeamType] = None)-> Tuple:
        """Get a list of available values for a given key.
        Keys: application_file, plasma_gas, current, detector_type, detector_mode
        """

        values = []
        if key == "application_file":
            values = self.connection.patterning.list_all_application_files()

        if beam_type is BeamType.ION and self.system.ion.plasma:
            if key == "plasma_gas":
                values = self.connection.beams.ion_beam.source.plasma_gas.available_values

        if key == "current":
            if beam_type is BeamType.ION and self.is_available("ion_beam"):
                values = self.connection.beams.ion_beam.beam_current.available_values
            elif beam_type is BeamType.ELECTRON and self.is_available("electron_beam"):
                # loop through the beam current range, to match the available choices on microscope
                limits: Limits = self.connection.beams.electron_beam.beam_current.limits
                beam_current = limits.min
                while beam_current <= limits.max:
                    values.append(beam_current)
                    beam_current *= 2.0

        if key == "voltage":
            beam = self._get_beam(beam_type)
            limits: Limits = beam.high_voltage.limits
            # QUERY: match what is displayed on microscope, as list[float], or keep as range?
            # technically we can set any value, but primarily people would use what is on microscope
            # SEM: [1000, 2000, 3000, 5000, 10000, 20000, 30000]
            # FIB: [500, 1000, 2000, 8000, 1600, 30000]
            return (limits.min, limits.max) 
        
        if key == "detector_type":
            values = self.connection.detector.type.available_values
        
        if key == "detector_mode":
            values = self.connection.detector.mode.available_values
        
        if key == "scan_direction":
            TFS_SCAN_DIRECTIONS = [
                "BottomToTop",
                "DynamicAllDirections",
                "DynamicInnerToOuter",
                "DynamicLeftToRight",
                "DynamicTopToBottom",
                "InnerToOuter",
                "LeftToRight",
                "OuterToInner",
                "RightToLeft",
                "TopToBottom",
            ]
            values = TFS_SCAN_DIRECTIONS
        
        if key == "gis_ports":
            if self.is_available("gis"):
                values = self.connection.gas.list_all_gis_ports()
            elif self.is_available("multichem"):
                values = self.connection.gas.list_all_multichem_ports()
            else:
                values = []
                        
        logging.debug({"msg": "get_available_values", "key": key, "values": values})

        return values


    def _get(self, key: str, beam_type: Optional[BeamType] = None) -> Union[int, float, str, list, Point, FibsemStagePosition, FibsemManipulatorPosition, None]:
        """Get a property of the microscope."""
        # TODO: make the list of get and set keys available to the user
        if beam_type is not None:
            beam = self.connection.beams.electron_beam if beam_type == BeamType.ELECTRON else self.connection.beams.ion_beam
            _check_beam(beam_type, self.system)

        if key == "active_view":
            return self.connection.imaging.get_active_view()
        if key == "active_device":
            return self.connection.imaging.get_active_device()

        # beam properties
        if key == "on": 
            return beam.is_on
        if key == "blanked":
            return beam.is_blanked
        if key == "working_distance":
            return beam.working_distance.value
        if key == "current":
            return beam.beam_current.value
        if key == "voltage":
            return beam.high_voltage.value
        if key == "hfw":
            return beam.horizontal_field_width.value
        if key == "dwell_time":
            return beam.scanning.dwell_time.value
        if key == "scan_rotation":
            return beam.scanning.rotation.value
        if key == "voltage_limits":
            return beam.high_voltage.limits
        if key == "voltage_controllable":
            return beam.high_voltage.is_controllable
        if key == "shift": # beam shift
            return Point(beam.beam_shift.value.x, beam.beam_shift.value.y)
        if key == "stigmation": 
            return Point(beam.stigmator.value.x, beam.stigmator.value.y)
        if key == "resolution":
            resolution = beam.scanning.resolution.value
            width, height = int(resolution.split("x")[0]), int(resolution.split("x")[-1])
            return [width, height]

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

        # electron beam properties
        if beam_type is BeamType.ELECTRON:
            if key == "angular_correction_angle":
                return beam.angular_correction.angle.value

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                return self.system.ion.plasma
            else:
                return False

        if key == "plasma_gas":
            if beam_type is BeamType.ION and self.system.ion.plasma:
                return beam.source.plasma_gas.value # might need to check if this is available?
            else:
                return None

        # stage properties
        if key == "stage_position":
            _check_stage(self.system)

            # get stage position in raw coordinates 
            self.stage.set_default_coordinate_system(self._default_stage_coordinate_system) # TODO: remove this once testing is done
            stage_position = FibsemStagePosition.from_autoscript_position(self.stage.current_position)
            return stage_position
        
        if key == "stage_homed":
            _check_stage(self.system)
            return self.stage.is_homed
        if key == "stage_linked":
            _check_stage(self.system)
            return self.stage.is_linked

        # chamber properties
        if key == "chamber_state":
            return self.connection.vacuum.chamber_state
        
        if key == "chamber_pressure":
            return self.connection.vacuum.chamber_pressure.value

        # detector mode and type
        if key in ["detector_mode", "detector_type", "detector_brightness", "detector_contrast"]:
            
            # set beam active view and device
            self.set_channel(beam_type)

            if key == "detector_type":
                return self.connection.detector.type.value
            if key == "detector_mode":
                return self.connection.detector.mode.value
            if key == "detector_brightness":
                return self.connection.detector.brightness.value
            if key == "detector_contrast":
                return self.connection.detector.contrast.value

        # manipulator properties
        if key == "manipulator_position":
            _check_manipulator(self.system)
            position = self.connection.specimen.manipulator.current_position   
            return FibsemManipulatorPosition.from_autoscript_position(position)
        if key == "manipulator_state":
            _check_manipulator(self.system)
            state = self.connection.specimen.manipulator.state                 
            return True if state == ManipulatorState.INSERTED else False

        # manufacturer properties
        if key == "manufacturer":
            return self.system.info.manufacturer
        if key == "model":
            return self.system.info.model
        if key == "serial_number":
            return self.system.info.serial_number
        if key == "software_version":
            return self.system.info.software_version
        if key == "hardware_version":
            return self.system.info.hardware_version
        

            
        # logging.warning(f"Unknown key: {key} ({beam_type})")
        return None    

    def _set(self, key: str, value: Union[str, int, float, BeamType, Point, FibsemRectangle], beam_type: Optional[BeamType] = None) -> None:
        """Set a property of the microscope."""
        # required for setting shift, stigmation
        from autoscript_sdb_microscope_client.structures import Point as ThermoPoint

        # get beam
        if beam_type is not None:
            beam = self.connection.beams.electron_beam if beam_type == BeamType.ELECTRON else self.connection.beams.ion_beam
            _check_beam(beam_type, self.system)

        if key == "active_view":
            self.connection.imaging.set_active_view(value.value)  # the beam type is the active view (in ui)
            return
        if key == "active_device":
            self.connection.imaging.set_active_device(value.value)
            return

        # beam properties
        if key == "working_distance":
            beam.working_distance.value = value
            logging.info(f"{beam_type.name} working distance set to {value} m.")
            return 
        if key == "current":
            beam.beam_current.value = value
            logging.info(f"{beam_type.name} current set to {value} A.")
            return
        if key == "voltage":
            beam.high_voltage.value = value
            logging.info(f"{beam_type.name} voltage set to {value} V.")
            return
        if key == "hfw":
            limits = beam.horizontal_field_width.limits
            value = np.clip(value, limits.min, limits.max-10e-6)
            beam.horizontal_field_width.value = value
            logging.info(f"{beam_type.name} HFW set to {value} m.")
            return 
        if key == "dwell_time":
            beam.scanning.dwell_time.value = value
            logging.info(f"{beam_type.name} dwell time set to {value} s.")
            return
        if key == "scan_rotation":
            beam.scanning.rotation.value = value
            logging.info(f"{beam_type.name} scan rotation set to {value} radians.")
            return
        if key == "shift":
            beam.beam_shift.value = ThermoPoint(value.x, value.y) # TODO: resolve this coordinate system
            logging.info(f"{beam_type.name} shift set to {value}.")
            return
        if key == "stigmation":
            beam.stigmator.value = ThermoPoint(value.x, value.y)
            logging.info(f"{beam_type.name} stigmation set to {value}.")
            return

        if key == "resolution":
            resolution = f"{value[0]}x{value[1]}"  # WidthxHeight e.g. 1536x1024
            beam.scanning.resolution.value = resolution
            return 

        # scanning modes
        if key == "reduced_area":
            beam.scanning.mode.set_reduced_area(left=value.left, 
                                                top=value.top, 
                                                width=value.width, 
                                                height=value.height)
            return

        if key == "spot_mode":
            # value: Point, image pixels
            beam.scanning.mode.set_spot(x=value.x, y=value.y)
            return

        if key == "full_frame":
            beam.scanning.mode.set_full_frame()
            return

        # beam control
        if key == "on":
            _check_beam(beam_type, self.system)
            beam.turn_on() if value else beam.turn_off()
            logging.info(f"{beam_type.name} beam turned {'on' if value else 'off'}.")
            return
        if key == "blanked":
            _check_beam(beam_type, self.system)
            beam.blank() if value else beam.unblank()
            logging.info(f"{beam_type.name} beam {'blanked' if value else 'unblanked'}.")
            return

        # detector properties
        if key in ["detector_mode", "detector_type", "detector_brightness", "detector_contrast"]:
            self.set_channel(beam_type)

            if key == "detector_mode":
                if value in self.connection.detector.mode.available_values:
                    self.connection.detector.mode.value = value
                    logging.info(f"Detector mode set to {value}.")
                else:
                    logging.warning(f"Detector mode {value} not available.")
                return
            if key == "detector_type":
                if value in self.connection.detector.type.available_values:
                    self.connection.detector.type.value = value
                    logging.info(f"Detector type set to {value}.")
                else:
                    logging.warning(f"Detector type {value} not available.")
                return
            if key == "detector_brightness":
                if 0 < value <= 1 :
                    self.connection.detector.brightness.value = value
                    logging.info(f"Detector brightness set to {value}.")
                else:
                    logging.warning(f"Detector brightness {value} not available, must be between 0 and 1.")
                return
            if key == "detector_contrast":
                if 0 < value <= 1 :
                    self.connection.detector.contrast.value = value
                    logging.info(f"Detector contrast set to {value}.")
                else:
                    logging.warning(f"Detector contrast {value} not available, mut be between 0 and 1.")
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

        # electron beam properties
        if beam_type is BeamType.ELECTRON:
            if key == "angular_correction_angle":
                beam.angular_correction.angle.value = value
                logging.info(f"Angular correction angle set to {value} radians.")
                return

            if key == "angular_correction_tilt_correction":
                beam.angular_correction.tilt_correction.turn_on() if value else beam.angular_correction.tilt_correction.turn_off()
                return
    
        # ion beam properties
        if beam_type is BeamType.ION:
            if key == "plasma_gas":
                if not self.system.ion.plasma:
                    logging.debug("Plasma gas cannot be set on this microscope.")
                    return
                if not self.check_available_values("plasma_gas", [value], beam_type):
                    logging.warning(f"Plasma gas {value} not available. Available values: {self.get_available_values('plasma_gas', beam_type)}")
                
                logging.info(f"Setting plasma gas to {value}... this may take some time...")
                beam.source.plasma_gas.value = value
                logging.info(f"Plasma gas set to {value}.")

                return

        # stage properties
        if key == "stage_home":
            _check_stage(self.system)
            logging.info("Homing stage...")
            self.stage.home()
            logging.info("Stage homed.")
            return

        if key == "stage_link":
            _check_stage(self.system)
            
            if self.stage_is_compustage:
                logging.debug("Compustage does not support linking.")
                return

            logging.info("Linking stage...")
            self.stage.link() if value else self.stage.unlink()
            logging.info(f"Stage {'linked' if value else 'unlinked'}.")    
            return

        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info("Pumping chamber...")
                self.connection.vacuum.pump()
                logging.info("Chamber pumped.") 
                return
            else:
                logging.warning(f"Invalid value for pump_chamber: {value}.")
                return

        if key == "vent_chamber":
            if value:
                logging.info("Venting chamber...")
                self.connection.vacuum.vent()
                logging.info("Chamber vented.") 
                return
            else:
                logging.warning(f"Invalid value for vent_chamber: {value}.")
                return

        # patterning
        if key == "patterning_mode":
            if value in ["Serial", "Parallel"]:
                self.connection.patterning.mode = value
                logging.info(f"Patterning mode set to {value}.")
                return

        logging.warning(f"Unknown key: {key} ({beam_type})")

        return

    def check_available_values(self, key:str, values: list, beam_type: Optional[BeamType] = None) -> bool:
        """Check if the given values are available for the given key."""

        available_values = self.get_available_values(key, beam_type)

        if available_values is None:
            return False

        for value in values:
            if value not in available_values:
                return False

            if isinstance(value, float):
                if value < min(available_values) or value > max(available_values):
                    return False
        return True

    def _get_beam(self, beam_type: BeamType) -> Union['ElectronBeam', 'IonBeam']:
        """Get the beam connection api for the given beam type.
        Args:
            beam_type (BeamType): The type of beam to get (ELECTRON or ION).
        Returns:
            Union['ElectronBeam', 'IonBeam']: The autoscript beam connection object for the given beam type."""
        if beam_type is BeamType.ELECTRON:
            return self.connection.beams.electron_beam
        elif beam_type is BeamType.ION:
            return self.connection.beams.ion_beam
        else:
            raise ValueError(f"Unknown beam type: {beam_type}")

    def _get_compucentric_rotation_offset(self) -> FibsemStagePosition:
        """Get the difference between the stage position in specimen coordinates and raw coordinates."""
        # get stage position in speciemn coordinates 
        self.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
        specimen_stage_position = FibsemStagePosition.from_autoscript_position(self.stage.current_position)

        # get stage position in raw coordinates 
        self.stage.set_default_coordinate_system(CoordinateSystem.RAW)
        raw_stage_position = FibsemStagePosition.from_autoscript_position(self.stage.current_position)

        # calculate the offset
        offset = specimen_stage_position - raw_stage_position # XY only

        # restore stage coordinate system
        self.stage.set_default_coordinate_system(self._default_stage_coordinate_system)

        return offset


######################################## Helper functions ########################################

# TODO: remove these, and use integrated class checks
def _check_beam(beam_type: BeamType, settings: SystemSettings):
    """
    Checks if beam is available.
    """
    if beam_type == BeamType.ELECTRON and settings.electron.enabled == False:
        warnings.warn("The microscope does not have an electron beam.")
    if beam_type == BeamType.ION and settings.ion.enabled  == False:
        warnings.warn("The microscope does not have an ion beam.")

def _check_stage(settings, rotation: bool = False, tilt: bool = False):
    """
    Checks if the stage is fully movable.
    """
    if settings.stage.enabled == False:
        warnings.warn("The microscope does not have a moving stage.")
    if settings.stage.rotation == False and rotation == True:
        warnings.warn("The microscope stage does not rotate.")
    if settings.stage.tilt == False and tilt == True:
        warnings.warn("The microscope stage does not tilt.")

def _check_manipulator(settings, rotation: bool = False, tilt: bool = False):
    """
    Checks if the needle is available.
    """
    if settings.manipulator.enabled == False:
        warnings.warn("The microscope does not have a needle.")
    if settings.manipulator.rotation == False and rotation == True:
        warnings.warn("The microscope needle does not rotate.")
    if settings.manipulator.tilt == False and tilt == True:
        warnings.warn("The microscope needle does not tilt.")

def _check_sputter(settings: SystemSettings):
    """
    Checks if the sputter is available.
    """
    if settings.gis.enabled == False:
        warnings.warn("The microscope does not have a GIS system.")
    if settings.gis.multichem == False:
        warnings.warn("The microscope does not have a multichem system.")
    
def _check_stage_movement(settings: SystemSettings, position: FibsemStagePosition):
    req_rotation = position.r is not None 
    req_tilt = position.t is not None
    _check_stage(settings, rotation=req_rotation, tilt=req_tilt)    

def _check_manipulator_movement(settings: SystemSettings, position: FibsemManipulatorPosition):
    req_rotation = position.r is not None
    req_tilt = position.t is not None

    _check_manipulator(settings, rotation=req_rotation, tilt=req_tilt)