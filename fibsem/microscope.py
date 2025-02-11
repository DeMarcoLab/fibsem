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
from queue import Queue
from typing import List, Union
from psygnal import Signal
import numpy as np

_THERMO_API_AVAILABLE = False

# DEVELOPMENT
_OVERWRITE_AUTOSCRIPT_VERSION = False
if os.environ.get("COMPUTERNAME", "hostname") == "MU00190108":
    _OVERWRITE_AUTOSCRIPT_VERSION = True
    print("Overwriting autoscript version to 4.7, for Monash dev install")

try:
    sys.path.append('C:\Program Files\Thermo Scientific AutoScript')
    sys.path.append('C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages')
    sys.path.append('C:\Program Files\Python36\envs\AutoScript')
    sys.path.append('C:\Program Files\Python36\envs\AutoScript\Lib\site-packages')
    import autoscript_sdb_microscope_client
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    version = autoscript_sdb_microscope_client.build_information.INFO_VERSIONSHORT
    VERSION = float(version[:3])
    if VERSION < 4.6:
        raise NameError("Please update your AutoScript version to 4.6 or higher.")
    
    if _OVERWRITE_AUTOSCRIPT_VERSION:
        VERSION = 4.7
        
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
        ManipulatorPosition,
        MoveSettings,
        StagePosition,
    )
    _THERMO_API_AVAILABLE = True 
except Exception as e:
    logging.debug("Autoscript (ThermoFisher) not installed.")
    if isinstance(e, NameError):
        raise e 

import fibsem.constants as constants
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
    FibsemPatternSettings,
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
    MillingState,
    ACTIVE_MILLING_STATES,
)


class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""
    milling_progress_signal = Signal(dict)
    _last_imaging_settings: ImageSettings

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
    
    @abstractmethod
    def live_imaging(self, image_settings: ImageSettings, image_queue: Queue, stop_event: threading.Event):
        pass
    
    @abstractmethod
    def acquire_chamber_image(self) -> FibsemImage:
        pass

    @abstractmethod
    def consume_image_queue(self, parent_ui = None, sleep: float = 0.1):
        pass
    
    @abstractmethod
    def autocontrast(self, beam_type: BeamType) -> None:
        pass

    @abstractmethod
    def auto_focus(self, beam_type: BeamType) -> None:
        pass

    def reset_beam_shifts(self) -> None:
        """Set the beam shift to zero for the electron and ion beams."""
        self.set("shift", Point(0, 0), BeamType.ELECTRON)
        self.set("shift", Point(0, 0), BeamType.ION)

    @abstractmethod
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:
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
        return stage_position
    
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
    def get_available_values(self, key:str, beam_type: BeamType = None) -> list:
        pass

    # TODO: use a decorator instead?
    def get(self, key: str, beam_type: BeamType = None) -> Union[float, int, bool, str, list]:
        """Get wrapper for logging."""
        logging.debug(f"Getting {key} ({beam_type})")
        value = self._get(key, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug({"msg": "get", "key": key, "beam_type": beam_name, "value": value})
        return value

    def set(self, key: str, value, beam_type: BeamType = None) -> None:
        """Set wrapper for logging"""
        logging.debug(f"Setting {key} to {value} ({beam_type})")
        self._set(key, value, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug({"msg": "set", "key": key, "beam_type": beam_name, "value": value})
    
    @abstractmethod
    def _get(self, key: str, beam_type: BeamType = None) -> Union[float, int, bool, str, list]:
        pass
        
    @abstractmethod
    def _set(self, key: str, value, beam_type: BeamType = None) -> None:
        pass
        
    
    # TODO: i dont think this is needed, you set the beam settings and detector settings separately
    # you can't set image settings, only when acquiring an image
    def get_imaging_settings(self, beam_type: BeamType) -> ImageSettings:
        """Get the current imaging settings for the specified beam type."""
        # TODO: finish this with the other imaging settings... @patrick
        logging.debug(f"Getting {beam_type.name} imaging settings...")
        image_settings = ImageSettings(
            beam_type=beam_type,
            resolution=self.get("resolution", beam_type),
            dwell_time=self.get("dwell_time", beam_type),
            hfw=self.get("hfw", beam_type),
            path=self._last_imaging_settings.path,
            filename=self._last_imaging_settings.filename,
        )
        logging.debug({"msg": "get_imaging_settings", "image_settings": image_settings.to_dict(), "beam_type": beam_type.name})
        return image_settings

    def set_imaging_settings(self, image_settings: ImageSettings) -> None:
        """Set the imaging settings for the specified beam type."""
        logging.debug(f"Setting {image_settings.beam_type.name} imaging settings...")
        self.set("resolution", image_settings.resolution, image_settings.beam_type)
        self.set("dwell_time", image_settings.dwell_time, image_settings.beam_type)
        self.set("hfw", image_settings.hfw, image_settings.beam_type)
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
            working_distance=self.get("working_distance", beam_type),
            beam_current=self.get("current", beam_type),
            voltage=self.get("voltage", beam_type),
            hfw=self.get("hfw", beam_type),
            resolution=self.get("resolution", beam_type),  
            dwell_time=self.get("dwell_time" , beam_type),
            stigmation=self.get("stigmation", beam_type),
            shift=self.get("shift", beam_type),
            scan_rotation=self.get("scan_rotation", beam_type),
            preset=self.get("preset", beam_type),
        )
        logging.debug({"msg": "get_beam_settings", "beam_settings": beam_settings.to_dict(), "beam_type": beam_type.name})
        
        return beam_settings
        
    def set_beam_settings(self, beam_settings: BeamSettings) -> None:
        """Set the beam settings for the specified beam type"""
        logging.debug(f"Setting {beam_settings.beam_type.name} beam settings...")
        self.set("working_distance", beam_settings.working_distance, beam_settings.beam_type)
        self.set("current", beam_settings.beam_current, beam_settings.beam_type)
        self.set("voltage", beam_settings.voltage, beam_settings.beam_type)
        self.set("hfw", beam_settings.hfw, beam_settings.beam_type)
        self.set("resolution", beam_settings.resolution, beam_settings.beam_type)
        self.set("dwell_time", beam_settings.dwell_time, beam_settings.beam_type)
        self.set("stigmation", beam_settings.stigmation, beam_settings.beam_type)
        self.set("shift", beam_settings.shift, beam_settings.beam_type)
        self.set("scan_rotation", beam_settings.scan_rotation, beam_settings.beam_type)
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
            type=self.get("detector_type", beam_type),
            mode=self.get("detector_mode", beam_type),
            brightness=self.get("detector_brightness", beam_type),
            contrast=self.get("detector_contrast", beam_type),
        )
        logging.debug({"msg": "get_detector_settings", "detector_settings": detector_settings.to_dict(), "beam_type": beam_type.name})
        return detector_settings
    
    def set_detector_settings(self, detector_settings: FibsemDetectorSettings, beam_type: BeamType = BeamType.ELECTRON) -> None:
        """Set the detector settings for the specified beam type"""
        logging.debug(f"Setting {beam_type.name} detector settings...")
        self.set("detector_type", detector_settings.type, beam_type)
        self.set("detector_mode", detector_settings.mode, beam_type)
        self.set("detector_brightness", detector_settings.brightness, beam_type)
        self.set("detector_contrast", detector_settings.contrast, beam_type)
        logging.debug({"msg": "set_detector_settings", "detector_settings": detector_settings.to_dict(), "beam_type": beam_type.name})
    
        return

    def get_microscope_state(self, beam_type: BeamType = None) -> MicroscopeState:
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

        return current_microscope_state

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
        if self.is_available("stage"):
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

    
    def apply_configuration(self, system_settings: SystemSettings = None) -> None:
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
    def check_available_values(self, key:str, values, beam_type: BeamType = None) -> bool:
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

    def __init__(self, system_settings: SystemSettings = None):
        if _THERMO_API_AVAILABLE == False:
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
        if not hasattr(self, "system"):
            raise Exception("Please connect to the microscope first")
        
        self.disconnect()
        self.connect_to_microscope(self.system.info.ip_address)

    def disconnect(self):
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

        
    def acquire_image(self, image_settings:ImageSettings) -> FibsemImage:
        """
        Acquire a new image with the specified settings.

            Args:
            image_settings (ImageSettings): The settings for the new image.

        Returns:
            FibsemImage: A new FibsemImage object representing the acquired image.
        """

        # check beam enable
        _check_beam(image_settings.beam_type, self.system)
        
        # get the beam api
        beam = (self.connection.beams.electron_beam 
                if image_settings.beam_type == BeamType.ELECTRON 
                else self.connection.beams.ion_beam)
            
        # set reduced area settings
        if image_settings.reduced_area is not None:
            reduced_area = image_settings.reduced_area.__to_FEI__()
            logging.debug(f"Set reduced are: {reduced_area} for beam type {image_settings.beam_type}")
        else:
            reduced_area = None
            beam.scanning.mode.set_full_frame()

        # set the imaging hfw
        self.set("hfw", image_settings.hfw, image_settings.beam_type)

        logging.info(f"acquiring new {image_settings.beam_type.name} image.")
        self.connection.imaging.set_active_view(image_settings.beam_type.value)
        self.connection.imaging.set_active_device(image_settings.beam_type.value)

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
            beam.scanning.mode.set_full_frame() 

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
        self.connection.imaging.set_active_view(beam_type.value)
        self.connection.imaging.set_active_device(beam_type.value)

        # get the last image
        image = self.connection.imaging.get_image()
        image = AdornedImage(data=image.data.astype(np.uint8), metadata=image.metadata)

        # get the microscope state (for metadata)
        state = self.get_microscope_state(beam_type=beam_type)

        # get the image settings from the image
        image_settings = FibsemImageMetadata.image_settings_from_adorned(
            image, beam_type
        )

        # create the fibsem image
        fibsem_image = FibsemImage.fromAdornedImage(image, image_settings, state)
        
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
    
    def live_imaging(self, image_settings: ImageSettings, image_queue: Queue, stop_event: threading.Event):
        pass
        # self.image_queue = image_queue
        # self.stop_event = stop_event
        # _check_beam(image_settings.beam_type, self.system)
        # logging.info(f"Live imaging: {image_settings.beam_type}")
        # while not self.stop_event.is_set():
        #     image = self.acquire_image(deepcopy(image_settings))
        #     image_queue.put(image)

    def consume_image_queue(self, parent_ui = None, sleep = 0.1):
        pass
        # logging.info("Consuming image queue")

        # while not self.stop_event.is_set():
        #     try:
        #         time.sleep(sleep)
        #         if not self.image_queue.empty():
        #             image = self.image_queue.get(timeout=1)f
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
        """
        Automatically adjust the microscope image contrast for the specified beam type.

        Args:
            beam_type (BeamType) The imaging beam type for which to adjust the contrast.
        """
        _check_beam(beam_type = beam_type, settings = self.system)
        logging.debug(f"Running autocontrast on {beam_type.name}.")
        self.connection.imaging.set_active_view(beam_type.value)
        self.connection.imaging.set_active_device(beam_type.value)
        self.connection.auto_functions.run_auto_cb()
        logging.debug({"msg": "autocontrast", "beam_type": beam_type.name})

    def auto_focus(self, beam_type: BeamType) -> None:
        """Automatically focus the specified beam type.

        Args:
            beam_type (BeamType): The imaging beam type for which to focus.
        """
        _check_beam(beam_type = beam_type, settings = self.system)
        logging.debug(f"Running auto-focus on {beam_type.name}.")
        self.connection.imaging.set_active_view(beam_type.value)  
        self.connection.imaging.set_active_device(beam_type.value)
        self.connection.auto_functions.run_auto_focus()
        logging.debug({"msg": "auto_focus", "beam_type": beam_type.name})       

    def beam_shift(self, dx: float, dy: float, beam_type: BeamType = BeamType.ION) -> None:
        """
        Adjusts the beam shift based on relative values that are provided.
        
        Args:
            self (FibsemMicroscope): Fibsem microscope object
            dx (float): the relative x term
            dy (float): the relative y term
        """
        # TODO: change this to use the set api    
        _check_beam(beam_type, self.system)

        logging.info(f"{beam_type.name} shifting by ({dx}, {dy})")
        if beam_type == BeamType.ELECTRON:
            self.connection.beams.electron_beam.beam_shift.value += (dx, dy)
        else:
            self.connection.beams.ion_beam.beam_shift.value += (dx, dy)
    
        logging.debug({"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name}) 

    
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
        wd = self.get("working_distance", BeamType.ELECTRON)
        
        # convert to autoscript position
        autoscript_position = position.to_autoscript_position(compustage=self.stage_is_compustage)
        if not self.stage_is_compustage:
            autoscript_position.coordinate_system = CoordinateSystem.RAW # TODO: check if this is necessary
        
        logging.info(f"Moving stage to {position}.")
        self.stage.absolute_move(autoscript_position, MoveSettings(rotate_compucentric=True)) # TODO: This needs at least an optional safe move to prevent collision?
                   
        # restore working distance to adjust for microscope compenstation
        self.set("working_distance", wd, BeamType.ELECTRON)
            
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
        if not self.stage_is_compustage:
            thermo_position.coordinate_system = CoordinateSystem.RAW # TODO: check if this is necessary

        # move stage
        self.stage.relative_move(thermo_position)

        logging.debug({"msg": "move_stage_relative", "position": position.to_dict()})

        return self.get_stage_position()

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

        wd = self.get("working_distance", BeamType.ELECTRON)

        scan_rotation = self.get("scan_rotation", beam_type)
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
        
        self.set("working_distance", wd, BeamType.ELECTRON)

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
    ) -> None:
        """ Move the stage vertically to correct coincidence point

        Args:
            dy (float): distance along the y-axis (image coordinates)
            dx (float, optional): distance along the x-axis (image coordinates). Defaults to 0.0.
            static_wd (bool, optional): whether to fix the working distance. Defaults to True.

        """
        # confirm stage is enabled
        _check_stage(self.system)

        # get current working distance, to be restored later
        wd = self.get("working_distance", BeamType.ELECTRON)

        # adjust for scan rotation
        scan_rotation = self.get("scan_rotation", BeamType.ION)
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
            self.set("working_distance", self.system.electron.eucentric_height, BeamType.ELECTRON)
            self.set("working_distance", self.system.ion.eucentric_height, BeamType.ION)
        else:
            self.set("working_distance", wd, BeamType.ELECTRON)

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
        stage_tilt_flat_to_electron = np.deg2rad(self.system.electron.column_tilt)
        stage_tilt_flat_to_ion = np.deg2rad(self.system.ion.column_tilt)

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

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        from fibsem import movement
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        # corrected_pretilt_angle = PRETILT_SIGN * stage_tilt_flat_to_electron
        corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + stage_tilt_flat_to_electron) # electron angle = 0, ion = 52

        # perspective tilt adjustment (difference between perspective view and sample coordinate system)
        if beam_type == BeamType.ELECTRON:
            perspective_tilt_adjustment = -corrected_pretilt_angle
        elif beam_type == BeamType.ION:
            perspective_tilt_adjustment = (
                -corrected_pretilt_angle - stage_tilt_flat_to_ion
            )

        # the amount the sample has to move in the y-axis
        y_sample_move = expected_y  / np.cos(
            stage_tilt + perspective_tilt_adjustment
        )
       
        # the amount the stage has to move in each axis
        y_move = y_sample_move * np.cos(corrected_pretilt_angle)
        z_move = -y_sample_move * np.sin(corrected_pretilt_angle) #TODO: investigate this

        return FibsemStagePosition(x=0, y=y_move, z=z_move)
    
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


    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition
    ) -> None:
        """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system

        """
        # safe movements are not required on the compustage, because it doesn't rotate
        if not self.stage_is_compustage:

            # tilt flat for large rotations to prevent collisions
            self._safe_rotation_movement(stage_position)

            # move to compucentric rotation
            self.move_stage_absolute(FibsemStagePosition(r=stage_position.r, coordinate_system="RAW"))

        logging.debug(f"safe moving to {stage_position}")
        self.move_stage_absolute(stage_position)

        logging.debug("safe movement complete.")

        return

    def project_stable_move(self, 
        dx:float, dy:float, 
        beam_type:BeamType, 
        base_position:FibsemStagePosition) -> FibsemStagePosition:
        
        scan_rotation = self.get("scan_rotation", beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0
        
        # stable-move-projection
        point_yz = self._y_corrected_stage_movement(dy, beam_type)
        dy, dz = point_yz.y, point_yz.z

        # calculate the corrected move to reach that point from base-state?
        _new_position = deepcopy(base_position)
        _new_position.x += dx
        _new_position.y += dy
        _new_position.z += dz

        return _new_position
    
    def insert_manipulator(self, name: str = "PARK"):
        """Insert the manipulator to the specified position"""

        if not self.is_available("manipulator"):
            raise ValueError("Manipulator not available.")
         
        if name not in ["PARK", "EUCENTRIC"]:
            raise ValueError(f"insert position {name} not supported.")
        if VERSION < 4.7:
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

        if VERSION < 4.7:
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
        if VERSION < 4.7:
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
        self.connection.imaging.set_active_view(self.milling_channel.value)  # the ion beam view
        self.connection.imaging.set_active_device(self.milling_channel.value)
        self.connection.patterning.set_default_beam_type(self.milling_channel.value)
        self.connection.patterning.set_default_application_file(mill_settings.application_file)
        self._default_application_file = mill_settings.application_file
        self.connection.patterning.mode = mill_settings.patterning_mode
        self.clear_patterns()  # clear any existing patterns
        self.set("hfw", mill_settings.hfw, self.milling_channel)
        self.set("current", mill_settings.milling_current, self.milling_channel)
        self.set("voltage", mill_settings.milling_voltage, self.milling_channel)

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
            if self.get("voltage", self.milling_channel) != milling_voltage:
                self.set("voltage", milling_voltage, self.milling_channel)
            if self.get("current", self.milling_channel) != milling_current:
                self.set("current", milling_current, self.milling_channel)
        except Exception as e:
            logging.warning(f"Failed to set voltage or current: {e}, voltage={milling_voltage}, current={milling_current}")

        # run milling (asynchronously)
        self.set("active_view", value=self.milling_channel)  # the ion beam view
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

            # update milling progress via signal
            self.milling_progress_signal.emit({"progress": {
                    "state": "update", 
                    "start_time": start_time, 
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
        self.set("current", imaging_current, self.milling_channel)
        self.set("voltage", imaging_voltage, self.milling_channel)
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
        self.connection.patterning.set_default_application_file("Si")
        pattern = self.connection.patterning.create_circle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            outer_diameter=2*pattern_settings.radius,
            inner_diameter = 0,
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

    def draw_annulus(self, pattern_settings: FibsemCircleSettings):

        outer_diameter = 2*pattern_settings.radius
        inner_diameter = outer_diameter - 2*pattern_settings.thickness

        self.connection.patterning.set_default_application_file("Si")
        pattern = self.connection.patterning.create_circle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            outer_diameter=outer_diameter,
            inner_diameter = inner_diameter,
            depth=pattern_settings.depth,
        )
        pattern.application_file = "Si"
        pattern.overlap_r = 0.8
        pattern.overlap_t = 0.8
        self.connection.patterning.set_default_application_file(self._default_application_file)

        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        logging.debug({"msg": "draw_annulus", "pattern_settings": pattern_settings.to_dict()})
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
        self.connection.imaging.set_active_view(BeamType.ELECTRON.value)
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

    # def setup_GIS(self,protocol):

    #     beamtype_value = BeamType.ELECTRON.value if protocol["beam_type"] == "electron" else BeamType.ION.value

    #     _check_sputter(self.system)
    #     self.original_active_view = self.connection.imaging.get_active_view()
    #     self.connection.imaging.set_active_view(beamtype_value)
    #     self.connection.patterning.clear_patterns()
    #     self.connection.patterning.set_default_application_file(protocol["application_file"])
    #     self.connection.patterning.set_default_beam_type(beamtype_value)

    #     gis_line = protocol["gas"]
    #     port = self.gis_lines[gis_line]
    #     if self.GIS_position(gis_line) == "Retracted":
    #         port.insert()

    #     if self.GIS_temp_ready(gis_line) == False:
    #         self.GIS_heat_up(gis_line)


    # def setup_GIS_pattern(self,protocol):

    #     hfw = protocol["hfw"]
    #     line_pattern_length = protocol["length"]
    #     sputter_time = protocol["sputter_time"]

    #     self.connection.beams.electron_beam.horizontal_field_width.value = hfw
    #     pattern = self.connection.patterning.create_line(
    #         -line_pattern_length / 2,  # x_start
    #         +line_pattern_length,  # y_start
    #         +line_pattern_length / 2,  # x_end
    #         +line_pattern_length,  # y_end
    #         2e-6,
    #     )  # milling depth
    #     pattern.time = sputter_time 

    # def run_GIS(self,protocol):

    #     gis_line = protocol["gas"]
    #     sputter_time = protocol["sputter_time"]

    #     blank =protocol["blank_beam"]

    #     print(f"blank beam: {blank}") 

    #     if protocol["blank_beam"]:
    #         self.connection.beams.electron_beam.blank()


    #     if self.connection.patterning.state == "Idle":
    #         logging.info(f"Sputtering with {gis_line} for {sputter_time} seconds...")
    #         self.connection.patterning.start()  # asynchronous patterning
    #         time.sleep(sputter_time + 5)
    #     else:
    #         raise RuntimeError("Can't sputter, patterning state is not ready.")
    #     if self.connection.patterning.state == "Running":
    #         self.connection.patterning.stop()
    #         logging.info(f"Finished sputtering with {gis_line}")
    #     else:
    #         logging.warning(f"Patterning state is {self.connection.patterning.state}")
    #         logging.warning("Consider adjusting the patterning line depth.")


    # def run_Multichem(self,protocol):

    #     _check_sputter(self.system)
    #     self.original_active_view = self.connection.imaging.get_active_view()
    #     self.connection.imaging.set_active_view(BeamType.ELECTRON.value)
    #     self.connection.patterning.clear_patterns()
    #     self.connection.patterning.set_default_application_file(protocol["application_file"])
    #     self.connection.patterning.set_default_beam_type(BeamType.ELECTRON.value)

    #     mc_line = protocol["gas"]
    #     port = self.multichem
    #     if self.multichem_position() == "Retracted":
    #         port.insert()

    #     if self.multichem_temp_ready(mc_line) == False:
    #         self.multichem_heat_up(mc_line)

    #     hfw = protocol["hfw"]
    #     line_pattern_length = protocol["length"]
    #     sputter_time = protocol["sputter_time"]

    #     self.connection.beams.electron_beam.horizontal_field_width.value = hfw
    #     pattern = self.connection.patterning.create_line(
    #         -line_pattern_length / 2,  # x_start
    #         +line_pattern_length,  # y_start
    #         +line_pattern_length / 2,  # x_end
    #         +line_pattern_length,  # y_end
    #         2e-6,
    #     )  # milling depth
    #     # pattern.time = sputter_time + 0.1
    #     pattern.time = sputter_time

    #     self.connection.beams.electron_beam.blank()
    #     # port.line.open()
    #     if self.connection.patterning.state == "Idle":
    #         logging.info(f"Sputtering with {mc_line} for {sputter_time} seconds...")
    #         self.connection.patterning.start()  # asynchronous patterning
    #         time.sleep(sputter_time + 5)
    #     else:
    #         raise RuntimeError("Can't sputter platinum, patterning state is not ready.")
    #     if self.connection.patterning.state == "Running":
    #         self.connection.patterning.stop()
    #     else:
    #         logging.warning(f"Patterning state is {self.connection.patterning.state}")
    #         logging.warning("Consider adjusting the patterning line depth.")
    #     # port.line.close()

    #     self.multichem.line.turn_heater_off(mc_line)





    # def GIS_available_lines(self) -> List[str]:
    #     """
    #     Returns a list of available GIS lines.
    #     Args:
    #         None
    #     Returns:
    #         Dictionary of available GIS lines.
    #     Notes:
    #         None
    #     """
    #     _check_sputter(self.system)

    #     gis_list = self.connection.gas.list_all_gis_ports()

    #     self.gis_lines = {}


    #     for line in gis_list:

    #         gis_port = ThermoGISLine(self.connection.gas.get_gis_port(line),name=line,status="Retracted")

    #         self.gis_lines[line] = gis_port



    #     return gis_list

    # def GIS_available_positions(self) -> List[str]:
    #     """Returns a list of available positions the GIS can move to.
    #     Returns:
    #         List[str]: positions
    #     """

    #     _check_sputter(self.system)

    #     positions = ["Insert", "Retract"]

    #     return positions

    # def GIS_move_to(self,line_name:str,position:str) -> None:

    #     """Moves the GIS to a specified position.
    #     Need to specify the line name and position to move to. Each GIS line is a seperate python object
    #     """

    #     _check_sputter(self.system)

    #     port = self.gis_lines[line_name]

    #     if position == "Insert":
    #         port.insert()
    #     elif position == "Retract":
    #         port.retract()

    # def GIS_position(self,line) -> str:
    #     """
    #     Returns the current position of the GIS line.
    #     """

    #     _check_sputter(self.system)

    #     port = self.gis_lines[line]

    #     return port.status

    # def GIS_heat_up(self,line):

    #     """Heats up the specified GIS line
    #     Does this by turning the heater on for 3 seconds and then off.
    #     If this procedure has been done once, it is considered temp_ready
    #     """

    #     _check_sputter(self.system)

    #     port = self.gis_lines[line]

    #     port.line.turn_heater_on()

    #     time.sleep(3)

    #     port.line.turn_heater_off()

    #     port.temp_ready = True

    # def GIS_temp_ready(self,line) -> bool:

    #     """Returns if the heat up procedure has been done. Internal function, not polled information
    #         from the hardware
    #     Returns:
    #         True if the heat up procedure has been done, False if not
    #     """

    #     _check_sputter(self.system)

    #     port = self.gis_lines[line]

    #     return port.temp_ready


    # def multichem_available_lines(self)-> List[str]:

    #     """
    #     Returns a list of available multichem lines.
    #     List is a str of names of the lines"""

    #     _check_sputter(self.system)

    #     self.multichem = ThermoMultiChemLine(self.connection.gas.get_multichem())

    #     self.mc_lines = self.multichem.line.list_all_gases()

    #     self.mc_lines_temp ={}

    #     for line in self.mc_lines:

    #         self.mc_lines_temp[line] = False

    #     return self.mc_lines

    # def multichem_available_positions(self) -> List[str]:

    #     """Available positions the multichem object can move to
    #     Returns:
    #         _type_: list of str of position names
    #     """

    #     _check_sputter(self.system)

    #     positions_enum = self.multichem.positions

    #     return positions_enum

    # def multichem_move_to(self,position:str):

    #     """
    #     Moves the multichem object to a specified position.
    #     """

    #     logging.info(f"Moving multichem to position: {position}" )

    #     _check_sputter(self.system)

    #     if position == "Retract":
    #         self.multichem.retract()
    #     else:
    #         self.multichem.insert(position=position)

        


    # def multichem_position(self) -> str:

    #     """Current position of the multichem object
    #     Returns:
    #         _type_: str name of the current position
    #     """

    #     _check_sputter(self.system)

    #     return self.multichem.current_position

    # def multichem_heat_up(self,line:str):

    #     """Heat up procedure of the multichem object, if this procedure has been done once,
    #     it is considered temp_ready. Procedure is turning the heater on for 3 seconds and then off
    #     Must specify the line to heat up
    #     """

    #     _check_sputter(self.system)

    #     assert line in self.mc_lines, "Line not available"

    #     self.multichem.line.turn_heater_on(line)

    #     time.sleep(3)

    #     # self.multichem.line.turn_heater_off(line)

    #     self.mc_lines_temp[line] = True

    # def multichem_temp_ready(self,line:str) -> bool:

    #     """Checks if the heat up procedure for a line in the multichem object has been done,
        
    #     Returns:
    #         _type_: Returns true if done, false if not
    #     """

    #     _check_sputter(self.system)

    #     assert line in self.mc_lines, "Line not available"

    #     return self.mc_lines_temp[line]
    
    def get_available_values(self, key: str, beam_type: BeamType = None)-> list:
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
                values = self.connection.beams.electron_beam.beam_current.available_values

            # print(microscope.connection.beams.ion_beam.high_voltage.limits)
            # print(microscope.connection.beams.electron_beam.high_voltage.limits)

            # print(microscope.connection.beams.ion_beam.beam_current.available_values)
            # print(microscope.connection.beams.electron_beam.beam_current.limits)
        
        if key == "detector_type":
            values = self.connection.detector.type.available_values
        
        if key == "detector_mode":
            values = self.connection.detector.mode.available_values
        
        if key == "scan_direction":
            values = ["BottomToTop", 
                "DynamicAllDirections", 
                "DynamicInnerToOuter", 
                "DynamicLeftToRight", 
                "DynamicTopToBottom", 
                "InnerToOuter", 	
                "LeftToRight", 	
                "OuterToInner", 
                "RightToLeft", 	
                "TopToBottom"] # TODO: store elsewhere...
        
        if key == "gis_ports":
            if self.is_available("gis"):
                values = self.connection.gas.list_all_gis_ports()
            elif self.is_availble("multichem"):
                values = self.connection.gas.list_all_multichem_ports()
            else:
                values = []
                        
        logging.debug({"msg": "get_available_values", "key": key, "values": values})

        return values


    def _get(self, key: str, beam_type: BeamType = None) -> Union[float, str, list, None]:
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
            self.connection.imaging.set_active_view(beam_type.value)
            self.connection.imaging.set_active_device(beam_type.value)

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

    def _set(self, key: str, value, beam_type: BeamType = None) -> None:
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
            beam.beam_shift.value = ThermoPoint(-value.x, value.y)
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
            # set beam active view and device
            self.connection.imaging.set_active_view(beam_type.value)
            self.connection.imaging.set_active_device(beam_type.value)

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
    
    def check_available_values(self, key:str, values: list, beam_type: BeamType = None) -> bool:

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
    


########################
SIMULATOR_KNOWN_UNKNOWN_KEYS = ["preset"]

class DemoMicroscope(FibsemMicroscope):

    def __init__(self, system_settings: SystemSettings):            

        # hack, do this properly @patrick
        from dataclasses import dataclass

        @dataclass
        class DemoMicroscopeClient:
            connected: bool = False

            def connect(self, ip_address: str, port: int = 8080):
                logging.debug(f"Connecting to microscope at {ip_address}:{port}")
                self.connected = True
                logging.debug(f"Connected to microscope at {ip_address}:{port}")

            def disconnect(self):
                self.connected = False

        self.connection = DemoMicroscopeClient()

        @dataclass
        class BeamSystem:
            on: bool
            blanked: bool
            beam: BeamSettings
            detector: FibsemDetectorSettings    

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
            patterns: List[FibsemPatternSettings] = None
            patterning_mode: str = "Serial"
            default_beam_type: BeamType = BeamType.ION
            default_application_file: str = "Si"

        @dataclass
        class ImagingSystem:
            active_view: int = BeamType.ELECTRON.value
            active_device: int = BeamType.ELECTRON.value

        # initialise system
        self.connection = DemoMicroscopeClient()
        self.system = system_settings    

        self.chamber = ChamberSystem(state="Pumped", 
                                        pressure=1e-6)
        self.stage_system = StageSystem(is_homed = True, 
                                        is_linked= True,
                                        position=FibsemStagePosition(x=0, y=0, z=0, r=0, t=0, coordinate_system="RAW"))
        
        self.manipulator_system = ManipulatorSystem(
                            inserted= False,
                            position = FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0, coordinate_system="RAW")
            )
                
        self.gis_system = GasInjectionSystem(gas="Pt dep")

        self.electron_system = BeamSystem(
            on=False,
            blanked=True,
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
            )
        )
            
        self.ion_system = BeamSystem(
            on=False,
            blanked=True,
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
            )
        )
        self.stage_is_compustage: bool = False
        self.milling_system = MillingSystem(patterns=[])
        self.imaging_system = ImagingSystem()
            
        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

        self._last_imaging_settings: ImageSettings = ImageSettings()
        self.milling_channel: BeamType.ION = BeamType.ION
        # logging
        logging.debug({"msg": "create_microscope_client", "system_settings": system_settings.to_dict()})


    def connect_to_microscope(self, ip_address: str, port: int = 8080):
        
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

    def disconnect(self):
        self.connection.disconnect()
        logging.info("Disconnected from Demo Microscope")

    def acquire_image(self, image_settings: ImageSettings) -> FibsemImage:
        _check_beam(image_settings.beam_type, self.system)
        vfw = image_settings.hfw * image_settings.resolution[1] / image_settings.resolution[0]
        pixelsize = Point(image_settings.hfw / image_settings.resolution[0], 
                          vfw / image_settings.resolution[1])
        
        logging.info(f"acquiring new {image_settings.beam_type.name} image.")
        image = FibsemImage(
            data=np.random.randint(low=0, high=256, 
                size=(image_settings.resolution[1],image_settings.resolution[0]), 
                dtype=np.uint8),
            metadata=FibsemImageMetadata(image_settings=image_settings, 
                                        pixel_size=pixelsize,
                                        microscope_state=self.get_microscope_state(beam_type=image_settings.beam_type),
                                        system=self.system
                                        )
        )

        image.metadata.user = self.user
        image.metadata.experiment = self.experiment
        image.metadata.system = self.system

        if image_settings.beam_type is BeamType.ELECTRON:
            self._eb_image = image
        else:
            self._ib_image = image

        # store last imaging settings
        self._last_imaging_settings = image_settings

        logging.debug({"msg": "acquire_image", "metadata": image.metadata.to_dict()})

        return image

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        _check_beam(beam_type, self.system)

        image = self._eb_image if beam_type is BeamType.ELECTRON else self._ib_image
        logging.debug({"msg": "last_image", "beam_type": beam_type.name, "metadata": image.metadata.to_dict()})
        return image
    
    def acquire_chamber_image(self) -> FibsemImage:
        """Acquire an image of the chamber inside."""
        image = FibsemImage(
            data=np.random.randint(low=0, high=256, 
                size=(1024,1536), 
                dtype=np.uint8),
                metadata=None)
        logging.debug({"msg": "acquire_chamber_image"})
        return image

    def live_imaging(self, image_settings: ImageSettings, image_queue: Queue, stop_event: threading.Event):
        pass
        # self.image_queue = image_queue
        # self.stop_event = stop_event
        # _check_beam(image_settings.beam_type, self.system)
        # logging.info(f"Live imaging: {image_settings.beam_type}")
        # while not stop_event.is_set():
        #     image = self.acquire_image(image_settings)
        #     image_queue.put(image)
        #     self.sleep_time = image_settings.dwell_time*image_settings.resolution[0]*image_settings.resolution[1]
        #     time.sleep(self.sleep_time)

    def consume_image_queue(self, parent_ui = None, sleep = 0.1):
        pass
        # logging.info("Consuming image queue")

        # try:
        #     while not self.stop_event.is_set():
        #         image = self.image_queue.get(timeout=1)
        #         if image.metadata.image_settings.save:
        #             image.metadata.image_settings.filename = f"{image.metadata.image_settings.filename}_{utils.current_timestamp()}"
        #             filename = os.path.join(image.metadata.image_settings.path, image.metadata.image_settings.filename)
        #             image.save(path=filename)
        #             logging.info(f"Saved image to {filename}")

        #         logging.info(f"Image: {image.data.shape}")
        #         logging.info("-" * 50)

        #         if parent_ui is not None:
        #                 parent_ui.live_imaging_signal.emit({"image": image})
        #         time.sleep(sleep)
        # except KeyboardInterrupt:
        #     self.stop_event
        #     logging.info("Keyboard interrupt, stopping live imaging")
        # except Exception as e:
        #     self.stop_event.set()
        #     import traceback
        #     logging.error(traceback.format_exc())
        # finally:
        #     logging.info("Stopped thread image consumption")
    
    def autocontrast(self, beam_type: BeamType) -> None:
        _check_beam(beam_type, self.system)
        logging.debug({"msg": "autocontrast", "beam_type": beam_type.name})

    def auto_focus(self, beam_type: BeamType) -> None:
        _check_beam(beam_type, self.system)
        logging.debug({"msg": "auto_focus", "beam_type": beam_type.name})
        
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:
        _check_beam(beam_type, self.system)

        logging.debug({"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name})         

        if beam_type == BeamType.ELECTRON:
            self.electron_system.beam.shift += Point(float(dx), float(dy))
        elif beam_type == BeamType.ION:
            self.ion_system.beam.shift += Point(float(dx), float(dy))

   
    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the specified position using safe strategy"""
        self.move_stage_absolute(stage_position)

    def project_stable_move(self, dx:float, dy:float, beam_type:BeamType, base_position:FibsemStagePosition) -> FibsemStagePosition:

        scan_rotation = self.get("scan_rotation", beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0
        
        # stable-move-projection
        point_yz = self._y_corrected_stage_movement(dy, beam_type)
        dy, dz = point_yz.y, point_yz.z

        # calculate the corrected move to reach that point from base-state?
        _new_position = deepcopy(base_position)
        _new_position.x += dx
        _new_position.y += dy
        _new_position.z += dz

        return _new_position

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
        _check_stage_movement(self.system, FibsemStagePosition(x=dx, y=dy))

        wd = self.get("working_distance", BeamType.ELECTRON) 

        scan_rotation = self.get("scan_rotation", beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0
        
        # calculate stable movement
        yz_move = self._y_corrected_stage_movement(
            expected_y=dy,
            beam_type=beam_type,
        )
        stage_position = FibsemStagePosition(
            x=dx, y=yz_move.y, z=yz_move.z,
            r=0, t=0, coordinate_system="RAW",
        )

        # move stage        
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        if static_wd:
            wd = self.system.electron.eucentric_height
        
        self.set("working_distance", wd, BeamType.ELECTRON)

        # logging
        logging.debug({"msg": "stable_move", "dx": dx, "dy": dy, 
                "beam_type": beam_type.name, "static_wd": static_wd, 
                "working_distance": wd, "scan_rotation": scan_rotation,
                "position": stage_position.to_dict()})

        return stage_position


    def vertical_move(self, dy: float, dx:float = 0.0, static_wd: bool=True) -> FibsemStagePosition:
        """Move the stage vertically by the specified amount."""
        # confirm stage is enabled
        _check_stage(self.system)

        # get current working distance, to be restored later
        wd = self.get("working_distance", BeamType.ELECTRON)

        # adjust for scan rotation
        scan_rotation = self.get("scan_rotation", BeamType.ION)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # TODO: implement perspective correction
        PERSPECTIVE_CORRECTION = 0.9
        z_move = dy / np.cos(np.deg2rad(90 - self.system.ion.column_tilt)) * PERSPECTIVE_CORRECTION  # TODO: MAGIC NUMBER, 90 - fib tilt

        # TODO: do this manually without autoscript in raw coordinates
        stage_position = FibsemStagePosition(
            x=dx,
            z=z_move, 
            coordinate_system="Specimen"
        )

        # move stage
        self.move_stage_relative(stage_position)

        if static_wd:
            self.set("working_distance", self.system.electron.eucentric_height, BeamType.ELECTRON)
            self.set("working_distance", self.system.ion.eucentric_height, BeamType.ION)
        else:
            self.set("working_distance", wd, BeamType.ELECTRON)
    
        # logging
        logging.debug({"msg": "vertical_move", "dy": dy, "dx": dx, 
                "static_wd": static_wd, "working_distance": wd, 
                "scan_rotation": scan_rotation, 
                "position": stage_position.to_dict()})

        return self.get_stage_position()

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
        
        # all angles in radians
        stage_tilt_flat_to_electron = np.deg2rad(self.system.electron.column_tilt)
        stage_tilt_flat_to_ion = np.deg2rad(self.system.ion.column_tilt)

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

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        from fibsem import movement
        if movement.rotation_angle_is_smaller(stage_rotation, stage_rotation_flat_to_eb, atol=5):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        # corrected_pretilt_angle = PRETILT_SIGN * stage_tilt_flat_to_electron
        corrected_pretilt_angle = PRETILT_SIGN * (stage_pretilt + stage_tilt_flat_to_electron) # electron angle = 0, ion = 52

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
        z_move = -y_sample_move * np.sin(corrected_pretilt_angle) #TODO: investigate this

        return FibsemStagePosition(x=0, y=y_move, z=z_move)

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
        _check_beam(self.milling_channel, self.system)
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
    
    def draw_annulus(self, pattern_settings: FibsemCircleSettings) -> None:
        logging.debug({"msg": "draw_annulus", "pattern_settings": pattern_settings.to_dict()})
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

    def get_available_values(self, key: str, beam_type: BeamType = None) -> List[float]:
        
        values = []
        if key == "current":
            _check_beam(beam_type, self.system)
            if beam_type == BeamType.ELECTRON:
                values = [1.0e-12]
            if beam_type == BeamType.ION:
                values = [20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.6e-9, 28.0e-9, 120e-9]
        

        if key == "application_file":
            values = ["Si", "Si-multipass", "Si-ccs", "autolamella", "cryo_Pt_dep"]

        if key == "detector_type":
            values = ["ETD", "TLD", "EDS"]
        if key == "detector_mode":
            values = ["SecondaryElectrons", "BackscatteredElectrons", "EDS"]
        
        if key == "scan_direction":
            values = ["BottomToTop", 
                "LeftToRight", 	
                "RightToLeft", 	
                "TopToBottom"]
            
        if key == "plasma_gas":
            values = ["Oxygen", "Argon", "Nitrogen", "Xenon"]
         
        if key == "gis_ports":
            values = ["Pt Dep", "Pt Dep Cryo2"]
                
        return values

    def _get(self, key, beam_type: BeamType = None) -> Union[float, int, bool, str, list]:
        """Get a value from the microscope."""
        # get beam
        if beam_type is not None:
            beam_system = self.electron_system if beam_type is BeamType.ELECTRON else self.ion_system
            beam, detector = beam_system.beam, beam_system.detector
            _check_beam(beam_type, self.system)
        
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

        # imaging system
        if key == "active_view":
            self.imaging_system.active_view = value.value
            return
        if key == "active_device":
            self.imaging_system.active_device = value.value

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


######################################## Helper functions ########################################


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