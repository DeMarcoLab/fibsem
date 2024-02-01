import copy
import datetime
import logging
import sys
from napari.qt.threading import thread_worker
import os

# for easier usage
import fibsem.constants as constants

_TESCAN_API_AVAILABLE = False
_THERMO_API_AVAILABLE = False

# DEVELOPMENT
_OVERWRITE_AUTOSCRIPT_VERSION = False
if os.environ.get("COMPUTERNAME", "hostname") == "MU00190108":
    _OVERWRITE_AUTOSCRIPT_VERSION = True
    print("Overwriting autoscript version to 4.7, for Monash dev install")

try:
    import re

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
except:
    logging.debug("Automation (TESCAN) not installed.")

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
        
    from autoscript_sdb_microscope_client.structures import (
    BitmapPatternDefinition)
    from autoscript_sdb_microscope_client._dynamic_object_proxies import (
        CleaningCrossSectionPattern, RectanglePattern, LinePattern, CirclePattern, RegularCrossSectionPattern )
    from autoscript_sdb_microscope_client.enumerations import (
        CoordinateSystem, ManipulatorCoordinateSystem,ManipulatorState,
        ManipulatorSavedPosition, PatterningState,MultiChemInsertPosition)
    from autoscript_sdb_microscope_client.structures import (
        GrabFrameSettings, ManipulatorPosition, MoveSettings, StagePosition)
    from autoscript_sdb_microscope_client.structures import AdornedImage

    from autoscript_sdb_microscope_client.enumerations import ManipulatorCoordinateSystem, ManipulatorSavedPosition ,MultiChemInsertPosition
    _THERMO_API_AVAILABLE = True 
except Exception as e:
    logging.debug("Autoscript (ThermoFisher) not installed.")
    if isinstance(e, NameError):
        raise e 

import sys

from fibsem.structures import (BeamSettings, BeamSystemSettings, BeamType,
                               FibsemImage, FibsemImageMetadata,
                               FibsemManipulatorPosition,
                               FibsemMillingSettings, FibsemRectangle,
                               FibsemStagePosition,
                               ImageSettings, SystemSettings, 
                               SystemInfo,
                               MicroscopeState, Point, FibsemDetectorSettings,
                               ThermoGISLine,ThermoMultiChemLine,
                            FibsemUser, FibsemExperiment, 
                            FibsemPatternSettings, FibsemRectangleSettings, 
                            FibsemCircleSettings, FibsemLineSettings, FibsemBitmapSettings)
import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from queue import Queue
from typing import Union
import numpy as np
    



class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""

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
    @thread_worker
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

        # updated safe rotation move
        logging.info(f"moving flat to {beam_type.name}")
        stage_position = FibsemStagePosition(r=rotation, t=tilt)

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
    def run_milling_drift_corrected(self, milling_current: float,  
        image_settings: ImageSettings, 
        ref_image: FibsemImage, 
        reduced_area: FibsemRectangle = None,
        asynch: bool = False
        ):
        pass

    @abstractmethod
    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        pass
    @abstractmethod
    def estimate_milling_time(self,patterns) -> float:
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
    def get_scan_directions(self) -> list:
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

    def get_microscope_state(self) -> MicroscopeState:
        """Get the current microscope state."""

        # default values
        electron_beam, electron_detector = None, None
        ion_beam, ion_detector = None, None
        stage_position = None

        # get the state of the electron beam
        if self.is_available("electron_beam"):
            electron_beam = self.get_beam_settings(beam_type=BeamType.ELECTRON)
            electron_detector = self.get_detector_settings(beam_type=BeamType.ELECTRON)
 
        # get the state of the ion beam        
        if self.is_available("ion_beam"):
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
            self.set_beam_settings(microscope_state.electron_beam)
            self.set_detector_settings(microscope_state.electron_detector, BeamType.ELECTRON)
        if self.is_available("ion_beam"):
            self.set_beam_settings(microscope_state.ion_beam)
            self.set_detector_settings(microscope_state.ion_detector, BeamType.ION)
        if self.is_available("stage"):
            self.safe_absolute_stage_movement(microscope_state.stage_position)
            
        logging.debug({"msg": "set_microscope_state", "state": microscope_state.to_dict()})

        return             
        
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
        
        logging.info(f"Applying Microscope Configuration...")
        
        if system_settings is None:
            system_settings = self.system
            logging.info(f"Using current system settings.")
        
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
        logging.info(f"Microscope configuration applied.")
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

        def run_milling_drift_corrected(self, milling_current: float, image_settings: ImageSettings, ref_image: FibsemImage, reduced_area: FibsemRectangle = None, asynch: bool = False):
            Run ion beam milling using the specified milling current, and correct for drift using the provided reference image.
        
        finish_milling(self, imaging_current: float):
            Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        get_scan_directions(self) -> list:
            Get the available scan directions for milling.

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

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

        # logging
        logging.debug({"msg": "create_microscope_client", "system_settings": system_settings.to_dict()})


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
            logging.warning(f"----------- REDUCED AREA: {reduced_area} -----------")
            logging.warning(f"----------- BEAM_TYPE {image_settings.beam_type} -----------")
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
        state = self.get_microscope_state()

        fibsem_image = FibsemImage.fromAdornedImage(
            copy.deepcopy(image), 
            copy.deepcopy(image_settings), 
            copy.deepcopy(state),
        )

        # set additional metadata
        fibsem_image.metadata.user = self.user
        fibsem_image.metadata.experiment = self.experiment
        fibsem_image.metadata.system = self.system

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
        state = self.get_microscope_state()

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
            self.image_queue = image_queue
            self.stop_event = stop_event
            _check_beam(image_settings.beam_type, self.system)
            logging.info(f"Live imaging: {image_settings.beam_type}")
            while not self.stop_event.is_set():
                image = self.acquire_image(deepcopy(image_settings))
                image_queue.put(image)



    @thread_worker
    def consume_image_queue(self, parent_ui = None, sleep = 0.1):

        logging.info("Consuming image queue")

        while not self.stop_event.is_set():
            try:
                time.sleep(sleep)
                if not self.image_queue.empty():
                    image = self.image_queue.get(timeout=1)
                    if image.metadata.image_settings.save:
                        image.metadata.image_settings.filename = f"{image.metadata.image_settings.filename}_{utils.current_timestamp()}"
                        filename = os.path.join(image.metadata.image_settings.path, image.metadata.image_settings.filename)
                        image.save(path=filename)
                        logging.info(f"Saved image to {filename}")

                    logging.info(f"Image: {image.data.shape}")
                    logging.info(f"-" * 50)

                    if parent_ui is not None:
                            parent_ui.live_imaging_signal.emit({"image": image})


            except KeyboardInterrupt:
                self.stop_event
                logging.info("Keyboard interrupt, stopping live imaging")
            except Exception as e:
                self.stop_event.set()
                import traceback
                logging.error(traceback.format_exc())

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
        logging.debug({f"msg": "autocontrast", "beam_type": beam_type.name})

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
        autoscript_position = position.to_autoscript_position()
        autoscript_position.coordinate_system = CoordinateSystem.RAW
        
        logging.info(f"Moving stage to {position}.")
        self.connection.specimen.stage.absolute_move(autoscript_position, MoveSettings(rotate_compucentric=True)) # TODO: This needs at least an optional safe move to prevent collision?
                   
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
        thermo_position = position.to_autoscript_position()
        thermo_position.coordinate_system = CoordinateSystem.RAW

        # move stage
        self.connection.specimen.stage.relative_move(thermo_position)

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
        move_settings = MoveSettings(link_z_y=True)
        autoscript_position = stage_position.to_autoscript_position() 
        self.connection.specimen.stage.relative_move(autoscript_position, move_settings)

        # restore working distance to adjust for microscope compenstation
        if static_wd:
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
            logging.info(f"tilting to flat for large rotation.")

        return


    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition
    ) -> None:
        """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system

        """

        # tilt flat for large rotations to prevent collisions
        self._safe_rotation_movement(stage_position)

        # move to compucentric rotation
        self.move_stage_absolute(FibsemStagePosition(r=stage_position.r))

        logging.debug(f"safe moving to {stage_position}")
        self.move_stage_absolute(stage_position)

        logging.debug(f"safe movement complete.")

        return

    def project_stable_move(self, 
        dx:float, dy:float, 
        beam_type:BeamType, 
        base_position:FibsemStagePosition) -> FibsemStagePosition:

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
        logging.info(f"inserting manipulator to {saved_position}: {insert_position}.")
        self.connection.specimen.manipulator.insert(insert_position)
        logging.info(f"insert manipulator complete.")

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
        logging.info(f"retracting needle...")
        needle.retract()
        logging.info(f"retract needle complete")
    
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
                named_position, ManipulatorCoordinateSystem.STAGE
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
        _check_beam(BeamType.ION, self.system)
        self.connection.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
        self.connection.imaging.set_active_device(BeamType.ION.value)
        self.connection.patterning.set_default_beam_type(
            BeamType.ION.value
        )  # ion beam default
        self.connection.patterning.set_default_application_file(mill_settings.application_file)
        self.connection.patterning.mode = mill_settings.patterning_mode
        self.connection.patterning.clear_patterns()  # clear any existing patterns
        self.connection.beams.ion_beam.horizontal_field_width.value = mill_settings.hfw

        self.set("current", mill_settings.milling_current, BeamType.ION)
        self.set("voltage", mill_settings.milling_voltage, BeamType.ION)
    
        logging.debug({"msg": "setup_milling", "mill_settings": mill_settings.to_dict()})

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False):
        """
        Run ion beam milling using the specified milling current.

        Args:
            milling_current (float): The current to use for milling in amps.
            asynch (bool, optional): If True, the milling will be run asynchronously. 
                                     Defaults to False, in which case it will run synchronously.

        """
        if not self.is_available("ion_beam"):
            raise ValueError("Ion beam not available.")
        
        # change to milling current, voltage
        if self.get("voltage", BeamType.ION) != milling_voltage:
            self.set("voltage", milling_voltage, BeamType.ION)
        if self.get("current", BeamType.ION) != milling_current:
            self.set("current", milling_current, BeamType.ION)

        # run milling (asynchronously)
        self.connection.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
        logging.info(f"running ion beam milling now... asynchronous={asynch}")
        if asynch:
            self.connection.patterning.start()
        else:
            self.connection.patterning.run()
            self.connection.patterning.clear_patterns()
        # NOTE: Make tescan logs the same??
                                    
        logging.debug({"msg": "run_milling", "milling_current": milling_current, "milling_voltage": milling_voltage, "asynch": asynch})

    def run_milling_drift_corrected(self, milling_current: float, 
                                    milling_voltage: float,  
                                    image_settings: ImageSettings, 
                                    ref_image: FibsemImage, 
                                    reduced_area: FibsemRectangle = None,
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
        # change to milling current
        self.connection.imaging.set_active_view(BeamType.ION.value)  # the ion beam view
        if self.get("voltage", BeamType.ION) != milling_voltage:
            self.set("voltage", milling_voltage, BeamType.ION)
        if self.get("current", BeamType.ION) != milling_current:
            self.set("current", milling_current, BeamType.ION)


        # run milling (asynchronously)
        logging.info(f"running ion beam milling now.")

        self.connection.patterning.start()
        # NOTE: Make tescan logs the same??
        from fibsem import alignment
        while self.connection.patterning.state == PatterningState.IDLE: # giving time to start 
            time.sleep(0.5)
        while self.connection.patterning.state == PatterningState.RUNNING:

            self.connection.patterning.pause()
            logging.info("Drift correction")
            if reduced_area is not None:
                reduced_area = reduced_area.__to_FEI__()
            image_settings.beam_type = BeamType.ION
            alignment.beam_shift_alignment(microscope = self, image_settings= image_settings, ref_image=ref_image, reduced_area=reduced_area)
            time.sleep(1) # need delay to switch back to patterning mode 
            self.connection.imaging.set_active_view(BeamType.ION.value)
            if self.connection.patterning.state == PatterningState.PAUSED: # check if finished 
                self.connection.patterning.resume()
                time.sleep(5)
            print(self.connection.patterning.state)


    def finish_milling(self, imaging_current: float, imaging_voltage: float):
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        """
        _check_beam(BeamType.ION, self.system)
        self.connection.patterning.clear_patterns()
        self.set("current", imaging_current, BeamType.ION)
        self.set("voltage", imaging_voltage, BeamType.ION)
        self.connection.patterning.mode = "Serial"

        logging.debug({"msg": "finish_milling", "imaging_current": imaging_current, "imaging_voltage": imaging_voltage})

    def estimate_milling_time(self, patterns: list ) -> float:
        """Calculates the estimated milling time for a list of patterns."""

        total_time = 0
        for pattern in patterns:
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
        create_pattern_function = patterning_api.create_cleaning_cross_section if pattern_settings.cleaning_cross_section else patterning_api.create_rectangle

        # patterning_api = self.connection.patterning
        # from fibsem.structures import CrossSectionPattern
        # if isinstance(pattern_settings.cross_section, CrossSectionPattern.RegularCrossSection):
        #     create_pattern_function = patterning_api.create_regular_cross_section
        # elif isinstance(pattern_settings.cross_section, CrossSectionPattern.CleaningCrossSection):
        #     create_pattern_function = patterning_api.create_cleaning_cross_section
        # else:
        #     patterning_api.create_rectangle
        
        # if pattern_settings.cleaning_cross_section:    
        #     create_pattern_function = patterning_api.create_cleaning_cross_section

        
            
        # create pattern
        pattern = create_pattern_function(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )

        # set pattern rotation
        pattern.rotation = pattern_settings.rotation

        # set scan direction
        available_scan_directions = self.get_scan_directions()
        if pattern_settings.scan_direction in available_scan_directions:
            pattern.scan_direction = pattern_settings.scan_direction
        else:
            pattern.scan_direction = "TopToBottom"
            logging.warning(f"Scan direction {pattern_settings.scan_direction} not supported. Using TopToBottom instead.")
            logging.warning(f"Supported scan directions are: {available_scan_directions}")        
        
        # set passes
        if pattern_settings.passes: # not zero
            pattern.dwell_time = pattern.dwell_time * (pattern.pass_count / pattern_settings.passes)
            
            # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
            # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
            # NB: the current must be set before doing this, otherwise it will be out of range
        
        # if pattern_settings.passes: # not zero
        #     if isinstance(pattern, RegularCrossSectionPattern):
        #         pattern.multi_scan_pass_count = 1
        #     else:
        #         pattern.dwell_time = pattern.dwell_time * (pattern.pass_count / pattern_settings.passes)
                
        #         # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
        #         # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
        #         # NB: the current must be set before doing this, otherwise it will be out of range

        logging.debug({"msg": "draw_rectangle", "pattern_settings": pattern_settings.to_dict()})

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
        pattern = self.connection.patterning.create_circle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            outer_diameter=2*pattern_settings.radius,
            inner_diameter = 0,
            depth=pattern_settings.depth,
        )

        logging.debug({"msg": "draw_circle", "pattern_settings": pattern_settings.to_dict()})

        return pattern

    def draw_annulus(self, pattern_settings: FibsemCircleSettings):

        outer_diameter = 2*pattern_settings.radius
        inner_diameter = outer_diameter - 2*pattern_settings.thickness

        pattern = self.connection.patterning.create_circle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            outer_diameter=outer_diameter,
            inner_diameter = inner_diameter,
            depth=pattern_settings.depth,
        )

        logging.debug({"msg": "draw_annulus", "pattern_settings": pattern_settings.to_dict()})

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

        return pattern

    def get_scan_directions(self) -> list:
        """
        Returns the available scan directions of the microscope.

        Returns:
            list: The scan direction of the microscope.

        Raises:
            None
        """
        list = ["BottomToTop", 
                "DynamicAllDirections", 
                "DynamicInnerToOuter", 
                "DynamicLeftToRight", 
                "DynamicTopToBottom", 
                "InnerToOuter", 	
                "LeftToRight", 	
                "OuterToInner", 
                "RightToLeft", 	
                "TopToBottom"]
        
        logging.debug({"msg": "get_scan_directions", "scan_directions": list})

        return list 


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





    # def GIS_available_lines(self) -> list[str]:
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

    # def GIS_available_positions(self) -> list[str]:
    #     """Returns a list of available positions the GIS can move to.
    #     Returns:
    #         list[str]: positions
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


    # def multichem_available_lines(self)-> list[str]:

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

    # def multichem_available_positions(self) -> list[str]:

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
                "TopToBottom"]

        return values


    def _get(self, key: str, beam_type: BeamType = None) -> Union[float, str, list, None]:
        """Get a property of the microscope."""
        # TODO: make the list of get and set keys available to the user
        if beam_type is not None:
            beam = self.connection.beams.electron_beam if beam_type == BeamType.ELECTRON else self.connection.beams.ion_beam
            _check_beam(beam_type, self.system)

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
            self.connection.specimen.stage.set_default_coordinate_system(
            CoordinateSystem.RAW
            )
            stage_position = self.connection.specimen.stage.current_position  
            self.connection.specimen.stage.set_default_coordinate_system(
                CoordinateSystem.SPECIMEN
            )
            stage_position = FibsemStagePosition.from_autoscript_position(stage_position)
            return stage_position
        
        if key == "stage_homed":
            _check_stage(self.system)
            return self.connection.specimen.stage.is_homed
        if key == "stage_linked":
            _check_stage(self.system)
            return self.connection.specimen.stage.is_linked

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
        
        # beam properties
        if key == "working_distance":
            beam.working_distance.value = value
            if beam_type is BeamType.ELECTRON:
                self.set("stage_link", True)  # link the specimen stage for electron
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
            value = np.clip(value, limits.min, limits.max)
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
                    logging.debug(f"Plasma gas cannot be set on this microscope.")
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
            logging.info(f"Homing stage...")
            self.connection.specimen.stage.home()
            logging.info(f"Stage homed.")
            return
        
        if key == "stage_link":
            _check_stage(self.system)
            logging.info(f"Linking stage...")
            self.connection.specimen.stage.link() if value else self.connection.specimen.stage.unlink()
            logging.info(f"Stage {'linked' if value else 'unlinked'}.")    
            return
        
        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info(f"Pumping chamber...")
                self.connection.vacuum.pump()
                logging.info(f"Chamber pumped.") 
                return
            else:
                logging.warning(f"Invalid value for pump_chamber: {value}.")
                return
            
        if key == "vent_chamber":
            if value:
                logging.info(f"Venting chamber...")
                self.connection.vacuum.vent()
                logging.info(f"Chamber vented.") 
                return
            else:
                logging.warning(f"Invalid value for vent_chamber: {value}.")
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

        get_scan_directions(self) -> list:
            Get the available scan directions for milling.    

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
        self.ion_detector_active = detectors[0]
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
        self.image_queue = image_queue
        self.stop_event = stop_event
        _check_beam(image_settings.beam_type, self.system)
        logging.info(f"Live imaging: {image_settings.beam_type}")
        while not self.stop_event.is_set():
            image = self.acquire_image(deepcopy(image_settings))
            image_queue.put(image)



    @thread_worker
    def consume_image_queue(self, parent_ui = None, sleep = 0.1):

        logging.info("Consuming image queue")

        while not self.stop_event.is_set():
            try:
                time.sleep(sleep)
                if not self.image_queue.empty():
                    image = self.image_queue.get(timeout=1)
                    if image.metadata.image_settings.save:
                        image.metadata.image_settings.filename = f"{image.metadata.image_settings.filename}_{utils.current_timestamp()}"                        
                        filename = os.path.join(image.metadata.image_settings.path, image.metadata.image_settings.filename)
                        image.save(path=filename)
                        logging.info(f"Saved image to {filename}")

                    logging.info(f"Image: {image.data.shape}")
                    logging.info(f"-" * 50)

                    if parent_ui is not None:
                            parent_ui.live_imaging_signal.emit({"image": image})


            except KeyboardInterrupt:
                self.stop_event
                logging.info("Keyboard interrupt, stopping live imaging")
            except Exception as e:
                self.stop_event.set()
                import traceback
                logging.error(traceback.format_exc())
        
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
        if self.system.stage_enabled is False:
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
        # self.connection.specimen.stage.link() # TODO how to link for TESCAN?

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
        logging.info(f"running ion beam milling now...")
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
        logging.info(f"running ion beam milling now...")
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
        except:
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
            self.layer.addAnnulusPolish(
                CenterX=pattern_settings.centre_x,
                CenterY=pattern_settings.centre_y,
                RadiusA=pattern_settings.radius,
                RadiusB=0,
                Depth=pattern_settings.depth,
                DepthUnit='m',
            )

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
    
    def get_scan_directions(self):
        
        list = ["Flyback", "RLE", "SpiralInsideOut", "SpiralOutsideIn", "ZigZag"]
        return list

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

        except:
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


    # def GIS_available_lines(self) -> list[str]:
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

    # def GIS_available_positions(self) -> list[str]:

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

        logging.info(f"restoring microscope state...")

        # restore electron beam
        _check_beam(BeamType.ELECTRON, self.system)
        logging.info(f"restoring electron beam settings...")
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
        logging.info(f"microscope state restored")
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
        if beam_type is None:
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
        if beam_type is None:
            beam = self.connection.SEM if beam_type == BeamType.ELECTRON else self.connection.FIB
            _check_beam(beam_type, self.system)

        if key == "working_distance":
            if beam_type == BeamType.ELECTRON:
                beam.Optics.SetWD(value * constants.METRE_TO_MILLIMETRE)
                logging.info(f"Electron beam working distance set to {value} m.")
            else: 
                logging.info(f"Setting working distance for ion beam is not supported by Tescan API.")
            return
        if key == "current":
            if beam_type == BeamType.ELECTRON:
                beam.Beam.SetCurrent(value * constants.SI_TO_PICO)
                logging.info(f"Electron beam current set to {value} A.")
            else: 
                logging.warning(f"Setting current for ion beam is not supported by Tescan API, please use the native microscope interface.")
            return
        if key == "voltage":
            if beam_type == BeamType.ELECTRON:
                beam.Beam.SetVoltage(value)
                logging.info(f"Electron beam voltage set to {value} V.")
            else:
                logging.warning(f"Setting voltage for ion beam is not supported by Tescan API, please use the native microscope interface.")
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
        logging.warning(f"No homing available, please use native UI.")
        return

########################
class DemoMicroscope(FibsemMicroscope):

    def __init__(self, system_settings: SystemSettings):            
        
        
        # hack, do this properly @patrick
        from dataclasses import dataclass

        @dataclass
        class DemoMicroscopeClient:
            self.connected: bool = False

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
            
        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

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
        logging.info(f"Disconnected from Demo Microscope")

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
                                        microscope_state=self.get_microscope_state(),
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
        self.image_queue = image_queue
        self.stop_event = stop_event
        _check_beam(image_settings.beam_type, self.system)
        logging.info(f"Live imaging: {image_settings.beam_type}")
        while not stop_event.is_set():
            image = self.acquire_image(image_settings)
            image_queue.put(image)
            self.sleep_time = image_settings.dwell_time*image_settings.resolution[0]*image_settings.resolution[1]
            time.sleep(self.sleep_time)

    @thread_worker
    def consume_image_queue(self, parent_ui = None, sleep = 0.1):

        logging.info("Consuming image queue")

        try:
            while not self.stop_event.is_set():
                image = self.image_queue.get(timeout=1)
                if image.metadata.image_settings.save:
                    image.metadata.image_settings.filename = f"{image.metadata.image_settings.filename}_{utils.current_timestamp()}"
                    filename = os.path.join(image.metadata.image_settings.path, image.metadata.image_settings.filename)
                    image.save(path=filename)
                    logging.info(f"Saved image to {filename}")

                logging.info(f"Image: {image.data.shape}")
                logging.info(f"-" * 50)

                if parent_ui is not None:
                        parent_ui.live_imaging_signal.emit({"image": image})
                time.sleep(sleep)
        except KeyboardInterrupt:
            self.stop_event
            logging.info("Keyboard interrupt, stopping live imaging")
        except Exception as e:
            self.stop_event.set()
            import traceback
            logging.error(traceback.format_exc())
        finally:
            logging.info("Stopped thread image consumption")
    
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

        return base_position + FibsemStagePosition(x=dx, y=dy) # TODO: implement

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
        logging.info(f"Retracting manipulator...")
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
        _check_beam(BeamType.ION, self.system)
        self.set("current", mill_settings.milling_current, BeamType.ION)
        self.set("voltage", mill_settings.milling_voltage, BeamType.ION)
    
        logging.debug({"msg": "setup_milling", "mill_settings": mill_settings.to_dict()})

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False) -> None:
        """Run milling with the specified current and voltage."""
        _check_beam(BeamType.ION, self.system)
        # import random
        # time.sleep(random.randint(1, 5))
        time.sleep(5)
        logging.debug({"msg": "run_milling", "milling_current": milling_current, "milling_voltage": milling_voltage, "asynch": asynch})

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        """Finish milling by restoring the imaging current and voltage."""
        _check_beam(BeamType.ION, self.system)
        logging.info(f"Finishing milling: {imaging_current:.2e}")
        self.set("current", imaging_current, BeamType.ION)
        self.set("voltage", imaging_voltage, BeamType.ION)


    def estimate_milling_time(self, patterns: list) -> float:
        """Estimate the milling time for the specified patterns."""
        total_time = 0
        for pattern in patterns:
            total_time += 5
        
        return total_time
        

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings) -> None:
        logging.debug({"msg": "draw_rectangle", "pattern_settings": pattern_settings.to_dict()})

    def draw_line(self, pattern_settings: FibsemLineSettings) -> None:
        logging.debug({"msg": "draw_line", "pattern_settings": pattern_settings.to_dict()})

    def draw_circle(self, pattern_settings: FibsemCircleSettings) -> None:
        logging.debug({"msg": "draw_circle", "pattern_settings": pattern_settings.to_dict()})
    
    def draw_annulus(self, pattern_settings: FibsemCircleSettings) -> None:
        logging.debug({"msg": "draw_annulus", "pattern_settings": pattern_settings.to_dict()})

    def get_scan_directions(self) -> list:
        """
        Returns the available scan directions of the microscope.

        Returns:
            list: The scan direction of the microscope.

        Raises:
            None
        """
        values = ["BottomToTop", 
                "LeftToRight", 	
                "RightToLeft", 	
                "TopToBottom"]
        return values 

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings,
        path: str):
        logging.debug({"msg": "draw_bitmap_pattern", "pattern_settings": pattern_settings.to_dict(), "path": path})
        return 
    def run_milling_drift_corrected(self):
        _check_beam(BeamType.ION, self.system)
        return

    def setup_sputter(self, protocol: dict) -> None:
        _check_sputter(self.system)
        logging.info(f"Setting up sputter: {protocol}")

    def draw_sputter_pattern(self, hfw: float, line_pattern_length: float, sputter_time: float):
        logging.debug({"msg": "draw_sputter_pattern", "hfw": hfw, "line_pattern_length": line_pattern_length, "sputter_time": sputter_time})

    def run_sputter(self, **kwargs):
        _check_sputter(self.system)
        logging.info(f"Running sputter: {kwargs}")

    def finish_sputter(self, **kwargs):
        _check_sputter(self.system)
        logging.info(f"Finishing sputter: {kwargs}")

    def get_available_values(self, key: str, beam_type: BeamType = None) -> list[float]:
        
        values = []
        if key == "current":
            _check_beam(beam_type, self.system)
            if beam_type == BeamType.ELECTRON:
                values = [1.0e-12]
            if beam_type == BeamType.ION:
                values = [20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.6e-9, 28.0e-9, 120e-9]
        

        if key == "application_file":
            values = ["Si", "autolamella", "cryo_Pt_dep"]

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
                    logging.debug(f"Plasma gas cannot be set on this microscope.")
                    return
                if not self.check_available_values("plasma_gas", value, beam_type):
                    logging.warning(f"Plasma gas {value} not available. Available values: {self.get_available_values('plasma_gas', beam_type)}")
                    return 
                logging.info(f"Setting plasma gas to {value}... this may take some time...")
                self.system.ion.plasma_gas = value
                logging.info(f"Plasma gas set to {value}.")

                return

        # stage properties
        if key == "stage_home":
            logging.info(f"Homing stage...")
            self.stage_system.is_homed = True
            logging.info(f"Stage homed.")
            return
        
        if key == "stage_link":
            logging.info(f"Linking stage...")
            self.stage_system.is_linked = True
            logging.info(f"Stage linked.")
            return

        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info(f"Pumping chamber...")
                self.chamber.state = "Pumped"
                self.chamber.pressure = 1e-6 # 1 uTorr
                logging.info(f"Chamber pumped.")
            else:
                logging.info(f"Invalid value for pump_chamber: {value}")
            return
        if key == "vent_chamber":
            if value:
                logging.info(f"Venting chamber...")
                self.chamber.state = "Vented"
                self.chamber.pressure = 1e5
                logging.info(f"Chamber vented.")
            else:
                logging.info(f"Invalid value for vent_chamber: {value}")
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
        logging.info(f"Stage homed.")
        return


######################################## Helper functions ########################################



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

import warnings

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
