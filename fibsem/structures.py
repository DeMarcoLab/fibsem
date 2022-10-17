# fibsem structures

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from autoscript_sdb_microscope_client.structures import (AdornedImage, StagePosition, ManipulatorPosition)
import yaml

#@patrickcleeve: dataclasses.asdict -> :(

@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0


# TODO: convert these to match autoscript...
class BeamType(Enum):
    ELECTRON = 1 # Electron
    ION = 2      # Ion
    # CCD_CAM = 3
    # NavCam = 4 # see enumerations/ImagingDevice


@dataclass
class GammaSettings:
    enabled: bool = False
    min_gamma: float = 0.0
    max_gamma: float = 2.0
    scale_factor: float = 0.1
    threshold: int  = 45 #px

    @staticmethod
    def __from_dict__(settings: dict) -> 'GammaSettings':
        gamma_settings = GammaSettings(
            enabled=settings["enabled"],
            min_gamma=settings["min_gamma"],
            max_gamma=settings["max_gamma"],
            scale_factor=settings["scale_factor"],
            threshold=settings["threshold"],
        )
        return gamma_settings

@dataclass
class ImageSettings:
    resolution: str
    dwell_time: float
    hfw: float
    autocontrast: bool
    beam_type: BeamType
    save: bool
    label: str
    gamma: GammaSettings
    save_path: Path = None

    @staticmethod
    def __from_dict__(settings: dict) -> 'ImageSettings':

        if "autocontrast" not in settings:
            settings["autocontrast"] = False
        if "save" not in settings: 
            settings["save"] = False
        if "save_path" not in settings:
            settings["save_path"] = os.getcwd()
        if "label" not in settings:
            settings["label"] = "default_image"
        if "gamma" not in settings:
            settings["gamma"] = {
                "enabled": False,
                "min_gamma": 0.0,
                "max_gamma": 2.0,
                "scale_factor": 0.1,
                "threshold": 45,
            }

        image_settings = ImageSettings(
            resolution=settings["resolution"],
            dwell_time=settings["dwell_time"],
            hfw=settings["hfw"], 
            autocontrast=settings["autocontrast"], 
            beam_type=BeamType[settings["beam_type"].upper()],
            gamma=GammaSettings.__from_dict__(settings["gamma"]),
            save=settings["save"], 
            save_path=settings["save_path"],
            label=settings["label"]
        )

        return image_settings


    def __to_dict__(self) -> dict:

        settings_dict = {
            "resolution": self.resolution,
            "dwell_time": self.dwell_time,
            "hfw": self.hfw,
            "autocontrast": self.autocontrast,
            "gamma": {
                "enabled": self.gamma.enabled,
                "min_gamma": self.gamma.min_gamma,
                "max_gamma": self.gamma.max_gamma,
                "scale_factor": self.gamma.scale_factor,
                "threshold": self.gamma.threshold,
            },
            "save": self.save,
            "save_path": self.save_path,
            "label": self.label
        }

        return settings_dict

@dataclass
class ReferenceImages:
    low_res_eb: AdornedImage
    high_res_eb: AdornedImage
    low_res_ib: AdornedImage
    high_res_ib: AdornedImage

    def __iter__(self) -> list[AdornedImage]:

        yield self.low_res_eb, self.high_res_eb, self.low_res_ib, self.high_res_ib


@dataclass
class BeamSettings:
    beam_type: BeamType
    working_distance: float = None
    beam_current: float = None
    hfw: float = None
    resolution: str = None
    dwell_time: float = None
    # stigmation: tuple[float] = None

    def __to_dict__(self) -> dict:

        state_dict = {
            "beam_type": self.beam_type.name,
            "working_distance": self.working_distance,
            "beam_current": self.beam_current,
            "hfw": self.hfw,
            "resolution": self.resolution,
            "dwell_time": self.dwell_time,
            # "stigmation": self.stigmation,
        }

        return state_dict

    @staticmethod
    def __from_dict__(state_dict: dict) -> 'BeamSettings':
        beam_settings = BeamSettings(
            beam_type=BeamType[state_dict["beam_type"].upper()], # TODO: remove this key, just assign directly
            working_distance=state_dict["working_distance"],
            beam_current=state_dict["beam_current"],
            hfw=state_dict["hfw"],
            resolution=state_dict["resolution"],
            dwell_time=state_dict["dwell_time"],
            # stigmation=state_dict["stigmation"],
        )

        return beam_settings


@dataclass
class MicroscopeState:
    timestamp: float = datetime.timestamp(datetime.now()) 
    absolute_position: StagePosition = StagePosition()
    eb_settings: BeamSettings = BeamSettings(beam_type=BeamType.ELECTRON)
    ib_settings: BeamSettings = BeamSettings(beam_type=BeamType.ION)

    def __to_dict__(self) -> dict:

        state_dict = {
            "timestamp": self.timestamp,
            "absolute_position": stage_position_to_dict(self.absolute_position), 
            "eb_settings": self.eb_settings.__to_dict__(),
            "ib_settings": self.ib_settings.__to_dict__(),
        }

        return state_dict

    @staticmethod
    def __from_dict__(state_dict: dict) -> 'MicroscopeState':
        microscope_state = MicroscopeState(
            timestamp=state_dict["timestamp"],
            absolute_position=stage_position_from_dict(state_dict["absolute_position"]),
            eb_settings=BeamSettings.__from_dict__(state_dict["eb_settings"]),
            ib_settings=BeamSettings.__from_dict__(state_dict["ib_settings"]),
        )

        return microscope_state


@dataclass
class MillingSettings:
    width: float
    height: float
    depth: float
    rotation: float = 0.0 # deg
    centre_x: float = 0.0 # TODO: change to Point?
    centre_y: float = 0.0 
    milling_current: float = 20.0e-12
    scan_direction: str = "TopToBottom"
    cleaning_cross_section: bool = False


    def __to_dict__(self) -> dict:

        settings_dict = {
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "rotation": np.rad2deg(self.rotation),
            "centre_x": self.centre_x,
            "centre_y": self.centre_y,
            "milling_current": self.milling_current,
            "scan_direction": self.scan_direction,
            "cleaning_cross_section": self.cleaning_cross_section
        }

        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict) -> 'MillingSettings':

        if "centre_x" not in settings:
            settings["centre_x"] = 0
        if "centre_y" not in settings:
            settings["centre_y"] = 0
        if "rotation" not in settings:
            settings["rotation"] = 0
        if "scan_direction" not in settings:
            settings["scan_direction"] = "TopToBottom"
        if "cleaning_cross_section" not in settings:
            settings["cleaning_cross_section"] = False
        
        milling_settings = MillingSettings(
            width=settings["width"],
            height=settings["height"],
            depth=settings["depth"],
            rotation=np.deg2rad(settings["rotation"]),
            centre_x=settings["centre_x"],
            centre_y=settings["centre_y"],
            milling_current=settings["milling_current"],
            scan_direction=settings["scan_direction"],
            cleaning_cross_section=settings["cleaning_cross_section"],
        )

        return milling_settings


def stage_position_to_dict(stage_position: StagePosition) -> dict:

    stage_position_dict = {
                "x": stage_position.x,
                "y": stage_position.y,
                "z": stage_position.z,
                "r": stage_position.r,
                "t": stage_position.t,
                "coordinate_system": stage_position.coordinate_system,
            }

    return stage_position_dict


def stage_position_from_dict(state_dict: dict) -> StagePosition:

    stage_position = StagePosition(
        x=state_dict["x"],
        y=state_dict["y"],
        z=state_dict["z"],
        r=state_dict["r"],
        t=state_dict["t"],
        coordinate_system=state_dict["coordinate_system"],
    )

    return stage_position


def manipulator_position_to_dict(position: ManipulatorPosition) -> dict:

    position_dict = {
        "x": position.x,
        "y": position.y,
        "z": position.z,
        "r": None,
        "coordinate_system": position.coordinate_system
    }

    return position_dict


def manipulator_position_from_dict(position_dict: dict) -> ManipulatorPosition:

    position = ManipulatorPosition(
        x=position_dict["x"],
        y=position_dict["y"],
        z=position_dict["z"],
        r=position_dict["r"],
        coordinate_system=position_dict["coordinate_system"] 

    )

    return position




@dataclass
class BeamSystemSettings:
    beam_type: BeamType
    voltage: float
    current: float
    detector_type: str
    detector_mode: str
    eucentric_height: float 
    plasma_gas: str = None

    def __to_dict__(self) -> dict:

        settings_dict = {
            "voltage": self.voltage,
            "current": self.current,
            "detector_type": self.detector_type,
            "detector_mode": self.detector_mode,
            "eucentric_height": self.eucentric_height,
            "plasma_gas": self.plasma_gas,

        }

        return settings_dict
    
    @staticmethod
    def __from_dict__(settings: dict, beam_type: BeamType) -> 'BeamSystemSettings':

        if "plasma_gas" not in settings:
            settings["plasma_gas"] = "NULL"


        system_settings = BeamSystemSettings(
            beam_type=beam_type,
            voltage = settings["voltage"],
            current = settings["current"],
            detector_type=settings["detector_type"],
            detector_mode=settings["detector_mode"],
            eucentric_height=settings["eucentric_height"],
            plasma_gas=settings["plasma_gas"].capitalize(),
        )

        return system_settings

# TODO: change this to use pretilt_angle, flat_to_electron, and flat_to_ion tilts, for better separation
@dataclass
class StageSettings:
    rotation_flat_to_electron: float = 50 # degrees
    rotation_flat_to_ion: float = 230 # degrees
    tilt_flat_to_electron: float = 27 # degrees (pre_tilt)
    tilt_flat_to_ion: float = 52 # degrees
    pre_tilt: float = 35 # degrees
    needle_stage_height_limit: float = 3.7e-3

    def __to_dict__(self) -> dict:

        settings = {
            "rotation_flat_to_electron": self.rotation_flat_to_electron,
            "rotation_flat_to_ion": self.rotation_flat_to_ion,
            "tilt_flat_to_electron": self.tilt_flat_to_electron,
            "tilt_flat_to_ion": self.tilt_flat_to_ion,
            "pre_tilt": self.pre_tilt,
            "needle_stage_height_limit": self.needle_stage_height_limit
        }
        return settings

    @staticmethod
    def __from_dict__(settings: dict) -> 'StageSettings':
            
        stage_settings = StageSettings(
            rotation_flat_to_electron=settings["rotation_flat_to_electron"],
            rotation_flat_to_ion=settings["rotation_flat_to_ion"],
            tilt_flat_to_electron=settings["tilt_flat_to_electron"],
            tilt_flat_to_ion=settings["tilt_flat_to_ion"],
            pre_tilt=settings["pre_tilt"],
            needle_stage_height_limit=settings["needle_stage_height_limit"]  
        )

        return stage_settings



@dataclass
class SystemSettings:
    ip_address: str = "10.0.0.1"
    application_file: str = "autolamella"
    stage: StageSettings = None
    ion: BeamSystemSettings = None
    electron: BeamSystemSettings = None


    def __to_dict__(self) -> dict:

        settings_dict = {
            "ip_address": self.ip_address,
            "application_file": self.application_file,
            "stage": self.stage.__to_dict__(),
            "ion": self.ion.__to_dict__(),
            "electron": self.electron.__to_dict__() 
        }

        return settings_dict
    
    @staticmethod
    def __from_dict__(settings: dict) -> 'SystemSettings':

        system_settings = SystemSettings(
            ip_address=settings["ip_address"],
            application_file=settings["application_file"],
            stage=StageSettings.__from_dict__(settings["stage"]),
            ion=BeamSystemSettings.__from_dict__(settings["ion"], BeamType.ION),
            electron=BeamSystemSettings.__from_dict__(settings["electron"], BeamType.ELECTRON)
        )

        return system_settings

@dataclass
class DefaultSettings:
    imaging_current: float = 20.e-12
    milling_current: float = 2.e-9


    @staticmethod
    def __from_dict__(settings: dict) -> 'DefaultSettings':
        
        default_settings = DefaultSettings(
            imaging_current=settings["imaging_current"],
            milling_current=settings["milling_current"],
        )
        return default_settings

@dataclass
class MicroscopeSettings:
    system: SystemSettings
    default: DefaultSettings
    image: ImageSettings
    protocol: dict = None


    def __to_dict__(self) -> dict:

        settings_dict = {
            "system": self.system.__to_dict__(),
            "user": self.image.__to_dict__(),

        }

        return settings_dict



# state
from abc import ABC, abstractmethod, abstractstaticmethod



class FibsemStage(Enum):
    Base = 1

@dataclass
class FibsemState:
    stage: FibsemStage = FibsemStage.Base
    microscope_state: MicroscopeState = MicroscopeState()
    start_timestamp: float = None
    end_timestamp: float = None

    def __to_dict__(self) -> dict:

        state_dict = {
            "stage": self.stage.name,
            "microscope_state": self.microscope_state.__to_dict__(),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
        }

        return state_dict

    @abstractstaticmethod
    def __from_dict__(self, state_dict: dict) -> 'FibsemState':

        autoliftout_state = FibsemState(
            stage=FibsemState[state_dict["stage"]],
            microscope_state=MicroscopeState.__from_dict__(state_dict["microscope_state"]),
            start_timestamp=state_dict["start_timestamp"],
            end_timestamp=state_dict["end_timestamp"],
        )

        return autoliftout_state