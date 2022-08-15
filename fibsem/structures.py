# fibsem structures

from dataclasses import dataclass
from autoscript_sdb_microscope_client.structures import AdornedImage, StagePosition
from enum import Enum
from datetime import datetime

from pathlib import Path
import numpy as np

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

    @classmethod
    def __from_dict__(self, settings: dict) -> 'GammaSettings':
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

    @classmethod
    def __from_dict__(self, settings: dict) -> 'ImageSettings':

        if "autoconstrast" not in settings:
            settings["autocontrast"] = False
        if "save" not in settings: 
            settings["save"] = False
        if "save_path" not in settings:
            settings["save_path"] = ""
        if "label" not in settings:
            settings["label"] = ""
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

    @classmethod
    def __from_dict__(self, state_dict: dict) -> 'BeamSettings':
        beam_settings = BeamSettings(
            beam_type=BeamType[state_dict["beam_type"]],
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
            "absolute_position": {
                "x": self.absolute_position.x,
                "y": self.absolute_position.y,
                "z": self.absolute_position.z,
                "r": self.absolute_position.r,
                "t": self.absolute_position.t,
                "coordinate_system": self.absolute_position.coordinate_system,
            },
            "eb_settings": self.eb_settings.__to_dict__(),
            "ib_settings": self.ib_settings.__to_dict__(),
        }

        return state_dict

    @classmethod
    def __from_dict__(self, state_dict: dict) -> 'MicroscopeState':
        # TOOD: class method
        microscope_state = MicroscopeState(
            timestamp=state_dict["timestamp"],
            absolute_position=StagePosition(
                x=state_dict["absolute_position"]["x"],
                y=state_dict["absolute_position"]["y"],
                z=state_dict["absolute_position"]["z"],
                r=state_dict["absolute_position"]["r"],
                t=state_dict["absolute_position"]["t"],
                coordinate_system=state_dict["absolute_position"]["coordinate_system"],
            ),
            eb_settings=BeamSettings.__from_dict__(state_dict["eb_settings"]),
            ib_settings=BeamSettings.__from_dict__(state_dict["ib_settings"]),
        )

        return microscope_state


@dataclass
class MillingSettings:
    width: float
    height: float
    depth: float
    rotation: float
    centre_x: float = 0.0
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

    @classmethod
    def __from_dict__(self, settings: dict) -> 'MillingSettings':

        if "centre_x" not in settings:
            settings["centre_x"] = 0
        if "centre_y" not in settings:
            settings["centre_y"] = 0
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


@dataclass
class SystemSettings:
    ip_address: str = "10.0.0.1"
    application_file: str = "autolamella"
    plasma_gas: str = "Argon" # proper case, e.g. Argon, Oxygen
    high_voltage: float = 30000 # volts


    def __to_dict__(self) -> dict:

        settings_dict = {
            "ip_address": self.ip_address,
            "application_file": self.application_file,
            "plasma_gas": self.plasma_gas,
            "high_voltage": self.high_voltage,
        }

        return settings_dict
    
    @classmethod
    def __from_dict__(self, settings: dict) -> 'SystemSettings':

        system_settings = SystemSettings(
            ip_address=settings["ip_address"],
            application_file=settings["application_file"],
            plasma_gas=settings["plasma_gas"],
            high_voltage=settings["high_voltage"]
        )

        return system_settings

@dataclass
class StageSettings:
    rotation_flat_to_electron: float = 50 # degrees
    rotation_flat_to_ion: float = 230 # degrees
    tilt_flat_to_electron: float = 27 # degrees (pre_tilt)
    tilt_flat_to_ion: float = 52 # degrees

    def __to_dict__(self) -> dict:

        settings = {
            "rotation_flat_to_electron": self.rotation_flat_to_electron,
            "rotation_flat_to_ion": self.rotation_flat_to_ion,
            "tilt_flat_to_electron": self.tilt_flat_to_electron,
            "tilt_flat_to_ion": self.tilt_flat_to_ion
        }
        return settings

    @classmethod
    def __from_dict__(self, settings: dict) -> 'StageSettings':
            
        stage_settings = StageSettings(
            rotation_flat_to_electron=settings["rotation_flat_to_electron"],
            rotation_flat_to_ion=settings["rotation_flat_to_ion"],
            tilt_flat_to_electron=settings["tilt_flat_to_electron"],
            tilt_flat_to_ion=settings["tilt_flat_to_ion"]  
        )

        return stage_settings


@dataclass
class CalibrationSettings:
    imaging_current: float = 20.e-12
    milling_current: float = 2.e-9
    max_hfw_eb: float = 2700e-6
    max_hfw_ib: float = 900e-6
    eucentric_height_eb: float =  4.0e-3
    eucentric_height_ib: float =16.5e-3
    eucentric_height_tolerance: float = 0.5e-3
    needle_stage_height_limit: float = 3.7e-3
    max_working_distance_eb: float = 6.0e-3

    def __to_dict__(self) -> dict:

        settings = {
                "imaging_current": self.imaging_current,
                "milling_current": self.milling_current,
                "max_hfw_eb": self.max_hfw_eb,
                "max_hfw_ib": self.max_hfw_ib,
                "eucentric_height_eb": self.eucentric_height_eb,
                "eucentric_height_ib": self.eucentric_height_ib,
                "eucentric_height_tolerance": self.eucentric_height_tolerance,
                "needle_stage_height_limit": self.needle_stage_height_limit,
                "max_working_distance_eb": self.max_working_distance_eb,
            }

        return settings

    @classmethod
    def __from_dict__(self, settings: dict) -> 'CalibrationSettings':

        calibration_settings = CalibrationSettings(
            imaging_current=settings["imaging_current"],
            milling_current=settings["milling_current"],
            max_hfw_eb=settings["max_hfw_eb"],
            max_hfw_ib=settings["max_hfw_ib"],
            eucentric_height_eb=settings["eucentric_height_eb"],
            eucentric_height_ib=settings["eucentric_height_ib"], 
            eucentric_height_tolerance=settings["eucentric_height_tolerance"],
            needle_stage_height_limit=settings["needle_stage_height_limit"], 
            max_working_distance_eb=settings["max_working_distance_eb"]
        )

        return calibration_settings
