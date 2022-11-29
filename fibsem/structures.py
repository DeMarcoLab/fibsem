# fibsem structures

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import tifffile as tff
import json

import numpy as np
from autoscript_sdb_microscope_client.structures import (AdornedImage, StagePosition, ManipulatorPosition, Rectangle)
import yaml
from fibsem.config import METADATA_VERSION


#@patrickcleeve: dataclasses.asdict -> :(

# TODO: overload constructors instead of from_dict...
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0

    def __to_dict__(self) -> dict:
        return {"x": self.x, "y": self.y}
    
    @staticmethod
    def __from_dict__(d: dict) -> "Point":
        return Point(d["x"], d["y"])
    
    def __to_list__(self) -> list:
        return [self.x, self.y]

    @staticmethod
    def __from_list__(l: list) -> "Point":
        return Point(l[0], l[1])

# TODO: convert these to match autoscript...
class BeamType(Enum):
    ELECTRON = 1 # Electron
    ION = 2      # Ion
    # CCD_CAM = 3
    # NavCam = 4 # see enumerations/ImagingDevice

@dataclass
class FibsemRectangle():
    """Universal Rectangle class used for ReducedArea"""
    left: float = 0.0,
    top: float = 0.0,
    width: float = 0.0,
    height: float = 0.0

    def __from_dict__(settings: dict) -> "FibsemRectangle":
        return FibsemRectangle(
            left=settings["left"],
            top=settings["top"],
            width=settings["width"],
            height=settings["height"],
        )
    def __to_dict__(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }
    
    def __to_FEI__(self) -> Rectangle:
        return Rectangle(self.left, self.top, self.width, self.height)
    
    @classmethod
    def __from_FEI__(cls, rect: Rectangle) -> "FibsemRectangle":
        return cls(rect.left, rect.top, rect.width, rect.height)

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
    resolution: str = None
    dwell_time: float = None
    hfw: float = None
    autocontrast: bool = None
    beam_type: BeamType = None
    save: bool = None
    label: str = None
    gamma: GammaSettings = None
    save_path: Path = None
    reduced_area: FibsemRectangle = None

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
        if "reduced_area" in settings and settings["reduced_area"] is not None:
            reduced_area = FibsemRectangle.__from_dict__(settings["reduced_area"]),
        else:
            reduced_area = None

        image_settings = ImageSettings(
            resolution=settings["resolution"],
            dwell_time=settings["dwell_time"],
            hfw=settings["hfw"], 
            autocontrast=settings["autocontrast"], 
            beam_type=BeamType[settings["beam_type"].upper()] if settings["beam_type"] is not None else None,
            gamma=GammaSettings.__from_dict__(settings["gamma"]) if settings["gamma"] is not None else None,
            save=settings["save"], 
            save_path=settings["save_path"],
            label=settings["label"],
            reduced_area=reduced_area)

        return image_settings


    def __to_dict__(self) -> dict:
        
        settings_dict = {
            "beam_type": self.beam_type.name if self.beam_type is not None else None,
            "resolution": self.resolution if self.resolution is not None else None, 
            "dwell_time": self.dwell_time if self.dwell_time is not None else None,
            "hfw": self.hfw if self.hfw is not None else None,
            "autocontrast": self.autocontrast if self.autocontrast is not None else None,
            "gamma": {
                "enabled": self.gamma.enabled,
                "min_gamma": self.gamma.min_gamma,
                "max_gamma": self.gamma.max_gamma,
                "scale_factor": self.gamma.scale_factor,
                "threshold": self.gamma.threshold,
            } if self.gamma is not None else None,
            "save": self.save if self.save is not None else None,
            "save_path": self.save_path if self.save_path is not None else None,
            "label": self.label if self.label is not None else None,
            "reduced_area": {
                "left": self.reduced_area.left,
                "top": self.reduced_area.top,
                "width": self.reduced_area.width,
                "height": self.reduced_area.height,
                } if self.reduced_area is not None else None,
        }

        return settings_dict


@dataclass
class BeamSettings:
    beam_type: BeamType
    working_distance: float = None
    beam_current: float = None
    hfw: float = None
    resolution: str = None
    dwell_time: float = None
    stigmation: Point = None
    shift: Point = None

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
            beam_type=BeamType[state_dict["beam_type"].upper()],
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
    # default: DefaultSettings
    image: ImageSettings
    protocol: dict = None

    def __to_dict__(self) -> dict:

        settings_dict = {
            "system": self.system.__to_dict__(),
            "user": self.image.__to_dict__(),

        }

        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict, protocol: dict = None) -> 'MicroscopeSettings':

        return MicroscopeSettings(
            system=SystemSettings.__from_dict__(settings["system"]),
            image = ImageSettings.__from_dict__(settings["user"]),
            # default=DefaultSettings.__from_dict__(settings["user"]),
            protocol= protocol
        )


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
    

@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage."""

    image_settings: ImageSettings
    pixel_size: Point
    microscope_state: MicroscopeState
    version: str = METADATA_VERSION

    def __to_dict__(self) -> dict:
        """Converts metadata to a dictionary.

        Returns:
            dictionary: self as a dictionary
        """
        if self.image_settings is not None:
            settings_dict = self.image_settings.__to_dict__()
        if self.version is not None:
            settings_dict["version"] = self.version
        if self.pixel_size is not None:
            settings_dict["pixel_size"] = self.pixel_size.__to_dict__()
        if self.microscope_state is not None:
            settings_dict["microscope_state"] = self.microscope_state.__to_dict__()
        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict) -> "ImageSettings":
        """Converts a dictionary to metadata."""

        image_settings = ImageSettings.__from_dict__(settings)
        if settings["version"] is not None:
            version = settings["version"]
        if settings["pixel_size"] is not None:
            pixel_size = Point.__from_dict__(settings["pixel_size"])
        if settings["microscope_state"] is not None:
            microscope_state = MicroscopeState(
                timestamp=settings["microscope_state"]["timestamp"],
                absolute_position=StagePosition(),
                eb_settings=BeamSettings.__from_dict__(settings["microscope_state"]["eb_settings"]),
                ib_settings=BeamSettings.__from_dict__(settings["microscope_state"]["ib_settings"]),
            )

        metadata = FibsemImageMetadata(
            image_settings=image_settings,
            version=version,
            pixel_size=pixel_size,
            microscope_state=microscope_state,
        )
        return metadata


class FibsemImage:
    def __init__(self, data: np.ndarray, metadata: FibsemImageMetadata = None):
        self.data = check_data_format(data)
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    @classmethod
    def load(cls, tiff_path: str) -> "FibsemImage":
        """Loads a FibsemImage from a tiff file.

        Args:
            tiff_path (path): path to the tif* file

        Returns:
            FibsemImage: instance of FibsemImage
        """
        with tff.TiffFile(tiff_path) as tiff_image:
            data = tiff_image.asarray()
            try:
                metadata = json.loads(
                    tiff_image.pages[0].tags["ImageDescription"].value
                )
                metadata = FibsemImageMetadata.__from_dict__(metadata)
            except Exception as e:
                metadata = None
                print(f"Error: {e}")
        return cls(data=data, metadata=metadata)

    def save(self, save_path: Path) -> None:
        """Saves a FibsemImage to a tiff file.

        Inputs:
            save_path (path): path to save directory and filename
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = Path(save_path).with_suffix(".tif")
        
        if self.metadata is not None:
            metadata_dict = self.metadata.__to_dict__()
        else:
            metadata_dict = None
        tff.imwrite(
            save_path,
            self.data,
            metadata=metadata_dict,
        )

    @classmethod
    def fromAdornedImage(
        cls, adorned: AdornedImage, image_settings: ImageSettings, state: MicroscopeState = None
    ) -> "FibsemImage":
        """Creates FibsemImage from an AdornedImage (microscope output format).

        Args:
            adorned (AdornedImage): Adorned Image from microscope
            metadata (FibsemImageMetadata, optional): metadata extracted from microscope output. Defaults to None.

        Returns:
            FibsemImage: instance of FibsemImage from AdornedImage
        """

        if state is None:
            state = MicroscopeState(
                timestamp=adorned.metadata.acquisition.acquisition_datetime,
                absolute_position=StagePosition(),
                eb_settings=BeamSettings(beam_type=BeamType.ELECTRON),
                ib_settings=BeamSettings(beam_type=BeamType.ION),
            )
        else:
            state.timestamp = adorned.metadata.acquisition.acquisition_datetime

        pixel_size = Point(adorned.metadata.binary_result.pixel_size.x, adorned.metadata.binary_result.pixel_size.y)
        metadata=FibsemImageMetadata(image_settings=image_settings, pixel_size=pixel_size, microscope_state=state)
        return cls(data=adorned.data, metadata=metadata)


@dataclass
class ReferenceImages:
    low_res_eb: FibsemImage
    high_res_eb: FibsemImage
    low_res_ib: FibsemImage
    high_res_ib: FibsemImage

    def __iter__(self) -> list[FibsemImage]:

        yield self.low_res_eb, self.high_res_eb, self.low_res_ib, self.high_res_ib

def check_data_format(data: np.ndarray) -> np.ndarray:
    """Checks that data is in the correct format."""
    assert data.ndim == 2  # or data.ndim == 3
    assert data.dtype == np.uint8
    # if data.ndim == 3 and data.shape[2] == 1:
    #     data = data[:, :, 0]
    return data