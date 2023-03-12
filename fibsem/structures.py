# fibsem structures

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import tifffile as tff

from fibsem.config import load_yaml

try:
    from tescanautomation.Common import Document
    TESCAN = True
except:
    TESCAN = False

try:
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage, ManipulatorPosition, Rectangle, StagePosition)
    THERMO = True
except:
    THERMO = False

import yaml

from fibsem.config import METADATA_VERSION

# @patrickcleeve: dataclasses.asdict -> :(

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
    
    def __add__(self, other) -> 'Point':
        return Point(self.x + other.x, self.y + other.y)


# TODO: convert these to match autoscript...
class BeamType(Enum):
    """Enumerator Class for Beam Type
        1: Electron Beam
        2: Ion Beam

    """
    ELECTRON = 1  # Electron
    ION = 2  # Ion
    # CCD_CAM = 3
    # NavCam = 4 # see enumerations/ImagingDevice

class MovementMode(Enum):
    Stable = 1
    Eucentric = 2
    # Needle = 3

@dataclass
class FibsemStagePosition:
    """Data class for storing stage position data.

Attributes:
    x (float): The X position of the stage in meters.
    y (float): The Y position of the stage in meters.
    z (float): The Z position of the stage in meters.
    r (float): The Rotation of the stage in radians.
    t (float): The Tilt of the stage in radians.
    coordinate_system (str): The coordinate system used for the stage position.

Methods:
    __to_dict__(): Convert the stage position object to a dictionary.
    __from_dict__(data: dict): Create a new stage position object from a dictionary.
    to_autoscript_position(stage_tilt: float = 0.0) -> StagePosition: Convert the stage position to a StagePosition object that is compatible with Autoscript.
    from_autoscript_position(position: StagePosition, stage_tilt: float = 0.0) -> None: Create a new FibsemStagePosition object from a StagePosition object that is compatible with Autoscript.
    to_tescan_position(stage_tilt: float = 0.0): Convert the stage position to a format that is compatible with Tescan.
    from_tescan_position(): Create a new FibsemStagePosition object from a Tescan-compatible stage position.
"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    t: float = 0.0
    coordinate_system: str = None

    def __to_dict__(self) -> dict:
        position_dict = {}
        position_dict["x"] = self.x
        position_dict["y"] = self.y
        position_dict["z"] = self.z
        position_dict["r"] = self.r
        position_dict["t"] = self.t
        position_dict["coordinate_system"] = self.coordinate_system

        return position_dict

    @classmethod
    def __from_dict__(cls, data: dict) -> "FibsemStagePosition":
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            r=data["r"],
            t=data["t"],
            coordinate_system=data["coordinate_system"],
        )

    if THERMO:

        def to_autoscript_position(self, stage_tilt: float = 0.0) -> StagePosition:
            return StagePosition(
                x=self.x,
                y=self.y, #/ np.cos(stage_tilt),
                z=self.z, #/ np.cos(stage_tilt),
                r=self.r,
                t=self.t,
                coordinate_system=self.coordinate_system,
            )

        @classmethod
        def from_autoscript_position(cls, position: StagePosition, stage_tilt: float = 0.0) -> None:
            return cls(
                x=position.x,
                y=position.y, # * np.cos(stage_tilt),
                z=position.z, # * np.cos(stage_tilt),
                r=position.r,
                t=position.t,
                coordinate_system=position.coordinate_system,
            )

    if TESCAN:

        def to_tescan_position(self, stage_tilt: float = 0.0):
            self.y=self.y / np.cos(stage_tilt),

        @classmethod
        def from_tescan_position(self, stage_tilt: float = 0.0):
            self.y = self.y * np.cos(stage_tilt)


    def __add__(self, other:'FibsemStagePosition') -> 'FibsemStagePosition':
        return FibsemStagePosition(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.r + other.r,
            self.t + other.t,
            self.coordinate_system,
        )


@dataclass
class FibsemManipulatorPosition:
    """Data class for storing manipulator position data.

Attributes:
    x (float): The X position of the manipulator in meters.
    y (float): The Y position of the manipulator in meters.
    z (float): The Z position of the manipulator in meters.
    r (float): The Rotation of the manipulator in radians.
    t (float): The Tilt of the manipulator in radians.
    coordinate_system (str): The coordinate system used for the manipulator position.

Methods:
    __to_dict__(): Convert the manipulator position object to a dictionary.
    __from_dict__(data: dict): Create a new manipulator position object from a dictionary.
    to_autoscript_position() -> ManipulatorPosition: Convert the manipulator position to a ManipulatorPosition object that is compatible with Autoscript.
    from_autoscript_position(position: ManipulatorPosition) -> None: Create a new FibsemManipulatorPosition object from a ManipulatorPosition object that is compatible with Autoscript.
    to_tescan_position(): Convert the manipulator position to a format that is compatible with Tescan.
    from_tescan_position(): Create a new FibsemManipulatorPosition object from a Tescan-compatible manipulator position.

"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    t: float = 0.0
    coordinate_system: str = None

    def __to_dict__(self) -> dict:
        position_dict = {}
        position_dict["x"] = self.x
        position_dict["y"] = self.y
        position_dict["z"] = self.z
        position_dict["r"] = self.r
        position_dict["t"] = self.t
        position_dict["coordinate_system"] = self.coordinate_system

        return position_dict
    
    @classmethod
    def __from_dict__(cls, data: dict) -> "FibsemManipulatorPosition":
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            r=data["r"],
            t=data["t"],
            coordinate_system=data["coordinate_system"],
        )
    
    if THERMO:
            
            def to_autoscript_position(self) -> ManipulatorPosition:
                return ManipulatorPosition(
                    x=self.x,
                    y=self.y,
                    z=self.z,
                    r=self.r,
                    t=self.t,
                    coordinate_system=self.coordinate_system,
                )
    
            @classmethod
            def from_autoscript_position(cls, position: ManipulatorPosition) -> None:
                return cls(
                    x=position.x,
                    y=position.y,
                    z=position.z,
                    r=position.r,
                    t=position.t,
                    coordinate_system=position.coordinate_system,
                )
            

    if TESCAN:
            
            def to_tescan_position(self):
                pass
    
            @classmethod
            def from_tescan_position(self):
                pass

    def __add__(self, other:'FibsemManipulatorPosition') -> 'FibsemManipulatorPosition':

        return FibsemManipulatorPosition(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.r + other.r,
            self.t + other.t,
            self.coordinate_system,
        )



@dataclass
class FibsemRectangle:
    """Universal Rectangle class used for ReducedArea"""

    left: float = 0.0
    top: float = 0.0
    width: float = 0.0
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
            "left": float(self.left),
            "top": float(self.top),
            "width": float(self.width),
            "height": float(self.height),
        }

    if THERMO:

        def __to_FEI__(self) -> Rectangle:
            return Rectangle(self.left, self.top, self.width, self.height)

        @classmethod
        def __from_FEI__(cls, rect: Rectangle) -> "FibsemRectangle":
            return cls(rect.left, rect.top, rect.width, rect.height)


@dataclass
class ImageSettings:
    """A data class representing the settings for an image acquisition.

    Attributes:
        resolution (list of int): The resolution of the acquired image in pixels, [x, y].
        dwell_time (float): The time spent per pixel during image acquisition, in seconds.
        hfw (float): The horizontal field width of the acquired image, in microns.
        autocontrast (bool): Whether or not to apply automatic contrast enhancement to the acquired image.
        beam_type (BeamType): The type of beam to use for image acquisition.
        save (bool): Whether or not to save the acquired image to disk.
        label (str): The label to use when saving the acquired image.
        gamma_enabled (bool): Whether or not to apply gamma correction to the acquired image.
        save_path (Path): The path to the directory where the acquired image should be saved.
        reduced_area (FibsemRectangle): The rectangular region of interest within the acquired image, if any.

    Methods:
        __from_dict__(settings: dict) -> ImageSettings:
            Converts a dictionary of image settings to an ImageSettings object.
        __to_dict__() -> dict:
            Converts the ImageSettings object to a dictionary of image settings.
    """

    resolution: list = None
    dwell_time: float = None
    hfw: float = None
    autocontrast: bool = None
    beam_type: BeamType = None
    save: bool = None
    label: str = None
    gamma_enabled: bool = None
    save_path: Path = None
    reduced_area: FibsemRectangle = None

    @staticmethod
    def __from_dict__(settings: dict) -> "ImageSettings":


        if "reduced_area" in settings and settings["reduced_area"] is not None:
            reduced_area = FibsemRectangle.__from_dict__(settings["reduced_area"])
        else:
            reduced_area = None

        
        image_settings = ImageSettings(
            resolution=settings.get("resolution", [1536, 1024]),
            dwell_time=settings.get("dwell_time", 1.0e-6),
            hfw=settings.get("hfw", 150e-6),
            autocontrast=settings.get("autocontrast", False),
            beam_type=BeamType[settings.get("beam_type", "Electron").upper()],
            gamma_enabled=settings.get("gamma_enabled", False),
            save=settings.get("save", False),
            save_path=settings.get("save_path", os.getcwd()),
            label=settings.get("label", "default_image"),
            reduced_area=reduced_area,
        )

        return image_settings

    def __to_dict__(self) -> dict:

        settings_dict = {
            "beam_type": self.beam_type.name if self.beam_type is not None else None,
            "resolution": self.resolution if self.resolution is not None else None,
            "dwell_time": self.dwell_time if self.dwell_time is not None else None,
            "hfw": self.hfw if self.hfw is not None else None,
            "autocontrast": self.autocontrast
            if self.autocontrast is not None
            else None,
            "gamma_enabled": self.gamma_enabled if self.gamma_enabled is not None else None,
            "save": self.save if self.save is not None else None,
            "save_path": self.save_path if self.save_path is not None else None,
            "label": self.label if self.label is not None else None,
            "reduced_area": {
                "left": self.reduced_area.left,
                "top": self.reduced_area.top,
                "width": self.reduced_area.width,
                "height": self.reduced_area.height,
            }
            if self.reduced_area is not None
            else None,
        }

        return settings_dict

    @staticmethod
    def fromFibsemImage(image: 'FibsemImage') -> "ImageSettings":
        """Returns the image settings for a FibsemImage object.

        Args:
            image (FibsemImage): The FibsemImage object to get the image settings from.

        Returns:
            ImageSettings: The image settings for the given FibsemImage object.
        """
        from fibsem import utils
        from copy import deepcopy
        image_settings = deepcopy(image.metadata.image_settings)
        image_settings.label = utils.current_timestamp()
        image_settings.save = True
        
        return image_settings


@dataclass
class BeamSettings:
    """
    Dataclass representing the beam settings for an imaging session.

    Attributes:
        beam_type (BeamType): The type of beam to use for imaging.
        working_distance (float): The working distance for the microscope, in meters.
        beam_current (float): The beam current for the microscope, in amps.
        hfw (float): The horizontal field width for the microscope, in meters.
        resolution (list): The desired resolution for the image.
        dwell_time (float): The dwell time for the microscope.
        stigmation (Point): The point for stigmation correction.
        shift (Point): The point for shift correction.

    Methods:
        __to_dict__(): Returns a dictionary representation of the object.
        __from_dict__(state_dict: dict) -> BeamSettings: Returns a new BeamSettings object created from a dictionary.

    """
    beam_type: BeamType
    working_distance: float = None
    beam_current: float = None
    voltage: float = None
    hfw: float = None
    resolution: list = None
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
            "stigmation": self.stigmation.__to_dict__() if self.stigmation is not None else None,
            "shift": self.shift.__to_dict__() if self.shift is not None else None,
        }

        return state_dict

    @staticmethod
    def __from_dict__(state_dict: dict) -> "BeamSettings":

        if "stigmation" in state_dict and state_dict["stigmation"] is not None:
            stigmation = FibsemRectangle.__from_dict__(state_dict["stigmation"])
        else:
            stigmation = Point()
        if "shift" in state_dict and state_dict["shift"] is not None:
            shift = FibsemRectangle.__from_dict__(state_dict["shift"])
        else:
            shift = Point()


        beam_settings = BeamSettings(
            beam_type=BeamType[state_dict["beam_type"].upper()],
            working_distance=state_dict["working_distance"],
            beam_current=state_dict["beam_current"],
            hfw=state_dict["hfw"],
            resolution=state_dict["resolution"],
            dwell_time=state_dict["dwell_time"],
            stigmation=stigmation,
            shift=shift,
            )

        return beam_settings
    


@dataclass
class MicroscopeState:

    """Data Class representing the state of a microscope with various parameters.

    Attributes:

        timestamp (float): A float representing the timestamp at which the state of the microscope was recorded. Defaults to the timestamp of the current datetime.
        absolute_position (FibsemStagePosition): An instance of FibsemStagePosition representing the current absolute position of the stage. Defaults to an empty instance of FibsemStagePosition.
        eb_settings (BeamSettings): An instance of BeamSettings representing the electron beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ELECTRON.
        ib_settings (BeamSettings): An instance of BeamSettings representing the ion beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ION.

    Methods:

        to_dict(self) -> dict: Converts the current state of the Microscope to a dictionary and returns it.
        from_dict(state_dict: dict) -> "MicroscopeState": Returns a new instance of MicroscopeState with attributes created from the passed dictionary.
    """

    timestamp: float = datetime.timestamp(datetime.now())
    absolute_position: FibsemStagePosition = FibsemStagePosition()
    eb_settings: BeamSettings = BeamSettings(beam_type=BeamType.ELECTRON)
    ib_settings: BeamSettings = BeamSettings(beam_type=BeamType.ION)

    def __to_dict__(self) -> dict:

        state_dict = {
            "timestamp": self.timestamp,
            "absolute_position": stage_position_to_dict(self.absolute_position) if self.absolute_position is not None else "Not defined",
            "eb_settings": self.eb_settings.__to_dict__() if self.eb_settings is not None else "Not defined",
            "ib_settings": self.ib_settings.__to_dict__() if self.ib_settings is not None else "Not defined",
        }

        return state_dict

    @staticmethod
    def __from_dict__(state_dict: dict) -> "MicroscopeState":
        microscope_state = MicroscopeState(
            timestamp=state_dict["timestamp"],
            absolute_position=stage_position_from_dict(state_dict["absolute_position"]),
            eb_settings=BeamSettings.__from_dict__(state_dict["eb_settings"]),
            ib_settings=BeamSettings.__from_dict__(state_dict["ib_settings"]),
        )

        return microscope_state

class FibsemPattern(Enum): # TODO: reanme to FibsemPatternType
    Rectangle = 1
    Line = 2
    Circle = 3

# TODO: convert this to a dataclass, rename to FibsemPattern
class FibsemPatternSettings: 
    '''
    FibsemPatternSettings is used to store all of the possible settings related to each pattern that may be drawn.
    
    Args:
        pattern (FibsemPattern): Used to indicate which pattern is utilised. Currently either Rectangle or Line.
        **kwargs: If FibsemPattern.Rectangle
                    width: float (m),
                    height: float (m), 
                    depth: float (m),
                    rotation: float = 0.0 (m), 
                    centre_x: float = 0.0 (m), 
                    centre_y: float = 0.0 (m),

                If FibsemPattern.Line
                    start_x: float (m), 
                    start_y: float (m), 
                    end_x: float (m), 
                    end_y: float (m), 
                    depth: float (m),
    '''
    def __init__(self, pattern: FibsemPattern = FibsemPattern.Rectangle, **kwargs):
        self.pattern = pattern
        if pattern == FibsemPattern.Rectangle:
            self.width = kwargs["width"]
            self.height = kwargs["height"]
            self.depth = kwargs["depth"]
            self.rotation = kwargs["rotation"] if "rotation" in kwargs else 0.0
            self.centre_x = kwargs["centre_x"] if "centre_x" in kwargs else 0.0
            self.centre_y = kwargs["centre_y"] if "centre_y" in kwargs else 0.0
            self.scan_direction= kwargs["scan_direction"] if "scan_direction" in kwargs else "TopToBottom"
            self.cleaning_cross_section= kwargs["cleaning_cross_section"] if "cleaning_cross_section" in kwargs else False
        elif pattern == FibsemPattern.Line:
            self.start_x = kwargs["start_x"]
            self.start_y = kwargs["start_y"]
            self.end_x = kwargs["end_x"]
            self.end_y = kwargs["end_y"]
            self.depth = kwargs["depth"]
            self.rotation = kwargs["rotation"] if "rotation" in kwargs else 0.0
            self.scan_direction= kwargs["scan_direction"] if "scan_direction" in kwargs else "TopToBottom"
            self.cleaning_cross_section= kwargs["cleaning_cross_section"] if "cleaning_cross_section" in kwargs else False
        elif pattern == FibsemPattern.Circle:
            self.centre_x = kwargs["centre_x"]
            self.centre_y = kwargs["centre_y"]
            self.radius = kwargs["radius"]
            self.depth = kwargs["depth"]
            self.start_angle = kwargs["start_angle"] if "start_angle" in kwargs else 0.0
            self.end_angle = kwargs["end_angle"] if "end_angle" in kwargs else 360.0
            self.rotation = kwargs["rotation"] if "rotation" in kwargs else 0.0
            self.scan_direction= kwargs["scan_direction"] if "scan_direction" in kwargs else "TopToBottom"
            self.cleaning_cross_section= kwargs["cleaning_cross_section"] if "cleaning_cross_section" in kwargs else False
    def __repr__(self) -> str:
        if self.pattern == FibsemPattern.Rectangle:
            return f"FibsemPatternSettings(pattern={self.pattern}, width={self.width}, height={self.height}, depth={self.depth}, rotation={self.rotation}, centre_x={self.centre_x}, centre_y={self.centre_y}, scan_direction={self.scan_direction}, cleaning_cross_section={self.cleaning_cross_section})"
        elif self.pattern == FibsemPattern.Line:
            return f"FibsemPatternSettings(pattern={self.pattern}, start_x={self.start_x}, start_y={self.start_y}, end_x={self.end_x}, end_y={self.end_y}, depth={self.depth}, rotation={self.rotation}, scan_direction={self.scan_direction}, cleaning_cross_section={self.cleaning_cross_section})"


    @staticmethod
    def __from_dict__(state_dict: dict) -> "FibsemPatternSettings":
        
        if state_dict["pattern"] == "Rectangle":
            return FibsemPatternSettings(
                pattern=FibsemPattern.Rectangle,
                width=state_dict["width"],
                height=state_dict["height"],
                depth=state_dict["depth"],
                rotation=state_dict["rotation"],
                centre_x=state_dict["centre_x"],
                centre_y=state_dict["centre_y"],
                scan_direction=state_dict["scan_direction"],
                cleaning_cross_section=state_dict["cleaning_cross_section"],
            )
        elif state_dict["pattern"] == "Line":
            return FibsemPatternSettings(
                pattern=FibsemPattern.Line,
                start_x=state_dict["start_x"],
                start_y=state_dict["start_y"],
                end_x=state_dict["end_x"],
                end_y=state_dict["end_y"],
                depth=state_dict["depth"],
                rotation=state_dict["rotation"],
                scan_direction=state_dict["scan_direction"],
                cleaning_cross_section=state_dict["cleaning_cross_section"],
            )
        elif state_dict["pattern"] == "Circle":
            return FibsemPatternSettings(
                pattern=FibsemPattern.Circle,
                centre_x=state_dict["centre_x"],
                centre_y=state_dict["centre_y"],
                radius=state_dict["radius"],
                depth=state_dict["depth"],
                start_angle=state_dict["start_angle"],
                end_angle=state_dict["end_angle"],
                rotation=state_dict["rotation"],
                scan_direction=state_dict["scan_direction"],
                cleaning_cross_section=state_dict["cleaning_cross_section"],
            )







@dataclass
class FibsemMillingSettings:
    """
    This class is used to store and retrieve settings for FIBSEM milling.

    Attributes:
    milling_current (float): The current used in the FIBSEM milling process. Default value is 20.0e-12 A.
    spot_size (float): The size of the beam spot used in the FIBSEM milling process. Default value is 5.0e-8 m.
    rate (float): The milling rate of the FIBSEM process. Default value is 3.0e-3 m^3/A/s.
    dwell_time (float): The dwell time of the beam at each point during the FIBSEM milling process. Default value is 1.0e-6 s.
    hfw (float): The high voltage field width used in the FIBSEM milling process. Default value is 150e-6 m.

    Methods:
    to_dict(): Converts the object attributes into a dictionary.
    from_dict(settings: dict) -> "FibsemMillingSettings": Creates a FibsemMillingSettings object from a dictionary of settings.
    """

    milling_current: float = 20.0e-12
    spot_size: float = 5.0e-8
    rate: float = 3.0e-3 # m3/A/s
    dwell_time: float = 1.0e-6 # s
    hfw: float = 150e-6
    patterning_mode: str = "Serial" 
    application_file: str = "Si"

    def __to_dict__(self) -> dict:

        settings_dict = {
            "milling_current": self.milling_current,
            "spot_size": self.spot_size,
            "rate": self.rate,
            "dwell_time": self.dwell_time,
            "hfw": self.hfw,
            "patterning_mode": self.patterning_mode,
            "application_file": self.application_file,
        }

        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict) -> "FibsemMillingSettings":

        milling_settings = FibsemMillingSettings(
            milling_current=settings.get("milling_current", 20.0e-12),
            spot_size=settings.get("spot_size", 5.0e-8),
            rate=settings.get("rate", 3.0e-3),
            dwell_time=settings.get("dwell_time", 1.0e-6),
            hfw=settings.get("hfw", 150e-6),
            patterning_mode=settings.get("patterning_mode", "Serial"),
            application_file=settings.get("application_file", "Si"),
        )

        return milling_settings


if THERMO:

    def save_needle_yaml(path: Path, position: ManipulatorPosition) -> None:
        """Save the manipulator position from disk"""
        from fibsem.structures import manipulator_position_to_dict

        with open(os.path.join(path, "needle.yaml"), "w") as f:
            yaml.dump(manipulator_position_to_dict(position), f, indent=4)

    def load_needle_yaml(path: Path) -> ManipulatorPosition:
        """Load the manipulator position from disk"""
        from fibsem.structures import manipulator_position_from_dict

        position_dict = load_yaml(os.path.join(path, "needle.yaml"))
        position = manipulator_position_from_dict(position_dict)

        return position


def stage_position_to_dict(stage_position: FibsemStagePosition) -> dict:
    """Converts the FibsemStagePosition Object into a dictionary"""

    stage_position_dict = {
        "x": stage_position.x,
        "y": stage_position.y,
        "z": stage_position.z,
        "r": stage_position.r,
        "t": stage_position.t,
        "coordinate_system": stage_position.coordinate_system,
    }

    return stage_position_dict


def stage_position_from_dict(state_dict: dict) -> FibsemStagePosition:
    """Converts a dictionary object to a fibsem stage position,
        dictionary must have correct keys"""

    stage_position = FibsemStagePosition(
        x=state_dict["x"],
        y=state_dict["y"],
        z=state_dict["z"],
        r=state_dict["r"],
        t=state_dict["t"],
        coordinate_system=state_dict["coordinate_system"],
    )

    return stage_position


if THERMO:

    def manipulator_position_to_dict(position: ManipulatorPosition) -> dict:

        position_dict = {
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "r": None,
            "coordinate_system": position.coordinate_system,
        }

        return position_dict

    def manipulator_position_from_dict(position_dict: dict) -> ManipulatorPosition:

        position = ManipulatorPosition(
            x=position_dict["x"],
            y=position_dict["y"],
            z=position_dict["z"],
            r=position_dict["r"],
            coordinate_system=position_dict["coordinate_system"],
        )

        return position


@dataclass
class BeamSystemSettings:
    """
    A data class that represents the settings of a beam system.

    Attributes:
        beam_type (BeamType): The type of beam used in the system (Electron or Ion).
        voltage (float): The voltage used in the system.
        current (float): The current used in the system.
        detector_type (str): The type of detector used in the system.
        detector_mode (str): The mode of the detector used in the system.
        eucentric_height (float): The eucentric height of the system.
        plasma_gas (str, optional): The type of plasma gas used in the system.

    Methods:
        __to_dict__(self) -> dict:
            Converts the instance variables to a dictionary.
        __from_dict__(settings: dict, beam_type: BeamType) -> BeamSystemSettings:
            Creates an instance of the class from a dictionary and a beam type.
    """

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
    def __from_dict__(settings: dict, beam_type: BeamType) -> "BeamSystemSettings":

        if "plasma_gas" not in settings:
            settings["plasma_gas"] = "NULL"

        system_settings = BeamSystemSettings(
            beam_type=beam_type,
            voltage=settings["voltage"],
            current=settings["current"],
            detector_type=settings["detector_type"],
            detector_mode=settings["detector_mode"],
            eucentric_height=settings["eucentric_height"],
            plasma_gas=settings["plasma_gas"].capitalize(),
        )

        return system_settings


# TODO: change this to use pretilt_angle, flat_to_electron, and flat_to_ion tilts, for better separation
@dataclass
class StageSettings:
    """
    A data class representing the settings for the stage.

    Attributes:
    rotation_flat_to_electron (float): The rotation from flat to electron in degrees.
    rotation_flat_to_ion (float): The rotation from flat to ion in degrees.
    tilt_flat_to_electron (float): The tilt from flat to electron in degrees.
    tilt_flat_to_ion (float): The tilt from flat to ion in degrees.
    pre_tilt (float): The pre-tilt in degrees.
    needle_stage_height_limit (float): The height limit of the needle stage in meters.

    Methods:
    __to_dict__() -> dict: Returns the settings as a dictionary.
    __from_dict__(settings: dict) -> "StageSettings": Returns an instance of StageSettings from a dictionary of its settings.
    """
    rotation_flat_to_electron: float = 50  # degrees
    rotation_flat_to_ion: float = 230  # degrees
    tilt_flat_to_electron: float = 27  # degrees (pre_tilt)
    tilt_flat_to_ion: float = 52  # degrees
    pre_tilt: float = 35  # degrees
    needle_stage_height_limit: float = 3.7e-3

    def __to_dict__(self) -> dict:

        settings = {
            "rotation_flat_to_electron": self.rotation_flat_to_electron,
            "rotation_flat_to_ion": self.rotation_flat_to_ion,
            "tilt_flat_to_electron": self.tilt_flat_to_electron,
            "tilt_flat_to_ion": self.tilt_flat_to_ion,
            "pre_tilt": self.pre_tilt,
            "needle_stage_height_limit": self.needle_stage_height_limit,
        }
        return settings

    @staticmethod
    def __from_dict__(settings: dict) -> "StageSettings":

        stage_settings = StageSettings(
            rotation_flat_to_electron=settings["rotation_flat_to_electron"],
            rotation_flat_to_ion=settings["rotation_flat_to_ion"],
            tilt_flat_to_electron=settings["tilt_flat_to_electron"],
            tilt_flat_to_ion=settings["tilt_flat_to_ion"],
            pre_tilt=settings["pre_tilt"],
            needle_stage_height_limit=settings["needle_stage_height_limit"],
        )

        return stage_settings


@dataclass
class SystemSettings:

    """
    Dataclass representing the system settings for the FIB-SEM instrument.

    :param ip_address: IP address of the instrument.
    :param stage: settings for the stage.
    :param ion: settings for the ion beam.
    :param electron: settings for the electron beam.
    :param manufacturer: name of the instrument manufacturer.

    :return: a new instance of `SystemSettings`.
    """

    ip_address: str = "10.0.0.1"
    stage: StageSettings = None
    ion: BeamSystemSettings = None
    electron: BeamSystemSettings = None
    manufacturer: str = None

    def __to_dict__(self) -> dict:

        settings_dict = {
            "ip_address": self.ip_address,
            "stage": self.stage.__to_dict__(),
            "ion": self.ion.__to_dict__(),
            "electron": self.electron.__to_dict__(),
            "manufacturer": self.manufacturer,
        }

        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict) -> "SystemSettings":

        system_settings = SystemSettings(
            ip_address=settings["ip_address"],
            stage=StageSettings.__from_dict__(settings["stage"]),
            ion=BeamSystemSettings.__from_dict__(settings["ion"], BeamType.ION),
            electron=BeamSystemSettings.__from_dict__(
                settings["electron"], BeamType.ELECTRON
            ),
            manufacturer=settings["manufacturer"],
        )

        return system_settings


@dataclass
class DefaultSettings:
    """
    Default settings for the imaging and milling current 
    """
    imaging_current: float = 20.0e-12
    milling_current: float = 2.0e-9

    @staticmethod
    def __from_dict__(settings: dict) -> "DefaultSettings":

        default_settings = DefaultSettings(
            imaging_current=settings["imaging_current"],
            milling_current=settings["milling_current"],
        )
        return default_settings


@dataclass
class MicroscopeSettings:

    """
    A data class representing the settings for a microscope system.

    Attributes:
        system (SystemSettings): An instance of the `SystemSettings` class that holds the system settings.
        image (ImageSettings): An instance of the `ImageSettings` class that holds the image settings.
        protocol (dict, optional): A dictionary representing the protocol settings. Defaults to None.
        milling (FibsemMillingSettings, optional): An instance of the `FibsemMillingSettings` class that holds the fibsem milling settings. Defaults to None.

    Methods:
        __to_dict__(): Returns a dictionary representation of the `MicroscopeSettings` object.
        __from_dict__(settings: dict, protocol: dict = None) -> "MicroscopeSettings": Returns an instance of the `MicroscopeSettings` class from a dictionary.
    """

    system: SystemSettings
    image: ImageSettings
    protocol: dict = None
    milling: FibsemMillingSettings = None
    

    def __to_dict__(self) -> dict:

        settings_dict = {
            "system": self.system.__to_dict__(),
            "user": self.image.__to_dict__(),

        }

        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict, protocol: dict = None) -> "MicroscopeSettings":

        return MicroscopeSettings(
            system=SystemSettings.__from_dict__(settings["system"]),
            image=ImageSettings.__from_dict__(settings["user"]),
            protocol=protocol,

        )


# state
from abc import ABC, abstractmethod, abstractstaticmethod

# TODO: convert to ABC
class FibsemStage(Enum):
    Base = 1


@dataclass
class FibsemState:
    """
    FibsemState data class that represents the current state of FIBSEM system 

    Attributes:
    stage (FibsemStage): The current stage of the autoliftout workflow, as a `FibsemStage` enum member.
    microscope_state (MicroscopeState): The current state of the microscope, as a `MicroscopeState` object.
    start_timestamp (float): The timestamp when the autoliftout workflow began, as a Unix timestamp.
    end_timestamp (float): The timestamp when the autoliftout workflow ended, as a Unix timestamp.

    Methods:
    __to_dict__(): Serializes the `FibsemState` object to a dictionary.
    __from_dict__(state_dict: dict) -> FibsemState: Deserializes a dictionary to a `FibsemState` object.

    """

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
    def __from_dict__(self, state_dict: dict) -> "FibsemState":

        autoliftout_state = FibsemState(
            stage=FibsemState[state_dict["stage"]],
            microscope_state=MicroscopeState.__from_dict__(
                state_dict["microscope_state"]
            ),
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
                absolute_position=FibsemStagePosition(),
                eb_settings=BeamSettings.__from_dict__(
                    settings["microscope_state"]["eb_settings"]
                ),
                ib_settings=BeamSettings.__from_dict__(
                    settings["microscope_state"]["ib_settings"]
                ),
            )

        metadata = FibsemImageMetadata(
            image_settings=image_settings,
            version=version,
            pixel_size=pixel_size,
            microscope_state=microscope_state,
        )
        return metadata

    if THERMO:

        def image_settings_from_adorned(
            image=AdornedImage, beam_type: BeamType = BeamType.ELECTRON
        ) -> ImageSettings:

            from fibsem.utils import current_timestamp

            image_settings = ImageSettings(
                resolution=[image.width, image.height],
                dwell_time=image.metadata.scan_settings.dwell_time,
                hfw=image.width * image.metadata.binary_result.pixel_size.x,
                autocontrast=True,
                beam_type=beam_type,
                gamma_enabled=True,
                save=False,
                save_path="path",
                label=current_timestamp(),
                reduced_area=None,
            )
            return image_settings

    def compare_image_settings(self, image_settings: ImageSettings) -> bool:
        """Compares image settings to the metadata image settings.

        Args:
            image_settings (ImageSettings): Image settings to compare to.

        Returns:
            bool: True if the image settings match the metadata image settings.
        """
        assert (
            self.image_settings.resolution == image_settings.resolution
        ), f"resolution: {self.image_settings.resolution} != {image_settings.resolution}"
        assert (
            self.image_settings.dwell_time == image_settings.dwell_time
        ), f"dwell_time: {self.image_settings.dwell_time} != {image_settings.dwell_time}"
        assert (
            self.image_settings.hfw == image_settings.hfw
        ), f"hfw: {self.image_settings.hfw} != {image_settings.hfw}"
        assert (
            self.image_settings.autocontrast == image_settings.autocontrast
        ), f"autocontrast: {self.image_settings.autocontrast} != {image_settings.autocontrast}"
        assert (
            self.image_settings.beam_type.value == image_settings.beam_type.value
        ), f"beam_type: {self.image_settings.beam_type.value} != {image_settings.beam_type.value}"
        assert (
            self.image_settings.gamma_enabled== image_settings.gamma_enabled
        ), f"gamma: {self.image_settings.gamma_enabled} != {image_settings.gamma_enabled}"
        assert (
            self.image_settings.save == image_settings.save
        ), f"save: {self.image_settings.save} != {image_settings.save}"
        assert (
            self.image_settings.save_path == image_settings.save_path
        ), f"save_path: {self.image_settings.save_path} != {image_settings.save_path}"
        assert (
            self.image_settings.label == image_settings.label
        ), f"label: {self.image_settings.label} != {image_settings.label}"
        assert (
            self.image_settings.reduced_area == image_settings.reduced_area
        ), f"reduced_area: {self.image_settings.reduced_area} != {image_settings.reduced_area}"

        return True


class FibsemImage:
    
    """
    Class representing a FibsemImage and its associated metadata. 
    Has in built methods to deal with image types of TESCAN and ThermoFisher API 

    Args:
        data (np.ndarray): The image data stored in a numpy array.
        metadata (FibsemImageMetadata, optional): The metadata associated with the image. Defaults to None.

    Methods:
        load(cls, tiff_path: str) -> "FibsemImage":
            Loads a FibsemImage from a tiff file.

            Args:
                tiff_path (path): path to the tif* file

            Returns:
                FibsemImage: instance of FibsemImage

        save(self, save_path: Path) -> None:
            Saves a FibsemImage to a tiff file.

            Inputs:
                save_path (path): path to save directory and filename
    """

    def __init__(self, data: np.ndarray, metadata: FibsemImageMetadata = None):

        if check_data_format(data):
            self.data = data
        else:
            raise Exception("Invalid Data format for Fibsem Image")
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

    if THERMO:

        @classmethod
        def fromAdornedImage(
            cls,
            adorned: AdornedImage,
            image_settings: ImageSettings,
            state: MicroscopeState = None,
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
                    absolute_position=FibsemStagePosition(
                        adorned.metadata.stage_settings.stage_position.x,
                        adorned.metadata.stage_settings.stage_position.y,
                        adorned.metadata.stage_settings.stage_position.z,
                        adorned.metadata.stage_settings.stage_position.r,
                        adorned.metadata.stage_settings.stage_position.t,
                    ),
                    eb_settings=BeamSettings(beam_type=BeamType.ELECTRON),
                    ib_settings=BeamSettings(beam_type=BeamType.ION),
                )
            else:
                state.timestamp = adorned.metadata.acquisition.acquisition_datetime

            pixel_size = Point(
                adorned.metadata.binary_result.pixel_size.x,
                adorned.metadata.binary_result.pixel_size.y,
            )
            metadata = FibsemImageMetadata(
                image_settings=image_settings,
                pixel_size=pixel_size,
                microscope_state=state,
            )
            return cls(data=adorned.data, metadata=metadata)

    if TESCAN:

        @classmethod
        def fromTescanImage(
            cls,
            image: Document,
            image_settings: ImageSettings,
            state: MicroscopeState,
        ) -> "FibsemImage":
            """Creates FibsemImage from an AdornedImage (microscope output format).

            Args:
                adorned (AdornedImage): Adorned Image from microscope
                metadata (FibsemImageMetadata, optional): metadata extracted from microscope output. Defaults to None.

            Returns:
                FibsemImage: instance of FibsemImage from AdornedImage
            """

            pixel_size = Point(
                float(image.Header["MAIN"]["PixelSizeX"]),
                float(image.Header["MAIN"]["PixelSizeY"]),
            )
            metadata = FibsemImageMetadata(
                image_settings=image_settings,
                pixel_size=pixel_size,
                microscope_state=state,
            )
            return cls(data=np.array(image.Image), metadata=metadata)


@dataclass
class ReferenceImages:
    low_res_eb: FibsemImage
    high_res_eb: FibsemImage
    low_res_ib: FibsemImage
    high_res_ib: FibsemImage

    def __iter__(self) -> list[FibsemImage]:

        yield self.low_res_eb, self.high_res_eb, self.low_res_ib, self.high_res_ib


def check_data_format(data: np.ndarray) -> bool:
    """Checks that data is in the correct format."""
    # assert data.ndim == 2  # or data.ndim == 3
    # assert data.dtype in [np.uint8, np.uint16]
    # if data.ndim == 3 and data.shape[2] == 1:
    #     data = data[:, :, 0]
    return data.ndim == 2 and data.dtype in [np.uint8, np.uint16]


@dataclass
class FibsemDetectorSettings:
    type: str
    mode: str 
    brightness: float
    contrast: float

    if TESCAN:
        def to_tescan(self):
            """Converts to tescan format."""
            tescan_brightness = self.brightness * 100
            tescan_contrast = self.contrast * 100
            return tescan_brightness, tescan_contrast