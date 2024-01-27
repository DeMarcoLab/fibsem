# fibsem structures

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from fibsem.config import SUPPORTED_COORDINATE_SYSTEMS
from typing import Optional
import numpy as np
import tifffile as tff
import fibsem
from fibsem.config import METADATA_VERSION
from abc import ABC, abstractmethod, abstractstaticmethod

try:
    from tescanautomation.Common import Document

    TESCAN = True
except:
    TESCAN = False

try:
    sys.path.append("C:\Program Files\Thermo Scientific AutoScript")
    sys.path.append(
        "C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages"
    )
    sys.path.append("C:\Program Files\Python36\envs\AutoScript")
    sys.path.append("C:\Program Files\Python36\envs\AutoScript\Lib\site-packages")
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        ManipulatorPosition,
        Rectangle,
        StagePosition,
    )

    THERMO = True
except:
    THERMO = False



# @patrickcleeve: dataclasses.asdict -> :(


# TODO: overload constructors instead of from_dict...
@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @staticmethod
    def from_dict(d: dict) -> "Point":
        x = float(d["x"])
        y = float(d["y"])
        return Point(x, y)

    def to_list(self) -> list:
        return [self.x, self.y]

    @staticmethod
    def from_list(l: list) -> "Point":
        x = float(l[0])
        y = float(l[1])
        return Point(x, y)

    def __add__(self, other) -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("Index out of range")

    def _to_metres(self, pixel_size: float) -> "Point":
        return Point(self.x * pixel_size, self.y * pixel_size)

    def _to_pixels(self, pixel_size: float) -> "Point":
        return Point(self.x / pixel_size, self.y / pixel_size)

    def distance(self, other: "Point") -> "Point":
        """Calculate the distance between two points. (other - self)"""
        return Point(x=(other.x - self.x), y=(other.y - self.y))

    def euclidean(self, other: "Point") -> float:
        """Calculate the euclidean distance between two points."""
        return np.linalg.norm(self.distance(other).to_list())


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
    Vertical = 2
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
        to_dict(): Convert the stage position object to a dictionary.
        from_dict(data: dict): Create a new stage position object from a dictionary.
        to_autoscript_position(stage_tilt: float = 0.0) -> StagePosition: Convert the stage position to a StagePosition object that is compatible with Autoscript.
        from_autoscript_position(position: StagePosition, stage_tilt: float = 0.0) -> None: Create a new FibsemStagePosition object from a StagePosition object that is compatible with Autoscript.
        to_tescan_position(stage_tilt: float = 0.0): Convert the stage position to a format that is compatible with Tescan.
        from_tescan_position(): Create a new FibsemStagePosition object from a Tescan-compatible stage position.
    """

    name: str = None
    x: float = None
    y: float = None
    z: float = None
    r: float = None
    t: float = None
    coordinate_system: str = None

    def to_dict(self) -> dict:
        position_dict = {}

        position_dict["name"] = self.name if self.name is not None else None
        position_dict["x"] = float(self.x) if self.x is not None else None
        position_dict["y"] = float(self.y) if self.y is not None else None
        position_dict["z"] = float(self.z) if self.z is not None else None
        position_dict["r"] = float(self.r) if self.r is not None else None
        position_dict["t"] = float(self.t) if self.t is not None else None
        position_dict["coordinate_system"] = self.coordinate_system

        return position_dict

    @classmethod
    def from_dict(cls, data: dict) -> "FibsemStagePosition":
        items = ["x", "y", "z", "r", "t"]

        for item in items:
            value = data[item]

            assert isinstance(value, float) or isinstance(value, int) or value is None

        return cls(
            name=data.get("name", None),
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
                y=self.y,  # / np.cos(stage_tilt),
                z=self.z,  # / np.cos(stage_tilt),
                r=self.r,
                t=self.t,
                coordinate_system=self.coordinate_system,
            )

        @classmethod
        def from_autoscript_position(
            cls, position: StagePosition, stage_tilt: float = 0.0
        ) -> None:
            return cls(
                x=position.x,
                y=position.y,  # * np.cos(stage_tilt),
                z=position.z,  # * np.cos(stage_tilt),
                r=position.r,
                t=position.t,
                coordinate_system=position.coordinate_system.upper(),
            )

    if TESCAN:

        def to_tescan_position(self, stage_tilt: float = 0.0):
            self.y = self.y  # / np.cos(stage_tilt),

        @classmethod
        def from_tescan_position(self, stage_tilt: float = 0.0):
            self.y = self.y  # * np.cos(stage_tilt)

    def __add__(self, other: "FibsemStagePosition") -> "FibsemStagePosition":
        return FibsemStagePosition(
            x=self.x + other.x if other.x is not None else self.x,
            y=self.y + other.y if other.y is not None else self.y,
            z=self.z + other.z if other.z is not None else self.z,
            r=self.r + other.r if other.r is not None else self.r,
            t=self.t + other.t if other.t is not None else self.t,
            coordinate_system=self.coordinate_system,
        )

    def __sub__(self, other: "FibsemStagePosition") -> "FibsemStagePosition":
        return FibsemStagePosition(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
            r=self.r - other.r,
            t=self.t - other.t,
            coordinate_system=self.coordinate_system,
        )

    def _scale_repr(self, scale: float, precision: int = 2):
        return f"x:{self.x*scale:.{precision}f}, y:{self.y*scale:.{precision}f}, z:{self.z*scale:.{precision}f}"



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
        to_dict(): Convert the manipulator position object to a dictionary.
        from_dict(data: dict): Create a new manipulator position object from a dictionary.
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
    coordinate_system: str = "RAW"

    def __post_init__(self):
        assert (
            isinstance(self.coordinate_system, str) or self.coordinate_system is None
        ), f"unsupported type {type(self.coordinate_system)} for coorindate system"
        assert (
            self.coordinate_system in SUPPORTED_COORDINATE_SYSTEMS
            or self.coordinate_system is None
        ), f"coordinate system value {self.coordinate_system} is unsupported or invalid syntax. Must be RAW or SPECIMEN"

    def to_dict(self) -> dict:
        position_dict = {}
        position_dict["x"] = self.x
        position_dict["y"] = self.y
        position_dict["z"] = self.z
        position_dict["r"] = self.r
        position_dict["t"] = self.t
        position_dict["coordinate_system"] = self.coordinate_system.upper()

        return position_dict

    @classmethod
    def from_dict(cls, data: dict) -> "FibsemManipulatorPosition":
        items = ["x", "y", "z", "r", "t"]

        for item in items:
            value = data[item]

            assert isinstance(value, float) or isinstance(value, int) or value is None

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
            if self.coordinate_system == "RAW":
                coordinate_system = "Raw"
            elif self.coordinate_system == "STAGE":
                coordinate_system = "Stage"
            return ManipulatorPosition(
                x=self.x,
                y=self.y,
                z=self.z,
                r=None,  # TODO figure this out, do we need it for real micrscope or just simulator ?
                # r=None,
                coordinate_system=coordinate_system,
            )

        @classmethod
        def from_autoscript_position(cls, position: ManipulatorPosition) -> None:
            return cls(
                x=position.x,
                y=position.y,
                z=position.z,
                coordinate_system=position.coordinate_system.upper(),
            )

    if TESCAN:

        def to_tescan_position(self):
            pass

        @classmethod
        def from_tescan_position(self):
            pass

    def __add__(
        self, other: "FibsemManipulatorPosition"
    ) -> "FibsemManipulatorPosition":
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

    def __post_init__(self):
        assert isinstance(self.left, float) or isinstance(
            self.left, int
        ), f"type {type(self.left)} is unsupported for left, must be int or floar"
        assert isinstance(self.top, float) or isinstance(
            self.top, int
        ), f"type {type(self.top)} is unsupported for top, must be int or floar"
        assert isinstance(self.width, float) or isinstance(
            self.width, int
        ), f"type {type(self.width)} is unsupported for width, must be int or floar"
        assert isinstance(self.height, float) or isinstance(
            self.height, int
        ), f"type {type(self.height)} is unsupported for height, must be int or floar"

    def from_dict(settings: dict) -> "FibsemRectangle":
        if settings is None:
            return None
        points = ["left", "top", "width", "height"]

        for point in points:
            value = settings[point]

            assert isinstance(value, float) or isinstance(value, int) or value is None

        return FibsemRectangle(
            left=settings["left"],
            top=settings["top"],
            width=settings["width"],
            height=settings["height"],
        )

    def to_dict(self) -> dict:
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
        filename (str): The filename to use when saving the acquired image.
        autogamma (bool): Whether or not to apply gamma correction to the acquired image.
        path (Path): The path to the directory where the acquired image should be saved.
        reduced_area (FibsemRectangle): The rectangular region of interest within the acquired image, if any.

    Methods:
        from_dict(settings: dict) -> ImageSettings:
            Converts a dictionary of image settings to an ImageSettings object.
        to_dict() -> dict:
            Converts the ImageSettings object to a dictionary of image settings.
    """

    resolution: list = None
    dwell_time: float = None
    hfw: float = None
    autocontrast: bool = None
    beam_type: BeamType = None
    save: bool = None
    filename: str = None
    autogamma: bool = None
    path: Path = None
    reduced_area: FibsemRectangle = None
    line_integration: int = None  # (int32) 2 - 255
    scan_interlacing: int = None  # (int32) 2 - 8
    frame_integration: int = None  # (int32) 2 - 512
    drift_correction: bool = False  # (bool) # requires frame_integration > 1

    def __post_init__(self):
        assert (
            isinstance(self.resolution, (list, tuple)) or self.resolution is None
        ), f"resolution must be a list, currently is {type(self.resolution)}"
        assert (
            isinstance(self.dwell_time, float) or self.dwell_time is None
        ), f"dwell time must be of type float, currently is {type(self.dwell_time)}"
        assert (
            isinstance(self.hfw, float) or isinstance(self.hfw, int) or self.hfw is None
        ), f"hfw must be int or float, currently is {type(self.hfw)}"
        assert (
            isinstance(self.autocontrast, bool) or self.autocontrast is None
        ), f"autocontrast setting must be bool, currently is {type(self.autocontrast)}"
        assert (
            isinstance(self.beam_type, BeamType) or self.beam_type is None
        ), f"beam type must be a BeamType object, currently is {type(self.beam_type)}"
        assert (
            isinstance(self.save, bool) or self.save is None
        ), f"save option must be a bool, currently is {type(self.save)}"
        assert (
            isinstance(self.filename, str) or self.filename is None
        ), f"filename must b str, currently is {type(self.filename)}"
        assert (
            isinstance(self.autogamma, bool) or self.autogamma is None
        ), f"gamma enabled setting must be bool, currently is {type(self.autogamma)}"
        assert (
            isinstance(self.path, (Path, str)) or self.path is None
        ), f"save path must be Path or str, currently is {type(self.path)}"
        assert (
            isinstance(self.reduced_area, FibsemRectangle) or self.reduced_area is None
        ), f"reduced area must be a fibsemRectangle object, currently is {type(self.reduced_area)}"

    @staticmethod
    def from_dict(settings: dict) -> "ImageSettings":
        if "reduced_area" in settings and settings["reduced_area"] is not None:
            reduced_area = FibsemRectangle.from_dict(settings["reduced_area"])
        else:
            reduced_area = None

        image_settings = ImageSettings(
            resolution=settings.get("resolution", (1536, 1024)),
            dwell_time=settings.get("dwell_time", 1.0e-6),
            hfw=settings.get("hfw", 150e-6),
            autocontrast=settings.get("autocontrast", False),
            beam_type=BeamType[settings.get("beam_type", "Electron").upper()],
            autogamma=settings.get("autogamma", False),
            save=settings.get("save", False),
            path=settings.get("path", os.getcwd()),
            filename=settings.get("filename", "default_image"),
            reduced_area=reduced_area,
            line_integration=settings.get("line_integration", None),
            scan_interlacing=settings.get("scan_interlacing", None),
            frame_integration=settings.get("frame_integration", None),
            drift_correction=settings.get("drift_correction", False),
        )

        return image_settings

    def to_dict(self) -> dict:
        settings_dict = {
            "beam_type": self.beam_type.name if self.beam_type is not None else None,
            "resolution": self.resolution if self.resolution is not None else None,
            "dwell_time": self.dwell_time if self.dwell_time is not None else None,
            "hfw": self.hfw if self.hfw is not None else None,
            "autocontrast": self.autocontrast
            if self.autocontrast is not None
            else None,
            "autogamma": self.autogamma
            if self.autogamma is not None
            else None,
            "save": self.save if self.save is not None else None,
            "path": str(self.path) if self.path is not None else None,
            "filename": self.filename if self.filename is not None else None,
            "reduced_area": {
                "left": self.reduced_area.left,
                "top": self.reduced_area.top,
                "width": self.reduced_area.width,
                "height": self.reduced_area.height,
            }
            if self.reduced_area is not None
            else None,
            "line_integration": self.line_integration,
            "scan_interlacing": self.scan_interlacing,
            "frame_integration": self.frame_integration,
            "drift_correction": self.drift_correction,
        }

        return settings_dict

    @staticmethod
    def fromFibsemImage(image: "FibsemImage") -> "ImageSettings":
        """Returns the image settings for a FibsemImage object.

        Args:
            image (FibsemImage): The FibsemImage object to get the image settings from.

        Returns:
            ImageSettings: The image settings for the given FibsemImage object.
        """
        from fibsem import utils
        from copy import deepcopy

        image_settings = deepcopy(image.metadata.image_settings)
        image_settings.filename = utils.current_timestamp()
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
        to_dict(): Returns a dictionary representation of the object.
        from_dict(state_dict: dict) -> BeamSettings: Returns a new BeamSettings object created from a dictionary.

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
    scan_rotation: float = None

    def __post_init__(self):
        assert (
            self.beam_type in [BeamType.ELECTRON, BeamType.ION]
            or self.beam_type is None
        ), f"beam_type must be instance of BeamType, currently {type(self.beam_type)}"
        assert (
            isinstance(self.working_distance, (float, int))
            or self.working_distance is None
        ), f"Working distance must be float or int, currently is {type(self.working_distance)}"
        assert (
            isinstance(self.beam_current, (float, int)) or self.beam_current is None
        ), f"beam current must be float or int, currently is {type(self.beam_current)}"
        assert (
            isinstance(self.voltage, (float, int)) or self.voltage is None
        ), f"voltage must be float or int, currently is {type(self.voltage)}"
        assert (
            isinstance(self.hfw, (float, int)) or self.hfw is None
        ), f"horizontal field width (HFW) must be float or int, currently is {type(self.hfw)}"
        assert (
            isinstance(self.resolution, list) or self.resolution is None
        ), f"resolution must be a list, currently is {type(self.resolution)}"
        assert (
            isinstance(self.dwell_time, (float, int)) or self.dwell_time is None
        ), f"dwell_time must be float or int, currently is {type(self.dwell_time)}"
        assert (
            isinstance(self.stigmation, Point) or self.stigmation is None
        ), f"stigmation must be a Point instance, currently is {type(self.stigmation)}"
        assert (
            isinstance(self.shift, Point) or self.shift is None
        ), f"shift must be a Point instance, currently is {type(self.shift)}"


    def to_dict(self) -> dict:
        state_dict = {
            "beam_type": self.beam_type.name,
            "working_distance": self.working_distance,
            "beam_current": self.beam_current,
            "voltage": self.voltage,
            "hfw": self.hfw,
            "resolution": self.resolution,
            "dwell_time": self.dwell_time,
            "stigmation": self.stigmation.to_dict()
            if self.stigmation is not None
            else None,
            "shift": self.shift.to_dict() if self.shift is not None else None,
            "scan_rotation": self.scan_rotation,
        }

        return state_dict

    @staticmethod
    def from_dict(state_dict: dict) -> "BeamSettings":
        if "stigmation" in state_dict and state_dict["stigmation"] is not None:
            stigmation = Point.from_dict(state_dict["stigmation"])
        else:
            stigmation = Point()
        if "shift" in state_dict and state_dict["shift"] is not None:
            shift = Point.from_dict(state_dict["shift"])
        else:
            shift = Point()
        
        wd = state_dict.get("working_distance", state_dict.get("eucentric_height", None))
        current = state_dict.get("beam_current", state_dict.get("current", None))

        beam_settings = BeamSettings(
            beam_type=BeamType[state_dict["beam_type"].upper()],
            working_distance=wd,
            beam_current=current,
            voltage=state_dict["voltage"],
            hfw=state_dict["hfw"],
            resolution=state_dict["resolution"],
            dwell_time=state_dict["dwell_time"],
            stigmation=stigmation,
            shift=shift,
            scan_rotation=state_dict.get("scan_rotation", 0.0),
        )

        return beam_settings


@dataclass
class FibsemDetectorSettings:
    type: str = None
    mode: str = None
    brightness: float = 0.5
    contrast: float = 0.5

    def __post_init__(self):
        assert (
            isinstance(self.type, str) or self.type is None
        ), f"type must be input as str, currently is {type(self.type)}"
        assert (
            isinstance(self.mode, str) or self.mode is None
        ), f"mode must be input as str, currently is {type(self.mode)}"
        assert (
            isinstance(self.brightness, (float, int)) or self.brightness is None
        ), f"brightness must be int or float value, currently is {type(self.brightness)}"
        assert (
            isinstance(self.contrast, (float, int)) or self.contrast is None
        ), f"contrast must be int or float value, currently is {type(self.contrast)}"

    if TESCAN:

        def to_tescan(self):
            """Converts to tescan format."""
            tescan_brightness = self.brightness * 100
            tescan_contrast = self.contrast * 100
            return tescan_brightness, tescan_contrast

    def to_dict(self) -> dict:
        """Converts to a dictionary."""
        return {
            "type": self.type,
            "mode": self.mode,
            "brightness": self.brightness,
            "contrast": self.contrast,
        }

    @staticmethod
    def from_dict(settings: dict) -> "FibsemDetectorSettings":
        """Converts from a dictionary."""
        return FibsemDetectorSettings(
            type=settings.get("type", "Unknown"),
            mode=settings.get("mode", "Unknown"),
            brightness=settings.get("brightness", 0.0),
            contrast=settings.get("contrast", 0.0),
        )


@dataclass
class MicroscopeState:

    """Data Class representing the state of a microscope with various parameters.

    Attributes:

        timestamp (float): A float representing the timestamp at which the state of the microscope was recorded. Defaults to the timestamp of the current datetime.
        stage_position (FibsemStagePosition): An instance of FibsemStagePosition representing the current absolute position of the stage. Defaults to an empty instance of FibsemStagePosition.
        electron_beam (BeamSettings): An instance of BeamSettings representing the electron beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ELECTRON.
        ion_beam (BeamSettings): An instance of BeamSettings representing the ion beam settings. Defaults to an instance of BeamSettings with beam_type set to BeamType.ION.

    Methods:

        to_dict(self) -> dict: Converts the current state of the Microscope to a dictionary and returns it.
        from_dict(state_dict: dict) -> "MicroscopeState": Returns a new instance of MicroscopeState with attributes created from the passed dictionary.
    """

    timestamp: float = datetime.timestamp(datetime.now())
    stage_position: FibsemStagePosition = FibsemStagePosition()
    electron_beam: BeamSettings = BeamSettings(beam_type=BeamType.ELECTRON)
    ion_beam: BeamSettings = BeamSettings(beam_type=BeamType.ION)
    electron_detector: FibsemDetectorSettings = FibsemDetectorSettings()
    ion_detector: FibsemDetectorSettings = FibsemDetectorSettings()

    def __post_init__(self):
        assert (
            isinstance(self.stage_position, FibsemStagePosition)
            or self.stage_position is None
        ), f"absolute position must be of type FibsemStagePosition, currently is {type(self.stage_position)}"
        assert (
            isinstance(self.electron_beam, BeamSettings) or self.electron_beam is None
        ), f"electron_beam must be of type BeamSettings, currently is {type(self.electron_beam)}"
        assert (
            isinstance(self.ion_beam, BeamSettings) or self.ion_beam is None
        ), f"ion_beam must be of type BeamSettings, currently us {type(self.ion_beam)}"
        assert (
            isinstance(self.electron_detector, FibsemDetectorSettings)
            or self.electron_detector is None
        ), f"electron_detector must be of type FibsemDetectorSettings, currently is {type(self.electron_detector)}"
        assert (
            isinstance(self.ion_detector, FibsemDetectorSettings)
            or self.ion_detector is None
        ), f"ion_detector must be of type FibsemDetectorSettings, currently is {type(self.ion_detector)}"

    def to_dict(self) -> dict:
        state_dict = {
            "timestamp": self.timestamp,
            "stage_position": self.stage_position.to_dict()
            if self.stage_position is not None
            else "Not defined",
            "electron_beam": self.electron_beam.to_dict()
            if self.electron_beam is not None
            else "Not defined",
            "ion_beam": self.ion_beam.to_dict()
            if self.ion_beam is not None
            else "Not defined",
            "electron_detector": self.electron_detector.to_dict()
            if self.electron_detector is not None
            else "Not defined",
            "ion_detector": self.ion_detector.to_dict()
            if self.ion_detector is not None
            else "Not defined",
        }

        return state_dict

    @staticmethod
    def from_dict(state_dict: dict) -> "MicroscopeState":
        microscope_state = MicroscopeState(
            timestamp=state_dict["timestamp"],
            stage_position=FibsemStagePosition.from_dict(
                state_dict["stage_position"]
            ),
            electron_beam=BeamSettings.from_dict(state_dict["electron_beam"]),
            ion_beam=BeamSettings.from_dict(state_dict["ion_beam"]),
            electron_detector=FibsemDetectorSettings.from_dict(
                state_dict.get("electron_detector", {})
            ),
            ion_detector=FibsemDetectorSettings.from_dict(
                state_dict.get("ion_detector", {})
            ),
        )

        return microscope_state



########### Base Pattern Settings
@dataclass
class FibsemPatternSettings(ABC):

    @abstractmethod
    def to_dict(self) -> dict:
        pass
    
    @staticmethod
    def from_dict(self, data: dict) -> "FibsemPatternSettings":
        pass


class CrossSectionPattern(Enum):
    Rectangle  = auto()
    RegularCrossSection = auto()
    CleaningCrossSection = auto()

@dataclass
class FibsemRectangleSettings(FibsemPatternSettings):
    width: float
    height: float
    depth: float
    centre_x: float
    centre_y: float
    rotation: float = 0
    cleaning_cross_section: bool = False
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    passes: int = 0

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "rotation": self.rotation,
            "centre_x": self.centre_x,
            "centre_y": self.centre_y,
            "cleaning_cross_section": self.cleaning_cross_section,
            "scan_direction": self.scan_direction,
            "cross_section": self.cross_section,
            "passes": self.passes,
        }

    @staticmethod
    def from_dict(data: dict) -> "FibsemRectangleSettings":
        return FibsemRectangleSettings(
            width=data["width"],
            height=data["height"],
            depth=data["depth"],
            centre_x=data["centre_x"],
            centre_y=data["centre_y"],
            cleaning_cross_section=data.get("cleaning_cross_section", False),
            rotation=data.get("rotation", 0),
            scan_direction=data.get("scan_direction", "TopToBottom"),
            cross_section=data.get("cross_section", CrossSectionPattern.Rectangle),
            passes=data.get("passes", 0),
        )

@dataclass
class FibsemLineSettings(FibsemPatternSettings):
    start_x: float
    end_x: float
    start_y: float
    end_y: float
    depth: float

    def to_dict(self) -> dict:
        return {
            "start_x": self.start_x,
            "end_x": self.end_x,
            "start_y": self.start_y,
            "end_y": self.end_y,
            "depth": self.depth,
        }

    @staticmethod
    def from_dict(data: dict) -> "FibsemLineSettings":
        return FibsemLineSettings(
            start_x=data["start_x"],
            end_x=data["end_x"],
            start_y=data["start_y"],
            end_y=data["end_y"],
            depth=data["depth"],
        )

@dataclass
class FibsemCircleSettings(FibsemPatternSettings):
    radius: float
    depth: float
    centre_x: float
    centre_y: float
    thickness: float = 0
    start_angle: float = 0.0
    end_angle: float = 360.0
    rotation: float = 0.0           # annulus -> thickness !=0

    def to_dict(self) -> dict:
        return {
            "radius": self.radius,
            "depth": self.depth,
            "centre_x": self.centre_x,
            "centre_y": self.centre_y,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "rotation": self.rotation,
            "thickness": self.thickness,
        }

    @staticmethod
    def from_dict(data: dict) -> "FibsemCircleSettings":
        return FibsemCircleSettings(
            radius=data["radius"],
            depth=data["depth"],
            centre_x=data["centre_x"],
            centre_y=data["centre_y"],
            start_angle=data.get("start_angle", 0),
            end_angle=data.get("end_angle", 360),
            rotation=data.get("rotation", 0),
            thickness=data.get("thickness", 0),
        )


@dataclass
class FibsemBitmapSettings(FibsemPatternSettings):
    width: float
    height: float
    depth: float
    rotation: float
    centre_x: float
    centre_y: float
    path: str = None

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "rotation": self.rotation,
            "centre_x": self.centre_x,
            "centre_y": self.centre_y,
            "path": self.path,
        }

    @staticmethod
    def from_dict(data: dict) -> "FibsemBitmapSettings":
        return FibsemBitmapSettings(
            width=data["width"],
            height=data["height"],
            depth=data["depth"],
            rotation=data["rotation"],
            centre_x=data["centre_x"],
            centre_y=data["centre_y"],
            path=data["path"],
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
    rate: float = 3.0e-11  # m3/A/s
    dwell_time: float = 1.0e-6  # s
    hfw: float = 150e-6
    patterning_mode: str = "Serial"
    application_file: str = "Si"
    preset: str = "30 keV; UHR imaging"
    spacing: float = 1.0
    milling_voltage: float = 30e3

    def __post_init__(self):
        assert isinstance(
            self.milling_current, (float, int)
        ), f"invalid type for milling_current, must be int or float, currently {type(self.milling_current)}"
        assert isinstance(
            self.spot_size, (float, int)
        ), f"invalid type for spot_size, must be int or float, currently {type(self.spot_size)}"
        assert isinstance(
            self.rate, (float, int)
        ), f"invalid type for rate, must be int or float, currently {type(self.rate)}"
        assert isinstance(
            self.dwell_time, (float, int)
        ), f"invalid type for dwell_time, must be int or float, currently {type(self.dwell_time)}"
        assert isinstance(
            self.hfw, (float, int)
        ), f"invalid type for hfw, must be int or float, currently {type(self.hfw)}"
        assert isinstance(
            self.patterning_mode, str
        ), f"invalid type for value for patterning_mode, must be str, currently {type(self.patterning_mode)}"
        assert isinstance(
            self.application_file, (str)
        ), f"invalid type for value for application_file, must be str, currently {type(self.application_file)}"
        assert isinstance(
            self.spacing, (float, int)
        ), f"invalid type for value for spacing, must be int or float, currently {type(self.spacing)}"
        # assert isinstance(self.preset,(str)), f"invalid type for value for preset, must be str, currently {type(self.preset)}"

    def to_dict(self) -> dict:
        settings_dict = {
            "milling_current": self.milling_current,
            "spot_size": self.spot_size,
            "rate": self.rate,
            "dwell_time": self.dwell_time,
            "hfw": self.hfw,
            "patterning_mode": self.patterning_mode,
            "application_file": self.application_file,
            "preset": self.preset,
            "spacing": self.spacing,
            "milling_voltage": self.milling_voltage,
        }

        return settings_dict

    @staticmethod
    def from_dict(settings: dict) -> "FibsemMillingSettings":
        milling_settings = FibsemMillingSettings(
            milling_current=settings.get("milling_current", 20.0e-12),
            spot_size=settings.get("spot_size", 5.0e-8),
            rate=settings.get("rate", 3.0e-11),
            dwell_time=settings.get("dwell_time", 1.0e-6),
            hfw=settings.get("hfw", 150e-6),
            patterning_mode=settings.get("patterning_mode", "Serial"),
            application_file=settings.get("application_file", "Si"),
            preset=settings.get("preset", "30 keV; 1nA"),
            spacing=settings.get("spacing", 1.0),
            milling_voltage=settings.get("milling_voltage", 30e3),
        )

        return milling_settings

# migrate to this
# v3

# TODO: start here
@dataclass
class StageSystemSettings:
    rotation_reference: float
    rotation_180: float
    shuttle_pre_tilt: float
    manipulator_height_limit: float
    enabled: bool = True
    rotation: bool = True
    tilt: bool  = True

    def to_dict(self):
        return {
            "rotation_reference": self.rotation_reference,
            "rotation_180": self.rotation_180,
            "shuttle_pre_tilt": self.shuttle_pre_tilt,
            "manipulator_height_limit": self.manipulator_height_limit,
            "enabled": self.enabled,
            "rotation": self.rotation,
            "tilt": self.tilt,
        }
    
    @staticmethod
    def from_dict(settings: dict):
        return StageSystemSettings(
            rotation_reference=settings["rotation_reference"],
            rotation_180=settings["rotation_180"],
            shuttle_pre_tilt=settings["shuttle_pre_tilt"],
            manipulator_height_limit=settings["manipulator_height_limit"],
            enabled=settings.get("enabled", True),
            rotation=settings.get("rotation", True),
            tilt=settings.get("tilt", True),
        )


@dataclass
class BeamSystemSettings:
    beam_type: BeamType
    enabled: bool
    beam: BeamSettings
    detector: FibsemDetectorSettings
    eucentric_height: float
    column_tilt: float
    plasma: bool = False
    plasma_gas: str = None

    def to_dict(self):
        ddict = {
            "beam_type": self.beam_type.value,
            "enabled": self.enabled,
            "eucentric_height": self.eucentric_height,
            "column_tilt": self.column_tilt,
            "plasma": self.plasma,
            "plasma_gas": self.plasma_gas,
        }
        ddict.update(self.beam.to_dict())
        ddict.update(self.detector.to_dict())
        
        # rename keys to match config
        ddict["detector_mode"] = ddict.pop("mode")
        ddict["detector_type"] = ddict.pop("type")
        ddict["detector_brightness"] = ddict.pop("brightness")
        ddict["detector_contrast"] = ddict.pop("contrast")
        ddict["current"] = ddict.pop("beam_current")

        return ddict
    
    @staticmethod
    def from_dict(settings: dict) -> 'BeamSystemSettings':
        return BeamSystemSettings(
            beam_type=BeamType[settings["beam_type"]],
            enabled=settings["enabled"],
            beam=BeamSettings.from_dict(settings),
            detector=FibsemDetectorSettings.from_dict(settings),
            eucentric_height=settings["eucentric_height"],
            column_tilt=settings["column_tilt"],
            plasma=settings.get("plasma", False),
            plasma_gas=settings.get("plasma_gas", None),
        )

@dataclass
class ManipulatorSystemSettings:
    enabled: bool
    rotation: bool
    tilt: bool

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "rotation": self.rotation,
            "tilt": self.tilt,
        }
    
    @staticmethod
    def from_dict(settings: dict):
        return ManipulatorSystemSettings(
            enabled=settings["enabled"],
            rotation=settings["rotation"],
            tilt=settings["tilt"],
        )



@dataclass
class GISSystemSettings:
    enabled: bool
    multichem: bool
    sputter_coater: bool
    inserted: bool = False

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "multichem": self.multichem,
            "sputter_coater": self.sputter_coater,
        }
    
    @staticmethod
    def from_dict(settings: dict):
        return GISSystemSettings(
            enabled=settings["enabled"],
            multichem=settings["multichem"],
            sputter_coater=settings["sputter_coater"],
        )

import fibsem

@dataclass
class SystemInfo:
    name: str
    ip_address: str
    manufacturer: str
    model: str
    serial_number: str
    hardware_version: str
    software_version: str
    fibsem_version: str = fibsem.__version__
    application: str = None
    application_version: str = None

    def to_dict(self):
        return {
            "name": self.name,
            "ip_address": self.ip_address,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "hardware_version": self.hardware_version,
            "software_version": self.software_version,
            "fibsem_version": self.fibsem_version,
            "application": self.application,
            "application_version": self.application_version,
        }
    
    @staticmethod
    def from_dict(settings: dict):
        return SystemInfo(
            name=settings.get("name", "Unknown"),
            ip_address=settings.get("ip_address", "Unknown"),
            manufacturer=settings.get("manufacturer", "Unknown"),
            model=settings.get("model", "Unknown"),
            serial_number=settings.get("serial_number", "Unknown"),
            hardware_version=settings.get("hardware_version", "Unknown"),
            software_version=settings.get("software_version", "Unknown"),
            fibsem_version=settings.get("fibsem_version", fibsem.__version__),
            application=settings.get("application", None),
            application_version=settings.get("application_version", None),  
        )

@dataclass
class SystemSettings:
    stage: StageSystemSettings
    electron: BeamSystemSettings
    ion: BeamSystemSettings
    manipulator: ManipulatorSystemSettings
    gis: GISSystemSettings
    info: SystemInfo    

    def to_dict(self):
        return {
            "stage": self.stage.to_dict(),
            "electron": self.electron.to_dict(),
            "ion": self.ion.to_dict(),
            "manipulator": self.manipulator.to_dict(),
            "gis": self.gis.to_dict(),
            "info": self.info.to_dict(),
        }
    
    @staticmethod
    def from_dict(settings: dict):

        # TODO: remove this once the settings are updated
        settings["electron"]["beam_type"] = BeamType.ELECTRON.name
        settings["ion"]["beam_type"] = BeamType.ION.name
            
        return SystemSettings(
            stage=StageSystemSettings.from_dict(settings["stage"]),
            electron=BeamSystemSettings.from_dict(settings["electron"]),
            ion=BeamSystemSettings.from_dict(settings["ion"]),
            manipulator=ManipulatorSystemSettings.from_dict(settings["manipulator"]),
            gis=GISSystemSettings.from_dict(settings["gis"]),
            info=SystemInfo.from_dict(settings["info"]),
        )

@dataclass
class MicroscopeSettings:

    """
    A data class representing the settings for a microscope system.

    Attributes:
        system (SystemSettings): An instance of the `SystemSettings` class that holds the system settings.
        image (ImageSettings): An instance of the `ImageSettings` class that holds the image settings.
        milling (FibsemMillingSettings): An instance of the `FibsemMillingSettings` class that holds the fibsem milling settings..
        protocol (dict, optional): A dictionary representing the protocol settings. Defaults to None.

    Methods:
        to_dict(): Returns a dictionary representation of the `MicroscopeSettings` object.
        from_dict(settings: dict, protocol: dict = None) -> "MicroscopeSettings": Returns an instance of the `MicroscopeSettings` class from a dictionary.
    """

    system: SystemSettings
    image: ImageSettings
    milling: FibsemMillingSettings
    protocol: dict = None

    def to_dict(self) -> dict:
        settings_dict = {
            "imaging": self.image.to_dict(),
            "protocol": self.protocol,
            "milling": self.milling.to_dict(),
        }
        settings_dict.update(self.system.to_dict())

        return settings_dict

    @staticmethod
    def from_dict(
        settings: dict, protocol: dict = None
    ) -> "MicroscopeSettings":
        
        if protocol is None:
            protocol = settings.get("protocol", {"name": "demo"})
     
        return MicroscopeSettings(
            system=SystemSettings.from_dict(settings),
            image=ImageSettings.from_dict(settings["imaging"]),
            protocol=protocol,
            milling=FibsemMillingSettings.from_dict(settings["milling"]),
        )





@dataclass
class FibsemExperiment:
    id: str = None
    method: str = None
    date: float = datetime.timestamp(datetime.now())
    application: str = "OpenFIBSEM"
    fibsem_version: str = fibsem.__version__
    application_version: str = None

    def to_dict(self) -> dict:
        """Converts to a dictionary."""
        return {
            "id": self.id,
            "method": self.method,
            "date": self.date,
            "application": self.application,
            "fibsem_version": self.fibsem_version,
            "application_version": self.application_version,
        }

    @staticmethod
    def from_dict(settings: dict) -> "FibsemExperiment":
        """Converts from a dictionary."""
        return FibsemExperiment(
            id=settings.get("id", "Unknown"),
            method=settings.get("method", "Unknown"),
            date=settings.get("date", "Unknown"),
            application=settings.get("application", "OpenFIBSEM"),
            fibsem_version=settings.get("fibsem_version", fibsem.__version__),
            application_version=settings.get("application_version", None),
        )


@dataclass
class FibsemUser:
    name: str = None
    email: str = None
    organization: str = None
    hostname: str = None
    # TODO: add host_ip_address

    def to_dict(self) -> dict:
        """Converts to a dictionary."""
        return {
            "name": self.name,
            "email": self.email,
            "organization": self.organization,
            "hostname": self.hostname,
        }

    @staticmethod
    def from_dict(settings: dict) -> "FibsemUser":
        """Converts from a dictionary."""
        return FibsemUser(
            name=settings.get("name", "Unknown"),
            email=settings.get("email", "Unknown"),
            organization=settings.get("organization", "Unknown"),
            hostname=settings.get("hostname", "Unknown"),
        )

    @staticmethod
    def from_environment() -> "FibsemUser":
        import platform
        import socket
        username = os.environ.get("USERNAME", "username")

        if platform.system() == "Windows":
            hostname = os.environ.get("COMPUTERNAME", "hostname")
        elif platform.system() in ["Linux", "Darwin"]:
            hostname = socket.gethostname()
        else:
            hostname = "hostname"
            
        user = FibsemUser(name=username, email="null", organization="null", hostname=hostname)
        
        return user


@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage."""

    image_settings: ImageSettings
    pixel_size: Point
    microscope_state: MicroscopeState
    system: SystemSettings = None
    version: str = METADATA_VERSION
    user: FibsemUser = FibsemUser()
    experiment: FibsemExperiment = FibsemExperiment()

    def to_dict(self) -> dict:
        """Converts metadata to a dictionary.

        Returns:
            dictionary: self as a dictionary
        """
        settings_dict = {}
        if self.image_settings is not None:
            settings_dict["image"] = self.image_settings.to_dict()
        if self.version is not None:
            settings_dict["version"] = self.version
        if self.pixel_size is not None:
            settings_dict["pixel_size"] = self.pixel_size.to_dict()
        if self.microscope_state is not None:
            settings_dict["microscope_state"] = self.microscope_state.to_dict()
            settings_dict["user"] = self.user.to_dict()
        settings_dict["experiment"] = self.experiment.to_dict()
        settings_dict["system"] = self.system.to_dict() if self.system is not None else {}

        return settings_dict

    @staticmethod
    def from_dict(settings: dict) -> "ImageSettings":
        """Converts a dictionary to metadata."""

        image_settings = ImageSettings.from_dict(settings["image"])
        version = settings.get("version", METADATA_VERSION)
        if settings["pixel_size"] is not None:
            pixel_size = Point.from_dict(settings["pixel_size"])
        if settings["microscope_state"] is not None:
            microscope_state = MicroscopeState.from_dict(
                settings["microscope_state"]
            )

        metadata = FibsemImageMetadata(
            image_settings=image_settings,
            version=version,
            pixel_size=pixel_size,
            microscope_state=microscope_state,
            # detector_settings=detector_settings, # TODO: remove this
            user=FibsemUser.from_dict(settings.get("user", {})),
            experiment=FibsemExperiment.from_dict(settings.get("experiment", {})),
            system=SystemSettings.from_dict(settings.get("system", {})),
        )
        return metadata

    if THERMO:
        # TODO: move to ImageSettings
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
                autogamma=True,
                save=False,
                path="path",
                filename=current_timestamp(),
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
            self.image_settings.resolution[0] == image_settings.resolution[0]
            and self.image_settings.resolution[1] == image_settings.resolution[1]
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
            self.image_settings.autogamma == image_settings.autogamma
        ), f"gamma: {self.image_settings.autogamma} != {image_settings.autogamma}"
        assert (
            self.image_settings.save == image_settings.save
        ), f"save: {self.image_settings.save} != {image_settings.save}"
        assert (
            self.image_settings.path == image_settings.path
        ), f"path: {self.image_settings.path} != {image_settings.path}"
        assert (
            self.image_settings.filename == image_settings.filename
        ), f"filename: {self.image_settings.filename} != {image_settings.filename}"
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

        save(self, path: Path) -> None:
            Saves a FibsemImage to a tiff file.

            Inputs:
                path (path): path to save directory and filename
    """

    def __init__(self, data: np.ndarray, metadata: FibsemImageMetadata = None):
        if check_data_format(data):
            if data.ndim == 3 and data.shape[2] == 1:
                data = data[:, :, 0]
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
                metadata = FibsemImageMetadata.from_dict(metadata)
            except Exception as e:
                metadata = None
                # print(f"Error: {e}")
                # import traceback
                # traceback.print_exc()
        return cls(data=data, metadata=metadata)

    def save(self, path: Path = None) -> None:
        """Saves a FibsemImage to a tiff file.

        Inputs:
            path (path): path to save directory and filename
        """
        if path is None:
            path = os.path.join(
                self.metadata.image_settings.path,
                self.metadata.image_settings.filename,
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = Path(path).with_suffix(".tif")

        if self.metadata is not None:
            metadata_dict = self.metadata.to_dict()
        else:
            metadata_dict = None
        tff.imwrite(
            path,
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
                    stage_position=FibsemStagePosition(
                        adorned.metadata.stage_settings.stage_position.x,
                        adorned.metadata.stage_settings.stage_position.y,
                        adorned.metadata.stage_settings.stage_position.z,
                        adorned.metadata.stage_settings.stage_position.r,
                        adorned.metadata.stage_settings.stage_position.t,
                    ),
                    electron_beam=BeamSettings(beam_type=BeamType.ELECTRON),
                    ion_beam=BeamSettings(beam_type=BeamType.ION),
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
            detector: FibsemDetectorSettings,
        ) -> "FibsemImage":
            """Creates FibsemImage from a tescan (microscope output format).

            Args:
                image (Tescan): Adorned Image from microscope
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
                # detector_settings=detector,
            )

            return cls(data=np.array(image.Image), metadata=metadata)

        @classmethod
        def fromTescanFile(
            cls,
            image_path: str,
            metadata_path: str,
            beam_type: BeamType,
        ) -> "FibsemImage":
            with tff.TiffFile(image_path) as tiff_image:
                data = tiff_image.asarray()

            stage = 0
            dictionary = {"MAIN": {}, "SEM": {}, "FIB": {}}
            with open(metadata_path, "r") as file:
                for line in file:
                    if line.startswith("["):
                        stage += 1
                        continue

                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines

                    key, value = line.split("=")
                    key = key.strip()
                    value = value.strip()
                    if stage == 1:
                        dictionary["MAIN"][key] = value
                    if stage == 2 and beam_type.name == "ELECTRON":
                        dictionary["SEM"][key] = value
                    if stage == 2 and beam_type.name == "ION":
                        dictionary["FIB"][key] = value

            if beam_type.name == "ELECTRON":
                image_settings = ImageSettings(
                    resolution=[data.shape[0], data.shape[1]],
                    dwell_time=float(dictionary["SEM"]["DwellTime"]),
                    hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
                    beam_type=BeamType.ELECTRON,
                    filename=Path(image_path).stem,
                    path=Path(image_path).parent,
                )
                pixel_size = Point(
                    float(dictionary["MAIN"]["PixelSizeX"]),
                    float(dictionary["MAIN"]["PixelSizeY"]),
                )
                microscope_state = MicroscopeState(
                    timestamp=datetime.strptime(
                        dictionary["MAIN"]["Date"] + " " + dictionary["MAIN"]["Time"],
                        "%Y-%m-%d %H:%M:%S",
                    ),
                    electron_beam=BeamSettings(
                        beam_type=BeamType.ELECTRON,
                        working_distance=float(dictionary["SEM"]["WD"]),
                        beam_current=float(dictionary["SEM"]["PredictedBeamCurrent"]),
                        voltage=float(dictionary["SEM"]["TubeVoltage"]),
                        hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
                        resolution=[data.shape[0], data.shape[1]],
                        dwell_time=float(dictionary["SEM"]["DwellTime"]),
                        shift=Point(
                            float(dictionary["SEM"]["ImageShiftX"]),
                            float(dictionary["SEM"]["ImageShiftY"]),
                        ),
                        stigmation=Point(
                            float(dictionary["SEM"]["StigmatorX"]),
                            float(dictionary["SEM"]["StigmatorY"]),
                        ),
                    ),
                    ion_beam=BeamSettings(beam_type=BeamType.ION),
                )
                detector_settings = FibsemDetectorSettings(
                    type=dictionary["SEM"]["Detector"],
                    brightness=float(dictionary["SEM"]["Detector0Offset"]),
                    contrast=float(dictionary["SEM"]["Detector0Gain"]),
                )

            if beam_type.name == "ION":
                image_settings = ImageSettings(
                    resolution=[data.shape[0], data.shape[1]],
                    dwell_time=float(dictionary["FIB"]["DwellTime"]),
                    hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
                    beam_type=BeamType.ELECTRON,
                    filename=Path(image_path).stem,
                    path=Path(image_path).parent,
                )
                pixel_size = Point(
                    float(dictionary["MAIN"]["PixelSizeX"]),
                    float(dictionary["MAIN"]["PixelSizeY"]),
                )
                microscope_state = MicroscopeState(
                    timestamp=datetime.strptime(
                        dictionary["MAIN"]["Date"] + " " + dictionary["MAIN"]["Time"],
                        "%Y-%m-%d %H:%M:%S",
                    ),
                    electron_beam=BeamSettings(beam_type=BeamType.ELECTRON),
                    ion_beam=BeamSettings(
                        beam_type=BeamType.ION,
                        working_distance=float(dictionary["FIB"]["WD"]),
                        beam_current=float(dictionary["FIB"]["PredictedBeamCurrent"]),
                        hfw=data.shape[0] * float(dictionary["MAIN"]["PixelSizeX"]),
                        resolution=[data.shape[0], data.shape[1]],
                        dwell_time=float(dictionary["FIB"]["DwellTime"]),
                        shift=Point(
                            float(dictionary["FIB"]["ImageShiftX"]),
                            float(dictionary["FIB"]["ImageShiftY"]),
                        ),
                        stigmation=Point(
                            float(dictionary["FIB"]["StigmatorX"]),
                            float(dictionary["FIB"]["StigmatorY"]),
                        ),
                    ),
                )
                detector_settings = FibsemDetectorSettings(
                    type=dictionary["FIB"]["Detector"],
                    brightness=float(dictionary["FIB"]["Detector0Offset"]) / 100,
                    contrast=float(dictionary["FIB"]["Detector0Gain"]) / 100,
                )

            metadata = FibsemImageMetadata(
                image_settings=image_settings,
                pixel_size=pixel_size,
                microscope_state=microscope_state,
                # detector_settings=detector_settings,
                version=METADATA_VERSION,
            )
            return cls(data=data, metadata=metadata)


@dataclass
class ReferenceImages:
    low_res_eb: FibsemImage
    high_res_eb: FibsemImage
    low_res_ib: FibsemImage
    high_res_ib: FibsemImage

    def __iter__(self) -> list[FibsemImage]:
        yield self.low_res_eb, self.high_res_eb, self.low_res_ib, self.high_res_ib


class ThermoGISLine:
    def __init__(self, line=None, name=None, status: str = "Retracted"):
        self.line = line
        self.name = name
        self.status = status
        self.temp_ready = False

    def insert(self):
        if self.line is not None:
            self.line.insert()
        self.status = "Inserted"

    def retract(self):
        if self.line is not None:
            self.line.retract()
        self.status = "Retracted"


class ThermoMultiChemLine:
    def __init__(self, line=None, status: str = "Retracted"):
        self.line = line
        self.status = status
        self.positions = ["Electron Default", "Ion Default", "Retract"]
        self.current_position = "Retract"
        self.temp_ready = False

    def insert(self, position):
        # position_str = getattr(MultiChemInsertPosition,position)

        if self.line is not None:
            self.line.insert(position)

        self.current_position = position
        self.status = "Inserted"

    def retract(self):
        if self.line is not None:
            self.line.retract()

        self.status = "Retracted"
        self.current_position = "Retracted"


def check_data_format(data: np.ndarray) -> bool:
    """Checks that data is in the correct format."""
    # assert data.ndim == 2  # or data.ndim == 3
    # assert data.dtype in [np.uint8, np.uint16]
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    return data.ndim == 2 and data.dtype in [np.uint8, np.uint16]
