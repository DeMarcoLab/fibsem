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
    enabled: bool
    min_gamma: float
    max_gamma: float
    scale_factor: float
    threshold: int  # px

    @classmethod
    def __from_dict__(self, settings):
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
    def __from_dict__(self, settings):

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

    def __iter__(self):

        yield self.low_res_eb, self.high_res_eb, self.low_res_ib, self.high_res_ib


@dataclass
class BeamSettings:
    beam_type: BeamType
    working_distance: float = None
    beam_current: float = None
    hfw: float = None
    resolution: str = None
    dwell_time: float = None
    stigmation: float = None

    def __to_dict__(self):

        state_dict = {
            "beam_type": self.beam_type.name,
            "working_distance": self.working_distance,
            "beam_current": self.beam_current,
            "hfw": self.hfw,
            "resolution": self.resolution,
            "dwell_time": self.dwell_time,
            "stigmation": self.stigmation,
        }

        return state_dict

    @classmethod
    def __from_dict__(self, state_dict: dict):
        beam_settings = BeamSettings(
            beam_type=BeamType[state_dict["beam_type"]],
            working_distance=state_dict["working_distance"],
            beam_current=state_dict["beam_current"],
            hfw=state_dict["hfw"],
            resolution=state_dict["resolution"],
            dwell_time=state_dict["dwell_time"],
            stigmation=state_dict["stigmation"],
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
    def __from_dict__(self, state_dict: dict):
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
    def __from_dict__(self, settings):

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