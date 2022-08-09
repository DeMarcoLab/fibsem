# fibsem structures

from dataclasses import dataclass
from autoscript_sdb_microscope_client.structures import AdornedImage, StagePosition
from enum import Enum


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0


# TODO: conver these to match autoscript...
class BeamType(Enum):
    ELECTRON = 1
    ION = 2


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
    resolution: float = None
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


def beam_settings_from_dict(state_dict: dict) -> None:
    # TOOD: class method?
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
    timestamp: float = None
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


def microscope_state_from_dict(state_dict: dict) -> MicroscopeState:
    # TOOD: class method?
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
        eb_settings=beam_settings_from_dict(state_dict["eb_settings"]),
        ib_settings=beam_settings_from_dict(state_dict["ib_settings"]),
    )

    return microscope_state
