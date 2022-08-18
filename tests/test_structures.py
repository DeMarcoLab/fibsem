from datetime import datetime

import numpy as np
import pytest
from autoscript_sdb_microscope_client.structures import StagePosition
from fibsem.structures import (BeamSettings, BeamType, CalibrationSettings,
                               GammaSettings, ImageSettings, MicroscopeState,
                               MillingSettings, StageSettings, SystemSettings,
                               stage_position_from_dict,
                               stage_position_to_dict)


@pytest.fixture
def milling_settings() -> MillingSettings:

    milling_settings = MillingSettings(
        width=10.0e-6,
        height=10.0e-6,
        depth=10.0e-6,
        rotation=np.deg2rad(45),
        centre_x=0.0,
        centre_y=0.0,
        milling_current=2.0e-9,
        scan_direction="TopToBottom",
        cleaning_cross_section=False,
    )

    return milling_settings


@pytest.fixture
def beam_settings() -> BeamSettings:

    beam_settings = BeamSettings(
        beam_type=BeamType.ELECTRON,
        working_distance=4.0e-3,
        beam_current=60.0e-12,
        hfw=150e-6,
        resolution="1536x1024",
        dwell_time=1.0e-6,
        # stigmation=1.0e-6,
    )

    return beam_settings


@pytest.fixture
def microscope_state(beam_settings: BeamSettings) -> MicroscopeState:

    microscope_state = MicroscopeState()
    beam_settings.beam_type = BeamType.ELECTRON
    microscope_state.eb_settings = beam_settings
    beam_settings.beam_type = BeamType.ION
    microscope_state.ib_settings = beam_settings

    return microscope_state

@pytest.fixture
def gamma_settings() -> GammaSettings:

    gamma_settings = GammaSettings(
        enabled=True, 
        min_gamma=0.15, 
        max_gamma=1.8, 
        scale_factor=0.1, 
        threshold=46
    )

    return gamma_settings

@pytest.fixture
def image_settings(gamma_settings: GammaSettings) -> ImageSettings:


    image_settings = ImageSettings(
        resolution="1536x1024",
        dwell_time=1.e-6,
        hfw=150.e-6, 
        save=False, 
        label = "label",
        save_path=None,
        gamma=gamma_settings
    )

    return image_settings

@pytest.fixture
def system_settings() -> SystemSettings:

    system_settings = SystemSettings()

    return system_settings

def test_milling_settings_to_dict(milling_settings: MillingSettings):

    milling_settings_dict = milling_settings.__to_dict__()

    assert milling_settings.width == milling_settings_dict["width"]
    assert milling_settings.height == milling_settings_dict["height"]
    assert milling_settings.depth == milling_settings_dict["depth"]
    assert milling_settings.rotation == np.deg2rad(milling_settings_dict["rotation"])
    assert milling_settings.centre_x == milling_settings_dict["centre_x"]
    assert milling_settings.centre_y == milling_settings_dict["centre_y"]
    assert milling_settings.milling_current == milling_settings_dict["milling_current"]
    assert milling_settings.scan_direction == milling_settings_dict["scan_direction"]
    assert (
        milling_settings.cleaning_cross_section
        == milling_settings_dict["cleaning_cross_section"]
    )


def test_milling_settings_from_dict():
    milling_settings_dict = {
        "width": 10.0e-6,
        "height": 10.0e-6,
        "depth": 10.0e-6,
        "rotation": 45,
        "centre_x": 0.0,
        "centre_y": 0.0,
        "milling_current": 2.0e-9,
        "scan_direction": "TopToBottom",
        "cleaning_cross_section": False,
    }

    milling_settings = MillingSettings.__from_dict__(milling_settings_dict)

    assert milling_settings.width == milling_settings_dict["width"]
    assert milling_settings.height == milling_settings_dict["height"]
    assert milling_settings.depth == milling_settings_dict["depth"]
    assert milling_settings.rotation == np.deg2rad(milling_settings_dict["rotation"])
    assert milling_settings.centre_x == milling_settings_dict["centre_x"]
    assert milling_settings.centre_y == milling_settings_dict["centre_y"]
    assert milling_settings.milling_current == milling_settings_dict["milling_current"]
    assert milling_settings.scan_direction == milling_settings_dict["scan_direction"]
    assert (
        milling_settings.cleaning_cross_section
        == milling_settings_dict["cleaning_cross_section"]
    )


def test_beam_settings_to_dict(beam_settings: BeamSettings):

    beam_settings_dict = beam_settings.__to_dict__()

    assert beam_settings.beam_type == BeamType[beam_settings_dict["beam_type"]]
    assert beam_settings.working_distance == beam_settings_dict["working_distance"]
    assert beam_settings.beam_current == beam_settings_dict["beam_current"]
    assert beam_settings.hfw == beam_settings_dict["hfw"]
    assert beam_settings.resolution == beam_settings_dict["resolution"]
    assert beam_settings.dwell_time == beam_settings_dict["dwell_time"]
    # assert beam_settings.stigmation == beam_settings_dict["stigmation"]


def test_beam_settings_from_dict():

    beam_settings_dict = {
        "beam_type": "ELECTRON",
        "working_distance": 4.0e-3,
        "beam_current": 60.0e-12,
        "hfw": 150.0e-6,
        "resolution": "1560x1240",
        "dwell_time": 1.0e-6,
        "stigmation": 1.0e-6,
    }

    beam_settings = BeamSettings.__from_dict__(beam_settings_dict)

    assert beam_settings.beam_type == BeamType[beam_settings_dict["beam_type"]]
    assert beam_settings.working_distance == beam_settings_dict["working_distance"]
    assert beam_settings.beam_current == beam_settings_dict["beam_current"]
    assert beam_settings.hfw == beam_settings_dict["hfw"]
    assert beam_settings.resolution == beam_settings_dict["resolution"]
    assert beam_settings.dwell_time == beam_settings_dict["dwell_time"]
    # assert beam_settings.stigmation == beam_settings_dict["stigmation"]


def test_microscope_state_to_dict(microscope_state: MicroscopeState):

    state_dict = microscope_state.__to_dict__()

    assert microscope_state.timestamp == state_dict["timestamp"]
    assert microscope_state.absolute_position.x == state_dict["absolute_position"]["x"]
    assert microscope_state.absolute_position.y == state_dict["absolute_position"]["y"]
    assert microscope_state.absolute_position.z == state_dict["absolute_position"]["z"]
    assert microscope_state.absolute_position.r == state_dict["absolute_position"]["r"]
    assert microscope_state.absolute_position.t == state_dict["absolute_position"]["t"]
    assert microscope_state.absolute_position.coordinate_system == state_dict["absolute_position"]["coordinate_system"]

    assert microscope_state.eb_settings.beam_type == BeamType[state_dict["eb_settings"]["beam_type"]]
    assert microscope_state.eb_settings.working_distance == state_dict["eb_settings"]["working_distance"]
    assert microscope_state.eb_settings.beam_current == state_dict["eb_settings"]["beam_current"]
    assert microscope_state.eb_settings.hfw == state_dict["eb_settings"]["hfw"]
    assert microscope_state.eb_settings.resolution == state_dict["eb_settings"]["resolution"]
    assert microscope_state.eb_settings.dwell_time == state_dict["eb_settings"]["dwell_time"]
    # assert microscope_state.eb_settings.stigmation == state_dict["eb_settings"]["stigmation"]
    
    assert microscope_state.ib_settings.beam_type == BeamType[state_dict["ib_settings"]["beam_type"]]
    assert microscope_state.ib_settings.working_distance == state_dict["ib_settings"]["working_distance"]
    assert microscope_state.ib_settings.beam_current == state_dict["ib_settings"]["beam_current"]
    assert microscope_state.ib_settings.hfw == state_dict["ib_settings"]["hfw"]
    assert microscope_state.ib_settings.resolution == state_dict["ib_settings"]["resolution"]
    assert microscope_state.ib_settings.dwell_time == state_dict["ib_settings"]["dwell_time"]
    # assert microscope_state.ib_settings.stigmation == state_dict["ib_settings"]["stigmation"]

    return NotImplemented


def test_microscope_state_from_dict():


    eb_beam_settings_dict = {
        "beam_type": "ELECTRON",
        "working_distance": 4.0e-3,
        "beam_current": 60.0e-12,
        "hfw": 150.0e-6,
        "resolution": "1560x1240",
        "dwell_time": 1.0e-6,
        "stigmation": 1.0e-6,
    }

    ib_beam_settings_dict = {
        "beam_type": "ION",
        "working_distance": 4.0e-3,
        "beam_current": 60.0e-12,
        "hfw": 150.0e-6,
        "resolution": "1560x1240",
        "dwell_time": 1.0e-6,
        "stigmation": 1.0e-6,
    }

    stage_position_dict = {
        "x": 0,
        "y": 0,
        "z": 0,
        "r": 0,
        "t": 0,
        "coordinate_system": "Raw"
    }

    state_dict = {
        "timestamp": datetime.timestamp(datetime.now()),
        "absolute_position": stage_position_dict,
        "eb_settings": eb_beam_settings_dict,
        "ib_settings": ib_beam_settings_dict
    }

    microscope_state = MicroscopeState.__from_dict__(state_dict)

    assert microscope_state.timestamp == state_dict["timestamp"]
    assert microscope_state.absolute_position.x == state_dict["absolute_position"]["x"]
    assert microscope_state.absolute_position.y == state_dict["absolute_position"]["y"]
    assert microscope_state.absolute_position.z == state_dict["absolute_position"]["z"]
    assert microscope_state.absolute_position.r == state_dict["absolute_position"]["r"]
    assert microscope_state.absolute_position.t == state_dict["absolute_position"]["t"]
    assert microscope_state.absolute_position.coordinate_system == state_dict["absolute_position"]["coordinate_system"]

    assert microscope_state.eb_settings.beam_type == BeamType[state_dict["eb_settings"]["beam_type"]]
    assert microscope_state.eb_settings.working_distance == state_dict["eb_settings"]["working_distance"]
    assert microscope_state.eb_settings.beam_current == state_dict["eb_settings"]["beam_current"]
    assert microscope_state.eb_settings.hfw == state_dict["eb_settings"]["hfw"]
    assert microscope_state.eb_settings.resolution == state_dict["eb_settings"]["resolution"]
    assert microscope_state.eb_settings.dwell_time == state_dict["eb_settings"]["dwell_time"]
    # assert microscope_state.eb_settings.stigmation == state_dict["eb_settings"]["stigmation"]
    
    assert microscope_state.ib_settings.beam_type == BeamType[state_dict["ib_settings"]["beam_type"]]
    assert microscope_state.ib_settings.working_distance == state_dict["ib_settings"]["working_distance"]
    assert microscope_state.ib_settings.beam_current == state_dict["ib_settings"]["beam_current"]
    assert microscope_state.ib_settings.hfw == state_dict["ib_settings"]["hfw"]
    assert microscope_state.ib_settings.resolution == state_dict["ib_settings"]["resolution"]
    assert microscope_state.ib_settings.dwell_time == state_dict["ib_settings"]["dwell_time"]
    # assert microscope_state.ib_settings.stigmation == state_dict["ib_settings"]["stigmation"]


def test_gamma_settings_from_dict():

    gamma_dict = {
        "enabled": True,
        "min_gamma": 0.5,
        "max_gamma": 1.8,
        "scale_factor": 0.01,
        "threshold": 46
    }

    gamma_settings = GammaSettings.__from_dict__(gamma_dict)

    assert gamma_settings.enabled == gamma_dict["enabled"]
    assert gamma_settings.min_gamma == gamma_dict["min_gamma"]
    assert gamma_settings.max_gamma == gamma_dict["max_gamma"]
    assert gamma_settings.scale_factor == gamma_dict["scale_factor"]
    assert gamma_settings.threshold == gamma_dict["threshold"]

def test_image_settings_from_dict():

    gamma_dict = {
        "enabled": True,
        "min_gamma": 0.5,
        "max_gamma": 1.8,
        "scale_factor": 0.01,
        "threshold": 46
    }

    image_settings_dict = {
        "resolution": "1536x1024",
        "dwell_time": 1.e-6,
        "hfw": 150.e-6,
        "autocontrast": True,
        "beam_type": "ELECTRON", 
        "gamma": gamma_dict,
        "save": False,
        "save_path": "path",
        "label": "label"
    }

    image_settings = ImageSettings.__from_dict__(image_settings_dict)

    assert image_settings.resolution == image_settings_dict["resolution"]
    assert image_settings.dwell_time == image_settings_dict["dwell_time"]
    assert image_settings.hfw == image_settings_dict["hfw"]
    assert image_settings.autocontrast == image_settings_dict["autocontrast"]
    assert image_settings.beam_type == BeamType[image_settings_dict["beam_type"].upper()]
    assert image_settings.save == image_settings_dict["save"]
    assert image_settings.save_path == image_settings_dict["save_path"]
    assert image_settings.label == image_settings_dict["label"]
    assert image_settings.gamma.enabled == image_settings_dict["gamma"]["enabled"]
    assert image_settings.gamma.min_gamma == image_settings_dict["gamma"]["min_gamma"]
    assert image_settings.gamma.max_gamma == image_settings_dict["gamma"]["max_gamma"]
    assert image_settings.gamma.scale_factor == image_settings_dict["gamma"]["scale_factor"]
    assert image_settings.gamma.threshold == image_settings_dict["gamma"]["threshold"]


def test_system_settings_to_dict(system_settings: SystemSettings):

    settings_dict = system_settings.__to_dict__()

    assert system_settings.ip_address == settings_dict["ip_address"]
    assert system_settings.application_file == settings_dict["application_file"]
    assert system_settings.plasma_gas == settings_dict["plasma_gas"]
    assert system_settings.ion_voltage == settings_dict["ion_voltage"]
    assert system_settings.electron_voltage == settings_dict["electron_voltage"]
    assert system_settings.ion_current == settings_dict["ion_current"]
    assert system_settings.electron_current == settings_dict["electron_current"]


def test_system_settings_from_dict():

    settings_dict = {
        "ip_address": "10.0.0.1",
        "application_file": "autolamella", 
        "plasma_gas": "Argon", 
        "ion_voltage": 30000,
        "electron_voltage": 2000,
        "electron_current": 1e-9,
        "ion_current": 20e-12,

    }

    system_settings = SystemSettings.__from_dict__(settings_dict)

    assert system_settings.ip_address == settings_dict["ip_address"]
    assert system_settings.application_file == settings_dict["application_file"]
    assert system_settings.plasma_gas == settings_dict["plasma_gas"]
    assert system_settings.ion_voltage == settings_dict["ion_voltage"]
    assert system_settings.electron_voltage == settings_dict["electron_voltage"]
    assert system_settings.ion_current == settings_dict["ion_current"]
    assert system_settings.electron_current == settings_dict["electron_current"]


def test_stage_settings_to_dict():
    
    stage_settings = StageSettings()

    settings_dict = stage_settings.__to_dict__()

    assert stage_settings.rotation_flat_to_electron == settings_dict["rotation_flat_to_electron"]
    assert stage_settings.rotation_flat_to_ion == settings_dict["rotation_flat_to_ion"]
    assert stage_settings.tilt_flat_to_electron == settings_dict["tilt_flat_to_electron"]
    assert stage_settings.tilt_flat_to_ion == settings_dict["tilt_flat_to_ion"]

def test_stage_settings_from_dict():

    settings_dict = {
        "rotation_flat_to_electron": 50,
        "rotation_flat_to_ion": 230,
        "tilt_flat_to_electron": 27,
        "tilt_flat_to_ion": 52
    }

    stage_settings = StageSettings.__from_dict__(settings_dict)

    assert stage_settings.rotation_flat_to_electron == settings_dict["rotation_flat_to_electron"]
    assert stage_settings.rotation_flat_to_ion == settings_dict["rotation_flat_to_ion"]
    assert stage_settings.tilt_flat_to_electron == settings_dict["tilt_flat_to_electron"]
    assert stage_settings.tilt_flat_to_ion == settings_dict["tilt_flat_to_ion"]


def test_calibration_settings_to_dict():

    calibration_settings = CalibrationSettings()

    settings = calibration_settings.__to_dict__()

    assert calibration_settings.max_hfw_eb == settings["max_hfw_eb"]
    assert calibration_settings.max_hfw_ib == settings["max_hfw_ib"]
    assert calibration_settings.eucentric_height_eb == settings["eucentric_height_eb"]
    assert calibration_settings.eucentric_height_ib == settings["eucentric_height_ib"]
    assert calibration_settings.eucentric_height_tolerance == settings["eucentric_height_tolerance"]
    assert calibration_settings.needle_stage_height_limit == settings["needle_stage_height_limit"]


def test_calibration_settings_from_dict():

    settings = {
        "max_hfw_eb": 2700e-6,
        "max_hfw_ib": 900e-6,
        "eucentric_height_eb": 4.0e-3,
        "eucentric_height_ib": 16.5e-3,
        "eucentric_height_tolerance": 0.5e-3,
        "needle_stage_height_limit": 3.7e-3,
    }

    calibration_settings = CalibrationSettings.__from_dict__(settings)


    assert calibration_settings.max_hfw_eb == settings["max_hfw_eb"]
    assert calibration_settings.max_hfw_ib == settings["max_hfw_ib"]
    assert calibration_settings.eucentric_height_eb == settings["eucentric_height_eb"]
    assert calibration_settings.eucentric_height_ib == settings["eucentric_height_ib"]
    assert calibration_settings.eucentric_height_tolerance == settings["eucentric_height_tolerance"]
    assert calibration_settings.needle_stage_height_limit == settings["needle_stage_height_limit"]



def test_stage_position_to_dict():
    
    stage_position = StagePosition(
        x=1, y=2, z=3, r=4, t=5, coordinate_system="Raw"
    )

    stage_position_dict = stage_position_to_dict(stage_position)

    assert stage_position.x == stage_position_dict["x"]
    assert stage_position.y == stage_position_dict["y"] 
    assert stage_position.z == stage_position_dict["z"] 
    assert stage_position.r == stage_position_dict["r"] 
    assert stage_position.t == stage_position_dict["t"] 
    assert stage_position.coordinate_system == stage_position_dict["coordinate_system"]  

def test_stage_position_from_dict():
    
    stage_position_dict = {
        "x": 1,
        "y": 2,
        "z": 3,
        "r": 4,
        "t": 5,
        "coordinate_system": "Raw"
    }

    stage_position = stage_position_from_dict(stage_position_dict)

    assert stage_position.x == stage_position_dict["x"]
    assert stage_position.y == stage_position_dict["y"] 
    assert stage_position.z == stage_position_dict["z"] 
    assert stage_position.r == stage_position_dict["r"] 
    assert stage_position.t == stage_position_dict["t"] 
    assert stage_position.coordinate_system == stage_position_dict["coordinate_system"] 

def test_default_settings_from_dict():

    return NotImplemented

def test_microscope_settings():

    return NotImplemented