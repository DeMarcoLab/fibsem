from fibsem.config import load_yaml, load_microscope_manufacturer, CONFIG_PATH
from fibsem.utils import load_protocol
from fibsem.structures import (
    BeamType,
    MicroscopeSettings,
    ImageSettings,
    SystemSettings,
    FibsemImage,
    FibsemMillingSettings,
    StageSettings,
    BeamSystemSettings,
    FibsemHardware
)
import os
import pytest


@pytest.fixture
def load_settings():
    
    settings = load_yaml(os.path.join(CONFIG_PATH, "system.yaml"))
    model_path = os.path.join(CONFIG_PATH, "model.yaml")
    
    
    system_settings = SystemSettings.__from_dict__(settings["system"])
    image_settings = ImageSettings.__from_dict__(settings["user"]["imaging"])
    milling_settings = FibsemMillingSettings.__from_dict__(settings["user"]["milling"])
    hardware_settings = FibsemHardware.__from_dict__(load_protocol(model_path))
    protocol = load_protocol()

    return [system_settings,image_settings,milling_settings,hardware_settings,hardware_settings,protocol]


def test_yaml_loaded(load_settings):

    system_settings = load_settings[0]
    image_settings = load_settings[1]
    milling_settings = load_settings[2]

    assert system_settings is not None
    assert image_settings is not None
    assert milling_settings is not None

    assert isinstance(system_settings,SystemSettings)
    assert isinstance(image_settings,ImageSettings)
    assert isinstance(milling_settings,FibsemMillingSettings)

    # check system settings attributes



def test_stage_settings(load_settings):


    sys_attributes = [
        "ip_address",
        "stage",
        "ion",
        "electron",
        "manufacturer"
    ]

    system_settings = load_settings[0]

    for attribute in sys_attributes:

        assert hasattr(system_settings,attribute)
        assert getattr(system_settings,attribute) is not None, f"{attribute} is None in StageSettings"
        

def test_image_settings(load_settings):

    image_attributes = [
        "resolution",
        "dwell_time", 
        "hfw",
        "autocontrast",
        "beam_type",
        "save",
        "label",
        "gamma_enabled",
        "save_path",
        "reduced_area",
    ]

    image_settings = load_settings[1]

    for attribute in image_attributes:

        if attribute == "reduced_area":
            continue

        assert hasattr(image_settings,attribute), f"ImageSettings has no attribute {attribute}"
        assert getattr(image_settings,attribute) is not None, f"{attribute} is None in ImageSettings"


def test_milling_settings(load_settings):
    
    milling_attributes = [
        "milling_current",
        "spot_size",
        "rate",
        "dwell_time",
        "hfw",
        "patterning_mode", 
        "application_file",
    ]

    milling_settings = load_settings[2]

    for attribute in milling_attributes:

        assert hasattr(milling_settings,attribute), f"FibsemMillingSettings has no attribute {attribute}"
        assert getattr(milling_settings,attribute) is not None, f"{attribute} is None in FibsemMillingSettings"


def test_load_microscope_manufacturer():

    manufacturer = load_microscope_manufacturer()

    assert manufacturer in ["Tescan","Thermo","Demo"], f"{manufacturer} is Not Tescan, Thermo or Demo"


def test_hardware_settings(load_settings):


    hardware_attributes = [
        "electron_beam",
        "ion_beam",
        "stage_enabled",
        "stage_rotation",
        "stage_tilt",
        "manipulator_enabled",
        "manipulator_rotation",
        "manipulator_tilt",
        "gis_enabled",
        "gis_multichem",
    ]

    hardware_settings = load_settings[3]

    for attribute in hardware_attributes:

        assert hasattr(hardware_settings,attribute), f"FibsemHardware has no attribute {attribute}"
        assert getattr(hardware_settings,attribute) is not None, f"{attribute} is None in FibsemHardware"