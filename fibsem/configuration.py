
import yaml
import os
from fibsem.config import CONFIG_PATH, DEFAULT_CONFIGURATION_VALUES

# TODO: 
# eucentric heights -> requires taking an image?
# reference rotation -> how to get this without having the user move to position?

def system_fingerprint(ip_address: str = "192.168.0.1", manufactuer: str = "Thermo") -> dict:
    """Automatically configure system based on microscope fingerprinting."""

    # load default configuration
    with open(os.path.join(CONFIG_PATH, "microscope-configuration.yaml"), "r") as f:
        config = yaml.safe_load(f)

    if manufactuer != "Thermo":
        raise ValueError("Only Thermo Fisher microscopes are currently supported for fingerprinting.")

    # ip,  manufacturer
    config["info"]["ip"] = ip_address               # default ip (support pc -> microscope pc)
    config["info"]["manufacturer"] = manufactuer    # default manufacturer

    from fibsem.microscope import ThermoMicroscope

    microscope = ThermoMicroscope()
    microscope.connect_to_microscope(config["info"]["ip"])

    # get name of system
    system_name = microscope.connection.service.system.name
    serial_number = microscope.connection.service.system.serial_number
    config["info"]["name"] = f"{system_name}-{serial_number}-configuration" 

    # stage
    # enabled, compustage, rotation, tilt
    stage_enabled = microscope.connection.specimen.stage.is_installed
    compustage_enabled = microscope.connection.specimen.stage.compustage.is_installed
    config["stage"]["enabled"] = stage_enabled or compustage_enabled
    config["stage"]["rotation"] = stage_enabled
    config["stage"]["tilt"] = stage_enabled or compustage_enabled

    # rotation reference, rotation_180
    if compustage_enabled: # TODO: check if this is accurate
        config["stage"]["rotation_reference"] = 0
        config["stage"]["rotation_180"] = 0

    # otherwise? I think need to get this manually?
    # TODO: get the user to move the stage to this position?
    # tilt = 0, rotation = same as when loading?

    # eucentric height map, [electron, ion] (otherwise we need to take an image to get this?)
    eucentric_height_map = {
        "Aquilios": [7e-3, 16.5e-3],
        "Helios": [4e-3, 16.5e-3],
        "Artcis": [4e-3, 16.5e-3],
        "Scios": [4e-3, 10.0e-3]
    }

    # electron beam
    config["electron"]["enabled"] = microscope.connection.beams.electron_beam.is_installed
    config["electron"]["column_tilt"] = get_column_tilt(config["info"]["manufacturer"], "electron")
    config["electron"]["eucentric_height"] = eucentric_height_map[system_name][0]

    # ion beam
    # is_installed, column tilt, eucentric_height (req taking image), plasma, plasma gas
    config["ion"]["enabled"] = microscope.connection.beams.ion_beam.is_installed
    config["ion"]["column_tilt"] = get_column_tilt(config["info"]["manufacturer"], "ion")
    config["ion"]["eucentric_height"] = eucentric_height_map[system_name][1]

    # plasma enabled
    ion_plasma_enabled = False
    ion_plasma_gas = "None"
    if config["ion"]["enabled"]:

        # check if plasma enabled
        ion_plasma_enabled = hasattr(microscope.connection.beams.ion_beam.source, "plasma_gas")

        if ion_plasma_enabled:
            ion_plasma_gas = microscope.connection.beams.ion_beam.source.plasma_gas.value

    config["ion"]["plasma"] = ion_plasma_enabled
    config["ion"]["plasma_gas"] = ion_plasma_gas

    # manipulator
    config["manipulator"]["enabled"] = microscope.connection.specimen.manipulator.is_installed
    config["manipulator"]["rotation"] = False   # always?
    config["maniuplator"]["tilt"] = False       # always?

    # gis
    gis_enabled = len(microscope.connection.list_all_gis_ports())
    multichem_enabled = len(microscope.connection.list_all_multichem_ports())
    sputter_coater_enabled = microscope.connection.specimen.sputter_coater.is_installed

    config["gis"]["enabled"] = gis_enabled
    config["gis"]["multichem"] = multichem_enabled
    config["gis"]["sputter_coater"] = sputter_coater_enabled


    return config


def get_column_tilt(manufacturer: str, beam: str) -> int:

    beam = f"{beam}-column-tilt"
    if manufacturer not in DEFAULT_CONFIGURATION_VALUES:
        raise ValueError(f"Unknown manufacturer: {manufacturer}")

    if beam not in DEFAULT_CONFIGURATION_VALUES[manufacturer]:
        raise ValueError(f"Unknown beam: {beam}")

    return DEFAULT_CONFIGURATION_VALUES[manufacturer][beam]
    

def generate_configuration(user_config: dict) -> dict:
    # load yaml
    with open(os.path.join(CONFIG_PATH, "microscope-configuration.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # microscope
    config["info"]["name"] = user_config["name"]
    config["info"]["ip_address"] = user_config["ip_address"]
    config["info"]["manufacturer"] = user_config["manufacturer"]

    # stage
    config["stage"]["rotation_reference"] = user_config["rotation-reference"]
    config["stage"]["rotation_180"] = user_config["rotation-reference"] + 180
    config["stage"]["shuttle_pre_tilt"] = user_config["shuttle-pre-tilt"]

    # electron
    config["electron"]["eucentric_height"] = user_config["electron-beam-eucentric-height"]
    config["electron"]["column_tilt"] = get_column_tilt(config["info"]["manufacturer"], "electron")

    # ion
    config["ion"]["eucentric_height"] = user_config["ion-beam-eucentric-height"]
    config["ion"]["column_tilt"] = get_column_tilt(config["info"]["manufacturer"], "ion")

    return config


def get_user_config() -> dict:

    # enter name
    name = input("Enter configuration name: ")
    

    # enter ip address
    ip_address = input("Enter microscope PC IP address: ")
    assert ip_address.count(".") == 3, "IP address must be in the form: XXX.XXX.XXX.XXX"

    # enter manufacturer
    manufacturer = input("Enter microscope manufacturer (Thermo, Tescan, Demo): ")
    assert manufacturer in ["Thermo", "Tescan", "Demo"], "Unknown manufacturer. Must be Thermo, Tescan or Demo"

    # enter rotation reference
    rotation_reference = input("Enter rotation reference (degrees): ")
    rotation_reference = float(rotation_reference)
    # enter shuttle pre-tilt
    shuttle_pre_tilt = input("Enter shuttle pre-tilt (degrees): ")
    shuttle_pre_tilt = float(shuttle_pre_tilt)

    # enter electron beam eucentric height
    electron_beam_eucentric_height = input("Enter electron beam eucentric height (metres): ")
    electron_beam_eucentric_height = float(electron_beam_eucentric_height)

    # enter ion beam eucentric height
    ion_beam_eucentric_height = input("Enter ion beam eucentric height (metres): ")
    ion_beam_eucentric_height = float(ion_beam_eucentric_height)
    
    # generate configuration
    user_config = {
        "name":                           name,                          # a descriptive name for your configuration 
        "ip_address":                     ip_address,                    # the ip address of the microscope PC
        "manufacturer":                   manufacturer,                  # the microscope manufactuer, Thermo, Tescan or Demo                       
        "rotation-reference":             rotation_reference,            # the reference rotation value (rotation when loading)  [degrees]
        "shuttle-pre-tilt":               shuttle_pre_tilt,              # the pre-tilt of the shuttle                           [degrees]
        "electron-beam-eucentric-height": electron_beam_eucentric_height,# the eucentric height of the electron beam             [metres]
        "ion-beam-eucentric-height":      ion_beam_eucentric_height,     # the eucentric height of the ion beam                  [metres]
    }
    return user_config

def save_configuration(config: dict, path: str):
    """Save a configuration to a yaml file."""
    
    # save yaml
    filename = os.path.join(path, f"{config['info']['name']}.yaml")
    with open(filename, "w") as f:
        yaml.dump(config, f)

    return filename

import argparse
def gen_config_cli(path: str = None):
    """Generate a configuration from the command line."""
    
    parser = argparse.ArgumentParser(description="Generate a configuration from the command line.")
    parser.add_argument("--path", type=str, help="The path to save the configuration to.")
    args = parser.parse_args()
    
    if path is None:
        path = args.path

    # check path exists
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    print(f"OpenFIBSEM Configuration Generator\n")

    user_config = get_user_config()
    
    print(f"\nGenerating configuration: {user_config['name']}")
    config = generate_configuration(user_config)
    print("Configuration generated successfully.")

    filename = save_configuration(config, path)
    print(f"Configuration saved to: {filename}")


if __name__ == "__main__":
    gen_config_cli()