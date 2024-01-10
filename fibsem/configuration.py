
import yaml
import os
from fibsem.config import CONFIG_PATH



def get_column_tilt(manufacturer: str, beam: str) -> int:
    if beam == "electron-beam":
        return 0

    if manufacturer == "Thermo":
        return 52
    elif manufacturer == "Tescan":
        return 55
    elif manufacturer == "Demo":
        return 52
    else:
        raise ValueError("Unknown manufacturer")



user_config = {
    "name":                           "tfs-aquilos-01",              # a descriptive name for your configuration 
    "ip_address":                     "192.168.0.1",                 # the ip address of the microscope PC
    "manufacturer":                   "Thermo",                      # the microscope manufactuer, Thermo, Tescan or Demo                       
    "rotation-reference":             0,                             # the reference rotation value (rotation when loading)  [degrees]
    "shuttle-pre-tilt":               35,                            # the pre-tilt of the shuttle                           [degrees]
    "electron-beam-eucentric-height": 7.0e-3,                        # the eucentric height of the electron beam             [metres]
    "ion-beam-eucentric-height":      16.5e-3,                       # the eucentric height of the ion beam                  [metres]
}


def generate_configuration(user_config: dict) -> dict:
    # load yaml
    with open(os.path.join(CONFIG_PATH, "microscope-configuration.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # microscope
    config["core"]["name"] = user_config["name"]
    config["core"]["ip_address"] = user_config["ip_address"]
    config["core"]["manufacturer"] = user_config["manufacturer"]

    # stage
    config["stage"]["rotation-reference"] = user_config["rotation-reference"]
    config["stage"]["rotation-180"] = user_config["rotation-reference"] + 180
    config["stage"]["shuttle-pre-tilt"] = user_config["shuttle-pre-tilt"]

    # electron
    config["electron-beam"]["eucentric-height"] = user_config["electron-beam-eucentric-height"]
    config["electron-beam"]["column-tilt"] = get_column_tilt(config["core"]["manufacturer"], "electron-beam")

    # ion
    config["ion-beam"]["eucentric-height"] = user_config["ion-beam-eucentric-height"]
    config["ion-beam"]["column-tilt"] = get_column_tilt(config["core"]["manufacturer"], "ion-beam")

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
    electron_beam_eucentric_height = input("Enter electron beam eucentric height (degrees): ")
    electron_beam_eucentric_height = float(electron_beam_eucentric_height)

    # enter ion beam eucentric height
    ion_beam_eucentric_height = input("Enter ion beam eucentric height (degrees): ")
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
    filename = os.path.join(path, f"{config['core']['name']}.yaml")
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