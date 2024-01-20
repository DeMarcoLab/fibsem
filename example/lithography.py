import os
from pprint import pprint

import numpy as np
from autoscript_sdb_microscope_client.structures import (
    BitmapPatternDefinition, StagePosition)
from fibsem import acquire, milling, movement, utils
# from fibsem.ui import windows
from fibsem.structures import BeamType
from PIL import Image

BASE_PATH = os.path.dirname(__file__)

def save_profile_to_bmp(arr: np.ndarray, fname: str = "profile.bmp"):

    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # save profile
    Image.fromarray(arr).convert("RGB").save(fname)


def main():

    PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol_lithography.yaml")
    microscope, settings = utils.setup_session(protocol_path=PROTOCOL_PATH)

    # lens plane
    microscope.system.electron.column_tilt = 0

    # move to the milling angle
    # stage_position = StagePosition(
    #     r=np.deg2rad(settings.protocol["stage_rotation"]),
    #     t=np.deg2rad(settings.protocol["stage_tilt"]),
    # )
    # movement.safe_absolute_stage_movement(microscope, stage_position)

    microscope.move_flat_to_beam(BeamType.ION)

    # eucentric, select position
    # windows.ask_user_movement(
    #     microscope,
    #     settings,
    #     msg_type="eucentric",
    #     msg="Select a position to mill the pattern.",
    # )

    # lens profile files
    npy_path = os.path.join(BASE_PATH, settings.protocol["profile"])
    bmp_path = os.path.join(BASE_PATH, "profile.bmp")

    # load milling properties
    profile = np.load(npy_path)
    pixel_size = settings.protocol["pixelsize"]
    lens_height = settings.protocol["milling"]["height"]
    lens_width = settings.protocol["milling"]["width"]

    # save profile to bmp
    save_profile_to_bmp(profile, bmp_path)

    # load bmp pattern
    bitmap_pattern = BitmapPatternDefinition()
    bitmap_pattern.load(bmp_path)

    # milling setup
    milling.setup_milling(
        microscope, application_file=settings.protocol["application_file"]
    )

    # surface milling
    microscope.patterning.create_bitmap(
        center_x=0,
        centre_y=0,
        width=lens_width,
        height=lens_height,
        depth=settings.protocol["initial_depth"],
        bitmap_pattern_definition=bitmap_pattern,
    )

    # mill bitmap
    microscope.patterning.create_bitmap(
        center_x=0,
        centre_y=0,
        width=lens_width,
        height=lens_height,
        depth=settings.protocol["milling"]["milling_depth"], 
        bitmap_pattern_definition=bitmap_pattern,
    )
    milling.run_milling(microscope, settings.protocol["milling"]["milling_current"])
    milling.finish_milling(microscope)


if __name__ == "__main__":
    main()
