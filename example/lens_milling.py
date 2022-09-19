import os
from fibsem import utils, acquire, milling, movement
import numpy as np
from autoscript_sdb_microscope_client.structures import StagePosition

from PIL import Image
from pprint import pprint

from autoscript_sdb_microscope_client.structures import BitmapPatternDefinition
from fibsem.ui import windows

BASE_PATH = os.path.dirname(__file__)

def save_profile_to_bmp(arr: np.ndarray, fname: str = "profile.bmp"):

    # scale values to int
    arr = (arr / np.max(arr)) * 255
    arr = arr.astype(np.uint8)

    # save profile
    Image.fromarray(arr).convert("RGB").save(fname)


def main():

    PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol_lens_milling.yaml")
    microscope, settings = utils.setup_session(protocol_path=PROTOCOL_PATH)

    # lens plane
    settings.system.stage.tilt_flat_to_electron = 0

    # move to the milling angle
    stage_position = StagePosition(
        r=np.deg2rad(settings.protocol["stage_rotation"]),
        t=np.deg2rad(settings.protocol["stage_tilt"]),
    )
    # movement.safe_absolute_stage_movement(microscope, stage_position)

    # eucentric, select position
    windows.ask_user_movement(
        microscope,
        settings,
        msg_type="eucentric",
        msg="Select a position to mill the lens.",
    )

    # get centre position
    # align chip (rotation)

    # set position
    # offset (distance)
    # centre (middle of 4 points)

    # lens profile files
    npy_path = os.path.join(BASE_PATH, settings.protocol["profile"])
    bmp_path = os.path.join(BASE_PATH, "profile.bmp")

    # load lens properties
    lens_profile = np.load(npy_path)
    pixel_size = settings.protocol["pixelsize"]
    lens_height = lens_profile.shape[0] * pixel_size
    lens_width = lens_profile.shape[1] * pixel_size

    # save profile to bmp
    save_profile_to_bmp(lens_profile, bmp_path)

    # load bmp pattern
    bitmap_pattern = BitmapPatternDefinition()
    bitmap_pattern.load(bmp_path)

    # milling setup
    milling.setup_milling(
        microscope, application_file=settings.protocol["application_file"]
    )

    # initial exposure
    microscope.patterning.create_bitmap(
        center_x=0,
        centre_y=0,
        width=lens_width,
        height=lens_height,
        depth=settings.protocol["initial_depth"],
        bitmap_pattern_definition=bitmap_pattern,
    )
    milling.run_milling(microscope, settings.protocol["milling"]["milling_current"])
    milling.finish_milling(microscope)

    # mill lens
    microscope.patterning.create_bitmap(
        center_x=0,
        centre_y=0,
        width=lens_width,
        height=lens_height,
        depth=None,  # TODO: how does depth work for a bitmap?
        bitmap_pattern_definition=bitmap_pattern,
    )
    milling.run_milling(microscope, settings.protocol["milling"]["milling_current"])
    milling.finish_milling(microscope)

    # TODO: stitching, initial selection helpers
    # stitching (1d ) -> x axis
    # move by width
    # cross correlate


if __name__ == "__main__":
    main()
