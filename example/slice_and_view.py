# slice and view

import os
import logging
from pprint import pprint

import fibsem
from fibsem import acquire, milling, utils
from fibsem.structures import ImageSettings, MillingSettings


def main():

    PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "protocol_slice_and_view.yaml")
    microscope, settings = utils.setup_session(protocol_path = PROTOCOL_PATH)


    # slice
    logging.info("------------------------ SLICE ------------------------")
    milling_settings = MillingSettings.__from_dict__(settings.protocol["milling"])
    pprint(milling_settings)

    # milling.setup_milling(
    #     microscope, settings["system"]["application_file"], hfw=150e-6
    # )
    # patterns = milling._draw_rectangle_pattern_v2(microscope, milling_settings)
    # milling.run_milling(microscope, milling_current=protocol["milling_current"])

    # view
    logging.info("------------------------ VIEW ------------------------")
    slice_idx = 0
    settings.image_settings.label = f"slice_{slice_idx}"
    # eb_image, ib_image = acquire.take_reference_images(microscope, settings.image_settings)
    print(settings.image_settings)
    
    # move?

    # align?

    


if __name__ == "__main__":
    main()
