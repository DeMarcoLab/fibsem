# slice and view

import os
import logging
from pprint import pprint

import fibsem
from fibsem import acquire, milling, utils, alignment
from fibsem.structures import ImageSettings, FibsemPattern, FibsemPatternSettings
import numpy as np

def main():

    PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "protocol_slice_and_view.yaml")
    microscope, settings = utils.setup_session(protocol_path = PROTOCOL_PATH)

    # setup for milling
    milling.setup_milling(microscope = microscope,
        application_file = "Si",
        patterning_mode  = "Serial",
        hfw = settings.image.hfw,
        mill_settings = settings.milling)

    # angle correction
    #microscope.beams.electron_beam.angular_correction.angle.value = np.deg2rad(-38)

    settings.image.label = "reference"
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)

    ref_eb, ref_ib = None, None

    for slice_idx in range(int(settings.protocol["steps"])):

        # slice
        logging.info("------------------------ SLICE ------------------------")
        milling_settings = settings.milling #FibsemMillingSettings.__from_dict__(settings.protocol["milling"])       
        pattern_settings = FibsemPatternSettings(
            pattern = FibsemPattern.Rectangle,
            width = settings.protocol["milling"]["width"],
            height = settings.protocol["milling"]["height"],
            depth = settings.protocol["milling"]["depth"],
            scan_direction= settings.protocol["milling"]["scan_direction"],
            cleaning_cross_section= settings.protocol["milling"]["cleaning_cross_section"]
            )
        patterns = milling.draw_rectangle(microscope, pattern_settings)
        # estimated_milling_time = milling.estimate_milling_time_in_seconds([patterns])
        # logging.info(f"Estimated milling time: {estimated_milling_time}")
        milling.run_milling(microscope, milling_current=milling_settings.milling_current)


        milling.finish_milling(microscope, settings.pro)

        # view
        logging.info("------------------------ VIEW ------------------------")
        settings.image.label = f"slice_{slice_idx}"
        eb_image, ib_image = acquire.take_reference_images(microscope, settings.user.imaging)


        # align
        # if ref_eb is not None:
        #     alignment.align_using_reference_images(microscope, settings, ref_eb, eb_image)
        #     ref_eb, ref_ib = eb_image, ib_image

    

if __name__ == "__main__":
    main()
