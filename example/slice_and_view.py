# slice and view

import os
import logging
from pprint import pprint

import fibsem
from fibsem import acquire, milling, utils, alignment
from fibsem.structures import BeamType, ImageSettings, FibsemRectangleSettings
import numpy as np

def main():

    PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "protocol_slice_and_view.yaml")
    microscope, settings = utils.setup_session(protocol_path = PROTOCOL_PATH)

    # setup for milling
    milling.setup_milling(microscope = microscope,
        patterning_mode  = "Serial",
        mill_settings = settings.milling)
    
    pattern_settings = FibsemRectangleSettings(
        width = settings.protocol["milling"]["width"],
        height = settings.protocol["milling"]["height"],
        depth = settings.protocol["milling"]["depth"],
        scan_direction= settings.protocol["milling"]["scan_direction"],
        cleaning_cross_section= settings.protocol["milling"]["cleaning_cross_section"]
    )

    # angle correction
    microscope.set("angular_correction_tilt_correction", True)
    microscope.set("angular_correction_angle", np.deg2rad(-38))

    # update image settings
    settings.image.filename = "reference"
    settings.image.save = True
    settings.image.beam_type = BeamType.ELECTRON

    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    ref_eb, ref_ib = None, None

    for slice_idx in range(int(settings.protocol["steps"])):

        # slice
        logging.info("------------------------ SLICE ------------------------")
        milling_settings = settings.milling  

        patterns = milling.draw_pattern(microscope, pattern_settings)
        # estimated_milling_time = milling.estimate_milling_time_in_seconds([patterns])
        # logging.info(f"Estimated milling time: {estimated_milling_time}")
        milling.run_milling(microscope, milling_current=milling_settings.milling_current)
        milling.finish_milling(microscope, settings.system.ion.current)

        # view
        logging.info("------------------------ VIEW ------------------------")
        settings.image.filename = f"slice_{slice_idx:04d}"
        eb_image = acquire.new_image(microscope, settings.image)


        # align
        if ref_eb is not None:
            alignment.align_using_reference_images(microscope, settings, ref_eb, eb_image)
            ref_eb = eb_image

    
        # update patterns
        pattern_settings.centre_y += settings.protocol["step_size"]


if __name__ == "__main__":
    main()
