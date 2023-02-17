

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
from fibsem import acquire, alignment, calibration, milling, movement, utils
from fibsem.structures import BeamType, MicroscopeState,  FibsemImage, FibsemStagePosition


@dataclass
class Lamella:
    state: MicroscopeState
    reference_image: FibsemImage
    path: Path

def main():

    PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "protocol_autolamella.yaml")
    microscope, settings = utils.setup_session(protocol_path=PROTOCOL_PATH)
    
    # move to the milling angle
    stage_position = FibsemStagePosition(
        r=np.deg2rad(settings.protocol["stage_rotation"]),
        t=np.deg2rad(settings.protocol["stage_tilt"])
    )
    microscope.move_stage_absolute(stage_position) # do need a safe version?

    # take a reference image    
    settings.image.label = "grid_reference"
    settings.image.beam_type = BeamType.ION
    settings.image.hfw = 900e-6
    acquire.take_reference_images(microscope, settings.image)

    # select positions
    sample: list[Lamella] = []
    lamella_no = 1
    settings.image.hfw = 80e-6

    while True:
        response = input(f"""Move to the desired position. 
        Do you want to select another lamella? [y]/n {len(sample)} selected so far.""")

        # store lamella information
        if response.lower() in ["", "y", "yes"]:
            
            # set filepaths
            path = os.path.join(settings.image.save_path, str(lamella_no))
            settings.image.save_path = path
            
            lamella = Lamella(
                state=microscope.get_current_microscope_state(),
                reference_image=acquire.new_image(microscope, settings.image),
                path = path
            )
            sample.append(lamella)
        else:
            break

    # sanity check
    if len(sample) == 0:
        logging.info(f"No lamella positions selected. Exiting.")
        return

    # setup milling
    milling.setup_milling(microscope = microscope,
        application_file = settings.system.application_file,
        patterning_mode  = "Serial",
        hfw = settings.image.hfw,
        mill_settings = settings.milling)

    # mill (fiducial, trench, thin, polish)
    for stage_no, milling_dict in enumerate(settings.protocol["lamella"]["protocol_stages"], 1):
        
        logging.info(f"Starting milling stage {stage_no}")

        lamella: Lamella
        for lamella_no, lamella in enumerate(sample):

            logging.info(f"Starting lamella {lamella_no}")

            # return to lamella
            microscope.set_microscope_state(lamella.state)

            # realign
            alignment.beam_shift_alignment(microscope, settings.image, lamella.reference_image)
                       
            if stage_no == 0:
                print("add microexpansion joints here")

            # mill trenches
            milling.draw_trench(microscope, milling_dict)
            milling.run_milling(microscope, milling_dict["milling_current"])
            milling.finish_milling(microscope)

            # retake reference image
            settings.image.save_path = lamella.path
            settings.image.label = f"ref_mill_stage_{stage_no}"
            lamella.reference_image = acquire.new_image(microscope, settings.image)
   
    logging.info(f"Finished autolamella: {settings.protocol['name']}")


if __name__ == "__main__":
    main()
