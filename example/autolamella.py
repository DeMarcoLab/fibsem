

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
from autoscript_sdb_microscope_client.structures import (AdornedImage,
                                                         StagePosition)
from fibsem import acquire, alignment, calibration, milling, movement, utils
from fibsem.structures import BeamType, MicroscopeState, MillingSettings


@dataclass
class Lamella:
    state: MicroscopeState
    reference_image: AdornedImage
    path: Path

def main():

    PROTOCOL_PATH = os.path.join(os.path.dirname(__file__), "protocol_autolamella.yaml")
    microscope, settings = utils.setup_session(protocol_path=PROTOCOL_PATH)
    
    # move to the milling angle
    stage_position = StagePosition(
        r=np.deg2rad(settings.protocol["stage_rotation"]),
        t=np.deg2rad(settings.protocol["stage_tilt"])
    )
    movement.safe_absolute_stage_movement(microscope, stage_position)

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
            
            # TODO: fiducial selection
            # milling._draw_fiducial_patterns(microscope, MillingSettings.__from_dict__(settings.protocol["fiducial"]))

            # set filepaths
            path = os.path.join(settings.image.save_path, str(lamella_no))
            settings.image.save_path = path
            
            lamella = Lamella(
                state=calibration.get_current_microscope_state(microscope),
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
    milling.setup_milling(microscope, settings.system.application_file)

    # mill (fiducial, trench, thin, polish)
    for stage_no, milling_dict in enumerate(settings.protocol["lamella"]["protocol_stages"], 1):
        
        logging.info(f"Starting milling stage {stage_no}")

        lamella: Lamella
        for lamella_no, lamella in enumerate(sample):

            logging.info(f"Starting lamella {lamella_no}")

            # return to lamella
            calibration.set_microscope_state(microscope, lamella.state)

            # realign
            alignment.beam_shift_alignment(microscope, settings.image, lamella.reference_image)
                       
            if stage_no == 0:
                print("add microexpansion joints here")

            # mill trenches
            milling._draw_trench_patterns(microscope, milling_dict)
            milling.run_milling(microscope, milling_dict["milling_current"])
            milling.finish_milling(microscope, settings.system.ion.current)

            # retake reference image
            settings.image.save_path = lamella.path
            settings.image.label = f"ref_mill_stage_{stage_no}"
            lamella.reference_image = acquire.new_image(microscope, settings.image)
   
    logging.info(f"Finished autolamella: {settings.protocol['name']}")


if __name__ == "__main__":
    main()
