

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
from fibsem import acquire, alignment, milling, patterning, utils
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
    settings.image.filename = "grid_reference"
    settings.image.beam_type = BeamType.ION
    settings.image.hfw = 900e-6
    settings.image.save = True
    acquire.take_reference_images(microscope, settings.image)

    # select positions
    experiment: list[Lamella] = []
    lamella_no = 1
    settings.image.hfw = 80e-6
    base_path = settings.image.path

    while True:
        response = input(f"""Move to the desired position. 
        Do you want to select another lamella? [y]/n {len(experiment)} selected so far.""")

        # store lamella information
        if response.lower() in ["", "y", "yes"]:
            
            # set filepaths
            path = os.path.join(base_path, f"{lamella_no:02d}")
            settings.image.path = path
            settings.image.filename = f"ref_lamella"
            acquire.take_reference_images(microscope, settings.image)

            lamella = Lamella(
                state=microscope.get_microscope_state(),
                reference_image=acquire.new_image(microscope, settings.image),
                path = path
            )
            experiment.append(lamella)
            lamella_no += 1
        else:
            break

    # sanity check
    if len(experiment) == 0:
        logging.info(f"No lamella positions selected. Exiting.")
        return

    # mill (rough, thin, polish)
    workflow_stages = ["rough", "thin", "polish"]
    for stage_no, stage_name in enumerate(workflow_stages):
        
        logging.info(f"Starting milling stage {stage_no}")

        lamella: Lamella
        for lamella_no, lamella in enumerate(experiment):

            logging.info(f"Starting lamella {lamella_no:02d}")

            # return to lamella
            microscope.set_microscope_state(lamella.state)

            # realign
            alignment.beam_shift_alignment(microscope, settings.image, lamella.reference_image)
                       
            if stage_no == 0:
                microexpansion_stage = patterning.get_milling_stages("microexpansion", settings.protocol)
                milling.mill_stage(microscope, settings, microexpansion_stage[0])

            # get trench milling pattern, and mill
            trench_stage = patterning.get_milling_stages("lamella", settings.protocol)[stage_no]
            milling.mill_stage(microscope, settings, trench_stage)

            # retake reference image
            settings.image.path = lamella.path
            settings.image.filename = f"ref_mill_stage_{stage_no:02d}"
            lamella.reference_image = acquire.new_image(microscope, settings.image)

            if stage_no == 3:
                # take final reference images
                settings.image.filename = f"ref_final"
                acquire.take_reference_images(microscope, settings.image)
   
    logging.info(f"Finished autolamella: {settings.protocol['name']}")


if __name__ == "__main__":
    main()
