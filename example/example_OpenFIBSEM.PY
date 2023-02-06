

from fibsem import utils, acquire
from fibsem.structures import BeamType, FibsemStagePosition, FibsemPatternSettings, FibsemPattern
from fibsem.FibsemMilling import milling_protocol

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg', force=True) # Activate 'agg' backend for off-screen plotting.


def main():

    #### connect to microscope ######
    microscope, settings = utils.setup_session()
    image_settings = settings.image 

    ##### take image with both beams #####
    ref_eb_image, ref_ib_image = acquire.take_reference_images(microscope, image_settings)

    ##### take image with only electron beam #####
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = acquire.new_image(microscope, image_settings)

    ##### take image with only ion beam #####
    image_settings.beam_type = BeamType.ION
    ib_image = acquire.new_image(microscope, image_settings)

    # show images

    fig, ax = plt.subplots(2, 2, figsize=(7, 5))
    ax[0].set_title("Reference Electron Beam Image")
    ax[1].set_title("Reference Ion Beam Image")
    ax[0].imshow(ref_eb_image.data, cmap="gray")
    ax[1].imshow(ref_ib_image.data, cmap="gray")
    ax[2].set_title("Individual Electron Beam Image")
    ax[3].set_title("Individual Ion Beam Image")
    ax[2].imshow(eb_image.data, cmap="gray")
    ax[3].imshow(ib_image.data, cmap="gray")
    plt.show()

    #### Moving to an absolute position ####
    new_position = FibsemStagePosition(x=0.010, y=0.010, z=0.010, r= 0.2, t = 0.2) # metres and radians
    microscope.move_stage_absolute(new_position) 

    #### Moving to a relative position ####
    relative_move = FibsemStagePosition(x=0.001, y=0.002, z=0.003, r= 0.2, t = 0.2) # metres and radians
    microscope.move_stage_relative(relative_move)

    #### Milling a rectangle and a line ####
    mill_settings = settings.milling
    mill_pattern_rec = FibsemPatternSettings(
        pattern = FibsemPattern.Rectangle,
        width = 10.0e-6,
        height = 10.0e-6,
        depth = 2.0e-6,
        rotation = 0.0,
        center_x = 0.0,
        center_y = 0.0,
    )

    mill_pattern_line = FibsemPatternSettings(
        pattern = FibsemPattern.Line,
        start_x = 0.0,
        start_y = 0.0,
        end_x = 10.0e-6,
        end_y = 10.0e-6,
        depth = 2.0e-6,
    )

    pattern_list = [mill_pattern_rec, mill_pattern_line]

    milling_protocol(
        microscope = microscope, 
        image_settings = image_settings, 
        mill_settings = mill_settings, 
        application_file = settings.system.application_file,
        patterning_mode = "Serial",
        pattern_settings = pattern_list
        )




if __name__ == "__main__":
    main()
