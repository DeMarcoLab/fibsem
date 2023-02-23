from fibsem import utils, acquire
from fibsem.structures import BeamType, FibsemStagePosition, FibsemPatternSettings, FibsemPattern
from fibsem.milling import milling_protocol
import matplotlib.pyplot as plt

def connect_to_microscope():

    #### connect to microscope ######
    microscope, settings = utils.setup_session()
    

    return microscope,settings


def milling_example(microscope,settings):

    image_settings = settings.image 

    #### Milling a rectangle and a line ####
    mill_settings = settings.milling

    # create a rectange shape pattern to be milled
    mill_pattern_rec = FibsemPatternSettings(
        pattern = FibsemPattern.Rectangle,
        width = 10.0e-6,
        height = 10.0e-6,
        depth = 2.0e-6,
        rotation = 0.0,
        center_x = 0.0,
        center_y = 0.0,
    )

    # create a line pattern to be milled
    mill_pattern_line = FibsemPatternSettings(
        pattern = FibsemPattern.Line,
        start_x = 0.0,
        start_y = 0.0,
        end_x = 10.0e-6,
        end_y = 10.0e-6,
        depth = 2.0e-6,
    )

    # setup patterns in a list
    pattern_list = [mill_pattern_rec, mill_pattern_line]

    # run the milling protocol to mill patterns created
    milling_protocol(
        microscope = microscope, 
        image_settings = image_settings, 
        mill_settings = mill_settings, 
        application_file = settings.system.application_file,
        patterning_mode = "Serial",
        pattern_settings = pattern_list
        )

def main():

    microscope, settings = connect_to_microscope()

    milling_example(microscope, settings)


if __name__ == "__main__":
    main()