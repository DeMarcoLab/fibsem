from fibsem import utils
from fibsem.structures import  FibsemStagePosition
import numpy as np
import logging


"""
This script demonstrates how to get the current stage position, and how to move the stage to a new position.

The basic movement methods are absolute_move and relative_move. 
- Relative move moves the stage by a certain amount in the current coordinate system.
- Absolute move moves the stage to a new position in the absolute coordinate system. 

This script will move the stage by 20um in the x direction (relative move), and then move back to the original position (absolute move).

Additional movement methods are available in the core api:
- Stable Move: the stage moves along the sample plane, accounting for stage tilt, and shuttle pre-tilt
- Vertical Move: the stage moves vertically in the chamber, regardless of tilt orientation

"""

def main():

    # connect to microscope
    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    
        # info about ImageSettings
    logging.info("---------------------------------- Current Position ----------------------------------\n")

    # get current position
    intial_position = microscope.get_stage_position()
    logging.info(f"\nStage Movement Example:")
    logging.info(f"Current stage position: {intial_position}")
    

    logging.info("\n---------------------------------- Relative Movement ----------------------------------\n")

    #### Moving to a relative position ####
    relative_move = FibsemStagePosition(x=20e-6,            # metres
                                        y=0,                # metres
                                        z=0.0,              # metres
                                        r=np.deg2rad(0),    # radians
                                        t=np.deg2rad(0))    # radians
    
    input(f"Press Enter to move by: {relative_move} (Relative)")
    
    # move by relative position    
    microscope.move_stage_relative(relative_move)
    current_position = microscope.get_stage_position()
    logging.info(f"After move stage position: {current_position}")


    logging.info("\n---------------------------------- Absolute Movement ----------------------------------\n")

    #### Moving to an absolute position ####
    stage_position = intial_position # move back to initial position

    # uncomment this if you want to move to a different position 
    # be careful to define a safe position to move too
    # relative_move = FibsemStagePosition(x=0,                # metres
    #                                     y=0,                # metres
    #                                 z=0.0,                  # metres
    #                                     r=np.deg2rad(0),    # radians
    #                                     t=np.deg2rad(0))    # radians

    input(f"Press Enter to move to: {stage_position} (Absolute)")

    # move to absolute position
    microscope.move_stage_absolute(stage_position) 
    current_position = microscope.get_stage_position()
    logging.info(f"After move stage position: {current_position}")


    logging.info("---------------------------------- End Example ----------------------------------")
   

if __name__ == "__main__":
    main()