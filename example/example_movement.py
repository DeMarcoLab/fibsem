from fibsem import utils
from fibsem.structures import  FibsemStagePosition
import numpy as np
import logging


"""
This script demonstrates how to get the current stage position, and how to move the stage to a new position.

The basic movement methods are absolute_move and relative_move. 
- Absolute move moves the stage to a new position in the absolute coordinate system. 
- Relative move moves the stage by a certain amount in the current coordinate system.

"""

def main():

    # connect to microscope
    microscope, settings = utils.setup_session()
    
        # info about ImageSettings
    logging.info("---------------------------------- Current Position ----------------------------------\n")

    # get current position
    current_position = microscope.get_stage_position()
    logging.info(f"\nStage Movement Example: \nFibsemStagePosition: {current_position.__doc__}")
    logging.info(f"Current stage position: {current_position}")

    logging.info("---------------------------------- Absolute Movement ----------------------------------\n")

    # position to move to
    stage_position = FibsemStagePosition(x=0.001,            # metres
                                         y=0.002,            # metres
                                         z=0.005,            # metres
                                         r=np.deg2rad(0),    # radians
                                         t=np.deg2rad(0))    # radians
    
    input(f"Press Enter to move to: {stage_position} (Absolute)")

    # move to absolute position
    microscope.move_stage_absolute(stage_position) 
    current_position = microscope.get_stage_position()
    logging.info(f"After move stage position: {current_position}")
    

    logging.info("---------------------------------- Relative Movement ----------------------------------\n")

    #### Moving to a relative position ####
    relative_move = FibsemStagePosition(x=0.001,            # metres
                                        y=0.002,            # metres
                                        z=0.0,              # metres
                                        r=np.deg2rad(30),   # radians
                                        t=np.deg2rad(10))   # radians
    
    input(f"Press Enter to move by: {relative_move} (Relative)")
    
    # move by relative position    
    microscope.move_stage_relative(relative_move)
    current_position = microscope.get_stage_position()
    logging.info(f"After move stage position: {current_position}")

    logging.info("---------------------------------- End Example ----------------------------------")
   

if __name__ == "__main__":
    main()