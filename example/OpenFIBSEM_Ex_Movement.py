from fibsem import utils, acquire
from fibsem.structures import BeamType, FibsemStagePosition, FibsemPatternSettings, FibsemPattern
import matplotlib.pyplot as plt

def connect_to_microscope():

    #### connect to microscope ######
    microscope, settings = utils.setup_session()
    

    return microscope,settings

def absolute_move(microscope):

    print("Moving to stage position (Absolute Coordinates)")

    # Setting up desired location to move to (position in meters, rotation and tilt in radians)

    x = 0.01
    y = 0.01
    z = 0.01
    rotation = 0.2
    tilt = 0.2

    #### Moving to an absolute position ####
    new_position = FibsemStagePosition(x, y, z, rotation, tilt ) # metres and radians
    microscope.move_stage_absolute(new_position) 

    print("Moved to Absolute position")

def relative_move(microscope):

    print("Moving to offset stage coordinates")

    # setting up a relative move

    dx = 0.001
    dy = 0.002
    dz = 0.003
    dr = 0.2
    dt = 0.2


    #### Moving to a relative position ####
    relative_move = FibsemStagePosition(dx, dy, dz, dr, dt ) # metres and radians
    microscope.move_stage_relative(relative_move)

    print("Moved to position")

def main():

    microscope,settings = connect_to_microscope()
    
    input("Microscope Connected: Press enter to move stage (Absolute)")

    absolute_move(microscope=microscope)

    input("Press enter to move stage (Relative) ")

    relative_move(microscope=microscope)
    

   

if __name__ == "__main__":
    main()