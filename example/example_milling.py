from fibsem import utils
from fibsem.structures import FibsemMillingSettings, FibsemRectangleSettings, FibsemLineSettings
from fibsem import milling
import logging

"""
This script demonstrates how to use the milling module to mill a rectangle and two lines.

The script will:
    - connect to the microscope
    - setup milling
    - draw a rectangle and two lines
    - run milling
    - finish milling (restore ion beam current)

"""

def main():

    # connect to microscope
    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    # rectangle pattern
    rectangle_pattern = FibsemRectangleSettings(
        width = 10.0e-6,
        height = 10.0e-6,
        depth = 2.0e-6,
        rotation = 0.0,
        center_x = 0.0,
        center_y = 0.0,
    )

    # line pattern one
    line_pattern_01 = FibsemLineSettings(
        start_x = 0.0,
        start_y = 0.0,
        end_x = 10.0e-6,
        end_y = 10.0e-6,
        depth = 2.0e-6,
    )

    # line pattern two (mirror of line pattern one)
    line_pattern_02 = line_pattern_01
    line_pattern_02.end_y = -line_pattern_01.end_y

    logging.info(f"""\nMilling Pattern Example: """)

    logging.info(f"The current milling settings are: \n{settings.milling}")
    logging.info(f"The current rectangle pattern is \n{rectangle_pattern}")
    logging.info(f"The current line pattern one is \n{line_pattern_01}")
    logging.info(f"The current line pattern two is \n{line_pattern_02}")
    logging.info("---------------------------------- Milling ----------------------------------\n")
    # setup patterns in a list
    patterns = [rectangle_pattern, line_pattern_01, line_pattern_02]

    # setup milling
    milling.setup_milling(microscope, settings.milling)

    # draw patterns
    for pattern in patterns:
        milling.draw_pattern(microscope, pattern)

    # run milling
    milling.run_milling(microscope, settings.milling.milling_current, milling_voltage=settings.milling.milling_voltage)

    # finish milling
    milling.finish_milling(microscope, microscope.system.ion.beam.beam_current)


if __name__ == "__main__":
    main()