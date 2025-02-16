import logging

from fibsem import milling, utils
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemRectangleSettings,
)
from fibsem.milling import FibsemMillingStage, MillingAlignment
from fibsem.milling.patterning.patterns2 import RectanglePattern, Point

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


    # setup milling stage (settings and alignment)
    stage = FibsemMillingStage(
        milling=FibsemMillingSettings(
            milling_current=2.0e-9,
            milling_voltage=30.0e3,
            hfw=100e-6,
            application_file="Si",
            patterning_mode="Serial",
        ),
        alignment=MillingAlignment(
            enabled=False
        )
    )

    # rectangle
    rectangle_shape = FibsemRectangleSettings(
        width = 10.0e-6,
        height = 10.0e-6,
        depth = 2.0e-6,
        rotation = 0.0,
        centre_x = 0.0,
        centre_y = 0.0,
    )

    # circle
    circle_shape = FibsemCircleSettings(
        radius = 10.0e-6,
        depth = 2.0e-6,
        centre_x = 10e-6,
        centre_y = 10e-6,
    )

    # line pattern
    line_shape = FibsemLineSettings(
        start_x = 0.0,
        start_y = 0.0,
        end_x = 10.0e-6,
        end_y = 10.0e-6,
        depth = 1.0e-6,
    )

    logging.info(f"""\nMilling Pattern Example: """)
    logging.info(f"The current milling settings are: \n{settings.milling}")
    logging.info(f"The current rectangle pattern is \n{rectangle_shape}")
    logging.info(f"The current circle pattern ins is \n{circle_shape}")
    logging.info(f"The current line pattern is \n{line_shape}")
    logging.info("---------------------------------- Milling ----------------------------------\n")
    # setup patterns in a list
    patterns = [rectangle_shape, circle_shape, line_shape]

    milling.setup_milling(microscope, stage)

    # draw patterns
    milling.draw_patterns(microscope, patterns)

    # run milling
    milling.run_milling(microscope, stage.milling.milling_current, milling_voltage=stage.milling.milling_voltage)

    # finish milling
    milling.finish_milling(microscope, microscope.system.ion.beam.beam_current)


    rect = RectanglePattern(
        width=10e-6,
        height=10e-6,
        depth=2e-6,
        rotation=0,
        point=Point(0, 0)
    )
    stage.pattern = rect

    milling.mill_stages(microscope, stage)

if __name__ == "__main__":
    main()
