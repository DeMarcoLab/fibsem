from fibsem.microscope import FibsemMicroscope
import logging
import time
from fibsem.structures import BeamType

def sputter_platinum(
    microscope: FibsemMicroscope,
    protocol: dict,
    whole_grid: bool = False,
    default_application_file: str = "autolamella",
):
    """Sputter platinum over the sample.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        protocol (dict): platinum protcol dictionary
        whole_grid (bool, optional): sputtering protocol. Defaults to False.

    Returns:
        None

    Raises:
        RuntimeError: Error Sputtering
    """

    # protcol = settings.protocol["platinum"] in old system
    # protocol = settings.protocol["platinum"] in new
    if whole_grid:
        sputter_time = protocol["whole_grid"]["time"]  # 20
        hfw = protocol["whole_grid"]["hfw"]  # 30e-6
        line_pattern_length = protocol["whole_grid"]["length"]  # 7e-6
        logging.info("sputtering platinum over the whole grid.")
    else:
        sputter_time = protocol["weld"]["time"]  # 20
        hfw = protocol["weld"]["hfw"]  # 100e-6
        line_pattern_length = protocol["weld"]["length"]  # 15e-6
        logging.info("sputtering platinum to weld.")

    # Setup
    microscope.setup_sputter(protocol=protocol)

    # Create sputtering pattern
    sputter_pattern = microscope.draw_sputter_pattern(hfw=hfw, line_pattern_length=line_pattern_length, sputter_time=sputter_time)

    # Run sputtering
    microscope.run_sputter(sputter_time=sputter_time, sputter_pattern=sputter_pattern)

    # Cleanup
    microscope.finish_sputter(application_file=default_application_file)
    