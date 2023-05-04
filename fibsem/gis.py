from fibsem.microscope import FibsemMicroscope
import logging
import time
from fibsem.structures import BeamType


gis_protocol = {
    "application_file": "cryo_Pt_dep",
    "gas": "Pt cryo",
    "position": "cryo",
    "hfw": 3.0e-05 ,
    "length": 7.0e-06,
    "time": 30.0,
}

def sputter_platinum(
    microscope: FibsemMicroscope,
    protocol: dict = None,
    default_application_file: str = "Si",
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

    if protocol is None:
        protocol = gis_protocol

    hfw = protocol["hfw"]
    line_pattern_length = protocol["length"]
    sputter_time = protocol["time"]
    

    # Setup
    microscope.setup_sputter(protocol=protocol)

    # Create sputtering pattern
    sputter_pattern = microscope.draw_sputter_pattern(hfw=hfw, line_pattern_length=line_pattern_length, sputter_time=sputter_time)

    # Run sputtering
    microscope.run_sputter(sputter_time=sputter_time, sputter_pattern=sputter_pattern)

    # Cleanup
    microscope.finish_sputter(application_file=default_application_file)
    