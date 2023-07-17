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
    "beam_current": 1.0e-8,
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
        default_application_file (str): application file to use if none is specified
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


def cryo_sputter(microscope: FibsemMicroscope, protocol: dict = None, name: str = None):

    # get current position
    position = microscope.get_current_microscope_state().absolute_position

    # move to sputter position
    if name is not None:
        
        # move to position
        from fibsem import utils
        sputter_position = utils._get_position(name)
        
        if sputter_position is None:
            raise RuntimeError(f"Position {name} requested but not found")
        
        logging.info(f"Moving to sputter position: {name}")
        microscope._safe_absolute_stage_movement(sputter_position)


    # move down
    from fibsem.structures import FibsemStagePosition
    microscope.move_stage_relative(FibsemStagePosition(z=-1e-3))

    # sputter
    sputter_platinum(microscope, protocol)

    # return to previous position
    microscope._safe_absolute_stage_movement(position)
