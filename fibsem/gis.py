import logging

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemGasInjectionSettings

gis_protocol = {
    "application_file": "cryo_Pt_dep",
    "gas": "Pt cryo",
    "position": "cryo",
    "hfw": 3.0e-05 ,
    "length": 7.0e-06,
    "beam_current": 1.0e-8,
    "time": 30.0,
}

def deposit_platinum(
    microscope: FibsemMicroscope,
    protocol: dict = None,
    default_application_file: str = "Si",
):
    """Deposit platinum over the sample.

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


def cryo_deposition(microscope: FibsemMicroscope, protocol: dict = None, name: str = None):

    # get current position
    position = microscope.get_microscope_state().stage_position

    # move to deposition position
    if name is not None:
        
        # move to position
        from fibsem import utils
        deposition_position = utils._get_position(name)
        
        if deposition_position is None:
            raise RuntimeError(f"Position {name} requested but not found")
        
        logging.info(f"Moving to depositon position: {name}")
        microscope.safe_absolute_stage_movement(deposition_position)


    # move down
    from fibsem.structures import FibsemStagePosition
    microscope.move_stage_relative(FibsemStagePosition(z=-1e-3))

    # sputter
    deposit_platinum(microscope, protocol)

    # return to previous position
    microscope.safe_absolute_stage_movement(position)

def cryo_deposition_v2(microscope: FibsemMicroscope, gis_settings: FibsemGasInjectionSettings, name: str = None, move_down: bool = True):

    # get current position
    position = microscope.get_microscope_state().stage_position

    # move to deposition position
    if name is not None:
        
        # move to position
        from fibsem import utils
        deposition_position = utils._get_position(name)
        
        if deposition_position is None:
            raise RuntimeError(f"Position {name} requested but not found")
        
        logging.info(f"Moving to depositon position: {name}")
        microscope.safe_absolute_stage_movement(deposition_position)

    # move down 
    if move_down:
        from fibsem.structures import FibsemStagePosition
        microscope.move_stage_relative(FibsemStagePosition(z=-1e-3))

    # cryo deposition
    microscope.cryo_deposition_v2(gis_settings)

    # return to previous position
    microscope.safe_absolute_stage_movement(position)

# # multi-chem dep
# # move the stage to deposition position

# # insert the multichem
# # multichem.insert()

# microscope.connection.patterning.set_default_beam_type(1) # SEM
# microscope.connection.patterning.set_default_application_file("W_M e")
# microscope.clear_patterns()
# line_pattern = microscope.connection.patterning.create_line(0, 0, 10e-6, 0, 1e-6)

# line_pattern.application_file = "W_M e"
# line_pattern.gas_type = "CRYO Pt "
# line_pattern.gas_flow = [80]
# line_pattern.time = 30

# microscope.connection.patterning.start()
# # microscope.connection.patterning.start()

# # retract
# # multichem.retract()

# microscope.connection.patterning.set_default_beam_type(2) # ION
# microscope.connection.patterning.set_default_application_file("Si")