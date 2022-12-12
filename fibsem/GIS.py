from autoscript_sdb_microscope_client import SdbMicroscopeClient
import logging
import time
from fibsem.structures import BeamType

def sputter_platinum(
    microscope: SdbMicroscopeClient,
    protocol: dict,
    whole_grid: bool = False,
    default_application_file: str = "autolamella",
):
    """Sputter platinum over the sample.

    Args:
        microscope (SdbMicroscopeClient): autoscript microscope instance
        protocol (dict): platinum protcol dictionary
        whole_grid (bool, optional): sputtering protocol. Defaults to False.

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
    original_active_view = microscope.imaging.get_active_view()
    microscope.imaging.set_active_view(BeamType.ELECTRON.value)
    microscope.patterning.clear_patterns()
    microscope.patterning.set_default_application_file(protocol["application_file"])
    microscope.patterning.set_default_beam_type(BeamType.ELECTRON.value)
    multichem = microscope.gas.get_multichem()
    multichem.insert(protocol["position"])
    multichem.turn_heater_on(protocol["gas"])  # "Pt cryo")
    time.sleep(3)

    # Create sputtering pattern
    microscope.beams.electron_beam.horizontal_field_width.value = hfw
    pattern = microscope.patterning.create_line(
        -line_pattern_length / 2,  # x_start
        +line_pattern_length,  # y_start
        +line_pattern_length / 2,  # x_end
        +line_pattern_length,  # y_end
        2e-6,
    )  # milling depth
    pattern.time = sputter_time + 0.1

    # Run sputtering
    microscope.beams.electron_beam.blank()
    if microscope.patterning.state == "Idle":
        logging.info("Sputtering with platinum for {} seconds...".format(sputter_time))
        microscope.patterning.start()  # asynchronous patterning
        time.sleep(sputter_time + 5)
    else:
        raise RuntimeError("Can't sputter platinum, patterning state is not ready.")
    if microscope.patterning.state == "Running":
        microscope.patterning.stop()
    else:
        logging.warning("Patterning state is {}".format(microscope.patterning.state))
        logging.warning("Consider adjusting the patterning line depth.")

    # Cleanup
    microscope.patterning.clear_patterns()
    microscope.beams.electron_beam.unblank()
    microscope.patterning.set_default_application_file(default_application_file)
    microscope.imaging.set_active_view(original_active_view)
    microscope.patterning.set_default_beam_type(BeamType.ION.value)  # set ion beam
    multichem.retract()
    logging.info("sputtering platinum finished.")