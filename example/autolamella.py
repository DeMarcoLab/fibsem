

import os
import fibsem
from fibsem import utils, milling
from fibsem.structures import MicroscopeState, ImageSettings

from pprint import pprint

def main():

    print("autolamella")

    # load settings
    PATH = os.path.join(fibsem.__file__, "config", "system.yaml")
    settings = utils.load_yaml(PATH)
    protocol = utils.load_yaml(os.path.join(os.path.dirname(__file__), "protocol_autolamella.yaml"))

    # connect to microscope
    microscope = utils.connect_to_microscope(settings["ip_address"])


    # select lamella

    
    # setup milling

    milling.setup_milling(microscope, settings["application_file"])

    # mill (fiducial, trench, thin, polish)
    for mill_settings in protocol["lamella"]["protocol_stages"]:

        pprint(mill_settings)

        milling.mill_trench_patterns(microscope, mill_settings)
        milling.run_milling(microscope, mill_settings["milling_current"])
        

    # TODO: reference images
    
    print("finished autolamella")