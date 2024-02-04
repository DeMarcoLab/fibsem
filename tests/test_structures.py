import pytest

from fibsem.structures import MicroscopeState, BeamType, ImageSettings, FibsemImage, FibsemRectangle, BeamSettings, FibsemDetectorSettings, FibsemStagePosition

import datetime
# microscope state


# electron_beam, electron_detector, ion_beam, ion_detector are now optional


def test_microscope_state():

    state = MicroscopeState()

    state.to_dict()

    state.electron_beam = None
    state.electron_detector = None

    state.to_dict()

    state.ion_beam = None
    state.ion_detector = None

    state.to_dict()
    
