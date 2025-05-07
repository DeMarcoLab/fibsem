import pytest

from fibsem import utils
from fibsem.structures import BeamType


def test_microscope():
    """Test get/set microscope functions."""

    microscope, settings = utils.setup_session(manufacturer="Demo")

    hfw = 150e-6
    microscope.set_field_of_view(hfw, BeamType.ELECTRON)
    assert microscope.get_field_of_view(BeamType.ELECTRON) == hfw

    beam_current = 1e-9
    microscope.set_beam_current(beam_current, BeamType.ION)
    assert microscope.get_beam_current(BeamType.ION) == beam_current