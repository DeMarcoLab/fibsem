import pytest

from fibsem import utils, microscope

@pytest.fixture
def microscope_connection():

    tescan_microscope,settings = utils.setup_session(ip_address='localhost',manufacturer='Tescan')

    return [tescan_microscope,settings]

def test_connection(microscope_connection):

    tescan_microscope = microscope_connection[0]
    settings = microscope_connection[1]

    assert tescan_microscope is not None
    assert isinstance(tescan_microscope,microscope.TescanMicroscope)

    assert settings is not None
    assert isinstance(settings,microscope.MicroscopeSettings)
    assert settings.system.manufacturer == "Tescan"

