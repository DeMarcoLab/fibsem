import pytest

from fibsem.structures import MicroscopeState, BeamType, ImageSettings, FibsemImage, FibsemRectangle, BeamSettings, FibsemDetectorSettings, FibsemStagePosition, FibsemGasInjectionSettings

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
    


















def test_gas_injection_settings():

    # gis
    gis_settings = FibsemGasInjectionSettings(
        gas="Pt",
        port=0,
        duration=30,
    )

    # to dict
    gdict = gis_settings.to_dict()  
    assert gdict["gas"] == gis_settings.gas
    assert gdict["port"] == gis_settings.port
    assert gdict["duration"] == gis_settings.duration
    assert gdict["insert_position"] == gis_settings.insert_position

    # from dict
    gis_settings2 = FibsemGasInjectionSettings.from_dict(gdict)
    assert gis_settings2.gas == gis_settings.gas
    assert gis_settings2.port == gis_settings.port
    assert gis_settings2.duration == gis_settings.duration
    assert gis_settings2.insert_position == gis_settings.insert_position

    multichem_settings = FibsemGasInjectionSettings(
        gas="Pt",
        port=0,
        duration=30,
        insert_position="ELECTRON_DEFAULT"
    )

    # to dict
    gdict = multichem_settings.to_dict()
    assert gdict["gas"] == multichem_settings.gas
    assert gdict["port"] == multichem_settings.port
    assert gdict["duration"] == multichem_settings.duration
    assert gdict["insert_position"] == multichem_settings.insert_position

    # from dict 
    multichem_settings2 = FibsemGasInjectionSettings.from_dict(gdict)
    assert multichem_settings2.gas == multichem_settings.gas
    assert multichem_settings2.port == multichem_settings.port
    assert multichem_settings2.duration == multichem_settings.duration
    assert multichem_settings2.insert_position == multichem_settings.insert_position