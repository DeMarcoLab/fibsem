import pytest


from fibsem.structures import MicroscopeState, BeamType, ImageSettings, FibsemImage, FibsemRectangle, BeamSettings, FibsemDetectorSettings, FibsemStagePosition, FibsemGasInjectionSettings

from fibsem.structures import MicroscopeState, FibsemStagePosition
from fibsem.structures import THERMO

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

if THERMO:
    from fibsem.structures import StagePosition, CompustagePosition
    from autoscript_sdb_microscope_client.enumerations import CoordinateSystem


    def test_to_autoscript_position():
        
        stage_position = FibsemStagePosition(x=1, y=2, z=3, r=4, t=5, coordinate_system="RAW")

        # test conversion to StagePosition
        autoscript_stage_position = stage_position.to_autoscript_position()

        assert autoscript_stage_position.x == stage_position.x
        assert autoscript_stage_position.y == stage_position.y
        assert autoscript_stage_position.z == stage_position.z
        assert autoscript_stage_position.r == stage_position.r
        assert autoscript_stage_position.t == stage_position.t
        assert autoscript_stage_position.coordinate_system == "RAW" # TODO: update to CoordinateSystem.Raw

        # test convesion to CompuStagePosition
        autoscript_compustage_position = stage_position.to_autoscript_position(compustage=True)

        assert autoscript_compustage_position.x == stage_position.x
        assert autoscript_compustage_position.y == stage_position.y
        assert autoscript_compustage_position.z == stage_position.z
        assert autoscript_compustage_position.a == stage_position.t
        assert autoscript_compustage_position.coordinate_system == "RAW" # TODO: update to CoordinateSystem.Raw  


    def test_from_autoscript_position():
        
        autoscript_stage_position = StagePosition(x=1, y=2, z=3, r=4, t=5, coordinate_system=CoordinateSystem.RAW)
        autoscript_compustage_position = CompustagePosition(x=1, y=2, z=3, a=5, coordinate_system=CoordinateSystem.RAW)

        # test conversion from StagePosition
        stage_position = FibsemStagePosition.from_autoscript_position(autoscript_stage_position)

        assert stage_position.x == autoscript_stage_position.x
        assert stage_position.y == autoscript_stage_position.y
        assert stage_position.z == autoscript_stage_position.z
        assert stage_position.r == autoscript_stage_position.r
        assert stage_position.t == autoscript_stage_position.t
        assert stage_position.coordinate_system == "RAW"

        # test conversion from CompuStagePosition
        stage_position = FibsemStagePosition.from_autoscript_position(autoscript_compustage_position)

        assert stage_position.x == autoscript_compustage_position.x
        assert stage_position.y == autoscript_compustage_position.y
        assert stage_position.z == autoscript_compustage_position.z
        assert stage_position.r == 0
        assert stage_position.t == autoscript_compustage_position.a
        assert stage_position.coordinate_system == "RAW"

