import pytest
from fibsem import structures

def test_point():

    p1 = structures.Point(2,3)

    p_dict = p1.__to_dict__()

    assert p_dict["x"] == 2
    assert p_dict["y"] == 3

    p_made_dict = {"x":5,"y":6}

    p2 = structures.Point.__from_dict__(p_made_dict)

    assert p2.x == 5
    assert p2.y == 6

    p_list = p2.__to_list__()

    assert p_list[0] == 5
    assert p_list[1] == 6

    p_made_list = [-2,-9]

    p3 = structures.Point.__from_list__(p_made_list)

    assert p3.x == -2
    assert p3.y == -9

    p1.x = 5

    assert p1.x == 5

    p4 = structures.Point.__add__(p1,p2)

    assert p4.x == 10
    assert p4.y == 9

    # with pytest.raises(Exception):

    #     bad_point = structures.Point("a","b")

def test_BeamType():

    bt = structures.BeamType

    assert bt.ELECTRON.value == 1
    assert bt.ION.value == 2

def test_movementMode():

    mm = structures.MovementMode

    assert mm.Stable.value == 1
    assert mm.Eucentric.value == 2

def test_fibsemStagePosition():

    fsp = structures.FibsemStagePosition(1,2,3,4,5)

    properties = ["x","y","z","r","t","coordinate_system"]
    values = [1,2,3,4,5,None]
    output_dict = fsp.__to_dict__()

    for property, value, dict_value in zip(properties,values,output_dict):

        output = getattr(fsp,property)
        assert output == value, f'output {output} does not match value {value}'
        assert output_dict[property] == value, f'output dict value {output_dict[property]} does not match value {value}'

    new_dict = {
        "x": 0.2,
        "y": 4,
        "z": 3,
        "r": 0.0012,
        "t": 0,
        "coordinate_system":None
    }

    fsp_from_dict = structures.FibsemStagePosition.__from_dict__(new_dict)

    new_dict_values = [0.2,4,3,0.0012,0,None]

    for property,answer in zip(properties,new_dict_values):

        output = getattr(fsp_from_dict,property)
        assert output == answer, f'Dict output {output} does not match answer {answer}'

    bad_dict = {
        "x": 0,
        "y": "a",
        "z": structures.Point(),
        "r": 0.123,
        "t": "hello"
    }

    with pytest.raises(Exception):

        bad_position = structures.FibsemStagePosition.__from_dict__(bad_dict)

def test_fibsemhardware():

    hardware_dict = {

        "electron":{"enabled":False},
        "ion":{"enabled":True},
        "stage":{"enabled":True,"rotation":True,"tilt":True},
        "manipulator":{"enabled":False,"rotation":False,"tilt":False},
        "gis":{"enabled":True,"multichem":True}
    }

    fbhardware = structures.FibsemHardware()

    from_dict_hardware = structures.FibsemHardware.__from_dict__(hardware_dict)

    attributes = [
        "electron_beam",
        "ion_beam",
        "stage_enabled",
        "stage_rotation",
        "stage_tilt",
        "manipulator_enabled",
        "manipulator_rotation",
        "manipulator_tilt",
        "gis_enabled",
        "gis_multichem",
    ]

    for attribute in attributes:

        assert getattr(fbhardware,attribute) == True, f"attribute: {attribute} does not match" 

        if "manipulator" in attribute or "electron" in attribute:

            assert getattr(from_dict_hardware,attribute) == False, f"attribute: {attribute} does not match" 
        else:
            assert getattr(from_dict_hardware,attribute) == True, f"attribute: {attribute} does not match" 


    bad_hardware_dict = {

        "electron":{"enabled":False},
        "ion":{"enabled":4},
        "stage":{"enabled":True,"rotation":"hello","tilt":True},
        "manipulator":{"enabled":False,"rotation":False,"tilt":False},
        "gis":{"enabled":True,"multichem":7}
    }

    with pytest.raises(Exception):

        bad_hardware = structures.FibsemHardware.__from_dict__(bad_hardware_dict)
    


def test_manipulator_position():

    fmp = structures.FibsemManipulatorPosition(1,2,3,4,5)

    properties = ["x","y","z","r","t","coordinate_system"]
    values = [1,2,3,4,5,None]
    output_dict = fmp.__to_dict__()

    for property, value, dict_value in zip(properties,values,output_dict):

        output = getattr(fmp,property)
        assert output == value, f'output {output} does not match value {value}'
        assert output_dict[property] == value, f'output dict value {output_dict[property]} does not match value {value}'

    new_dict = {
        "x": 0.2,
        "y": 4,
        "z": 3,
        "r": 0.0012,
        "t": 0,
        "coordinate_system":None
    }

    fmp_from_dict = structures.FibsemManipulatorPosition.__from_dict__(new_dict)

    new_dict_values = [0.2,4,3,0.0012,0,None]

    for property,answer in zip(properties,new_dict_values):

        output = getattr(fmp_from_dict,property)
        assert output == answer, f'Dict output {output} does not match answer {answer}'

    bad_dict = {
        "x": 0,
        "y": "a",
        "z": structures.Point(),
        "r": 0.123,
        "t": "hello"
    }

    with pytest.raises(Exception):

        bad_position = structures.FibsemManipulatorPosition.__from_dict__(bad_dict)
