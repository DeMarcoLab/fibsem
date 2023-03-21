import pytest
from fibsem import structures
import os
import numpy as np

@pytest.fixture
def fake_fibsem_image():
    fake_image_settings = structures.ImageSettings(
        resolution=(1234,1234)
    )
    fake_metadata = structures.FibsemImageMetadata(
        image_settings=fake_image_settings,
        pixel_size=structures.Point(0.001,0.001),
        microscope_state=structures.MicroscopeState(),
        detector_settings=structures.FibsemDetectorSettings(
        type=None,
        mode=None,
        brightness=1.234,
        contrast=83.2
        ),

    )
    pic = structures.FibsemImage(
        data=np.random.randint(0,256,size=(10,10)).astype(np.uint8),
        metadata=fake_metadata
    )

    return pic

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


def test_fibsem_rectangle():

    new_rect = structures.FibsemRectangle(
        left=2.31,
        top=4.5,
        width=5,
        height=1.1
    )

    assert new_rect.left == 2.31
    assert new_rect.top == 4.5
    assert new_rect.width == 5
    assert new_rect.height == 1.1


    rect_dict = {"left":1,"top":2,"width":3,"height":4}

    points = ["left","top","width","height"]

    rectangle_from_dict = structures.FibsemRectangle.__from_dict__(rect_dict)

    for point in points:
        
        output = getattr(rectangle_from_dict,point)
        assert output == rect_dict[point], f"point: {point} -- output {output} does not match expected {rect_dict[point]} "

    to_dict = new_rect.__to_dict__()

    assert to_dict["left"] == 2.31
    assert to_dict["top"] == 4.5
    assert to_dict["width"] == 5
    assert to_dict["height"] == 1.1

    bad_dict = {"left":1,"top":"a","width":34.4,"height": [1]}

    with pytest.raises(Exception):

        bad_rect = structures.FibsemRectangle.__from_dict__(bad_dict)


def test_image_settings(fake_fibsem_image):

    

    attributes_dict = {
        "resolution":(1000,2000),
        "dwell_time":2.0e-6,
        "hfw":100e-6,
        "autocontrast":False,
        "beam_type":"electron",
        "save":False,
        "label":"Fake_image",
        "gamma_enabled":False,
        "save_path":os.getcwd(),
        "reduced_area":{"left":1,"top":2,"width":3,"height":4}
    }


    new_image_settings = structures.ImageSettings.__from_dict__(attributes_dict)

    for attribute in attributes_dict:

        output = getattr(new_image_settings,attribute)
        if attribute == "beam_type":
            answer = structures.BeamType.ELECTRON
        elif attribute == "reduced_area":
            answer = structures.FibsemRectangle.__from_dict__(attributes_dict[attribute])
        else:    
            answer = attributes_dict[attribute]
        assert output == answer, f"output: {output} does not match answer: {answer}"

    from_fb_image = fake_fibsem_image.metadata.image_settings

    assert from_fb_image.resolution == (1234,1234)

    
    image_settings_2 = structures.ImageSettings(
        resolution=(100,100),
        dwell_time=1.23e-6,
        hfw=200e-6,
        autocontrast=True,
        beam_type=structures.BeamType.ION,
        save=False,
        label="my_image",
        gamma_enabled=True,
        save_path=None,
        reduced_area=None
    )

    answers_dict ={
        "resolution":(100,100),
        "dwell_time":1.23e-6,
        "hfw":200e-6,
        "autocontrast":True,
        "beam_type":structures.BeamType.ION.name,
        "save":False,
        "label":"my_image",
        "gamma_enabled":True,
        "save_path":None,
        "reduced_area":None
    }

    output_dict = image_settings_2.__to_dict__()

    for item in output_dict:

        output = output_dict[item]
        answer = answers_dict[item]

        assert output == answer, f"output: {output} does not match answer {answer}"

    assert image_settings_2.autocontrast is True
    assert image_settings_2.label == "my_image"
    assert image_settings_2.save_path is None


def test_beam_settings():

    attributes = {
        "beam_type":"ION",
        "working_distance": 1.2343,
        "beam_current": 34.545,
        "voltage":2.33,
        "hfw":1.23e-5,
        "resolution":[100,200],
        "dwell_time":1.23e-6,
        "stigmation":None,
        "shift":None
    }

    answers = {
        "beam_type":structures.BeamType.ION,
        "working_distance": 1.2343,
        "beam_current": 34.545,
        "voltage":2.33,
        "hfw":1.23e-5,
        "resolution":[100,200],
        "dwell_time":1.23e-6,
        "stigmation":structures.Point(),
        "shift":structures.Point()
    }

    new_BeamSettings = structures.BeamSettings.__from_dict__(attributes)

    assert isinstance(new_BeamSettings,structures.BeamSettings)
    assert new_BeamSettings.beam_current == 34.545
    assert new_BeamSettings.working_distance == 1.2343


    from_dict(new_BeamSettings,attributes=attributes,answers=answers)

    beamsettings_ = structures.BeamSettings(
        beam_type=structures.BeamType.ION,
        working_distance=1.23,
        beam_current=1.44,
        voltage=5.4,
        hfw=1.3e-6,
        resolution=[100,100],
        dwell_time=1.3e-5,
        stigmation=None,
        shift=None
    )

    beamsettings_to_dict = beamsettings_.__to_dict__()

    answers_to_dict = {
        "beam_type":"ION",
        "working_distance": 1.23,
        "beam_current": 1.44,
        "voltage":5.4,
        "hfw":1.3e-6,
        "resolution":[100,100],
        "dwell_time":1.3e-5,
        "stigmation":None,
        "shift":None
    }
    
    to_dict(beamsettings_to_dict,answers_to_dict)





def from_dict(main_object,attributes: dict, answers: dict):

    

    for attribute in attributes:

        output = getattr(main_object,attribute)
        answer = answers[attribute]

        assert output == answer, f"Attribute: {attribute} output: {output} does not match answer: {answer}"

def to_dict(attributes: dict, answers: dict):

    for attribute in attributes:

        output = attributes[attribute] 
        answer = answers[attribute]

        assert output == answer , f"Attribute: {attribute} output: {output} does not match answer: {answer}"








        


