# import pytest
# from fibsem import structures
# import os
# import numpy as np
# from pathlib import Path


# def from_dict(main_object,attributes: dict, answers: dict):

    

#     for attribute in attributes:

#         output = getattr(main_object,attribute)
#         answer = answers[attribute]

#         assert output == answer, f"Attribute: {attribute} output: {output} does not match answer: {answer}"

# def to_dict(attributes: dict, answers: dict):

#     for attribute in attributes:

#         output = attributes[attribute] 
#         answer = answers[attribute]

#         assert output == answer , f"Attribute: {attribute} output: {output} does not match answer: {answer}"

# @pytest.fixture
# def fake_image_settings():
#     fake_image_settings = structures.ImageSettings(
#         resolution=[1234,1234],
#         dwell_time= 150e-6,
#         beam_type=structures.BeamType.ELECTRON,
#         hfw = 150e-6,
#         autocontrast=True,
#         save = True,
#         label = "test",
#         gamma_enabled = True,
#         save_path = os.path.join(os.getcwd(),"test_images"),
#         reduced_area = structures.FibsemRectangle(0,0,10,10),
#     )
#     return fake_image_settings

# @pytest.fixture
# def fake_eb_settings():
#     eb_settings= structures.BeamSettings(
#             beam_type=structures.BeamType.ELECTRON, 
#             working_distance=0,
#             beam_current=0,
#             voltage=0,
#             hfw=0,
#             resolution=[1536,1024],
#             dwell_time=0,
#             stigmation=structures.Point(0,0),
#             shift=structures.Point(0,0),
#         )
#     return eb_settings

# @pytest.fixture
# def fake_ib_settings():
#     ib_settings= structures.BeamSettings(
#             beam_type=structures.BeamType.ION, 
#             working_distance=0,
#             beam_current=0,
#             voltage=0,
#             hfw=0,
#             resolution=[1536,1024],
#             dwell_time=0,
#             stigmation=structures.Point(0,0),
#             shift=structures.Point(0,0),
#         )
#     return ib_settings

# @pytest.fixture
# def fake_detector_settings():
#     detector_settings = structures.FibsemDetectorSettings(
#         type = "BSE",
#         mode= "NORMAL",
#         brightness= 1,
#         contrast= 1,
#     )
#     return detector_settings

# @pytest.fixture
# def fake_fibsem_image(fake_image_settings, fake_eb_settings, fake_ib_settings, fake_detector_settings):
#     state = structures.MicroscopeState(
#         timestamp=0,
#         absolute_position=structures.FibsemStagePosition(0,0,0,0,0),
#         eb_settings= fake_eb_settings,
#         ib_settings= fake_ib_settings,
#         )

#     fake_metadata = structures.FibsemImageMetadata(
#         image_settings=fake_image_settings,
#         pixel_size=structures.Point(0.001,0.001),
#         microscope_state=state,
#         detector_settings=fake_detector_settings,
#         version="v1", 
#         )

#     pic = structures.FibsemImage(
#         data=np.random.randint(0,256,size=(10,10)).astype(np.uint8),
#         metadata=fake_metadata
#     )

#     return pic

# def test_point():

#     p1 = structures.Point(2,3)

#     p_dict = p1.__to_dict__()

#     assert p_dict["x"] == 2
#     assert p_dict["y"] == 3

#     p_made_dict = {"x":5,"y":6}

#     p2 = structures.Point.__from_dict__(p_made_dict)

#     assert p2.x == 5
#     assert p2.y == 6

#     p_list = p2.__to_list__()

#     assert p_list[0] == 5
#     assert p_list[1] == 6

#     p_made_list = [-2,-9]

#     p3 = structures.Point.__from_list__(p_made_list)

#     assert p3.x == -2
#     assert p3.y == -9

#     p1.x = 5

#     assert p1.x == 5

#     p4 = structures.Point.__add__(p1,p2)

#     assert p4.x == 10
#     assert p4.y == 9

#     # with pytest.raises(Exception):

#     #     bad_point = structures.Point("a","b")

# def test_BeamType():

#     bt = structures.BeamType

#     assert bt.ELECTRON.value == 1
#     assert bt.ION.value == 2

# def test_movementMode():

#     mm = structures.MovementMode

#     assert mm.Stable.value == 1
#     assert mm.Eucentric.value == 2

# def test_fibsemStagePosition():

#     fsp = structures.FibsemStagePosition(1,2,3,4,5)

#     properties = ["x","y","z","r","t","coordinate_system"]
#     values = [1,2,3,4,5,None]
#     output_dict = fsp.__to_dict__()

#     for property, value, dict_value in zip(properties,values,output_dict):

#         output = getattr(fsp,property)
#         assert output == value, f'output {output} does not match value {value}'
#         assert output_dict[property] == value, f'output dict value {output_dict[property]} does not match value {value}'

#     new_dict = {
#         "x": 0.2,
#         "y": 4,
#         "z": 3,
#         "r": 0.0012,
#         "t": 0,
#         "coordinate_system":None
#     }

#     fsp_from_dict = structures.FibsemStagePosition.__from_dict__(new_dict)

#     new_dict_values = [0.2,4,3,0.0012,0,None]

#     for property,answer in zip(properties,new_dict_values):

#         output = getattr(fsp_from_dict,property)
#         assert output == answer, f'Dict output {output} does not match answer {answer}'

#     bad_dict = {
#         "x": 0,
#         "y": "a",
#         "z": structures.Point(),
#         "r": 0.123,
#         "t": "hello"
#     }

#     with pytest.raises(Exception):

#         bad_position = structures.FibsemStagePosition.__from_dict__(bad_dict)
#         bad_position_2 = structures.FibsemStagePosition(1,2,3,"hello",4,3)

# def test_fibsemhardware():

#     hardware_dict = {

#         "electron":{"enabled":False},
#         "ion":{"enabled":True},
#         "stage":{"enabled":True,"rotation":True,"tilt":True},
#         "manipulator":{"enabled":False,"rotation":False,"tilt":False},
#         "gis":{"enabled":True,"multichem":True}
#     }

#     fbhardware = structures.FibsemHardware()

#     from_dict_hardware = structures.FibsemHardware.__from_dict__(hardware_dict)

#     attributes = [
#         "electron_beam",
#         "ion_beam",
#         "stage_enabled",
#         "stage_rotation",
#         "stage_tilt",
#         "manipulator_enabled",
#         "manipulator_rotation",
#         "manipulator_tilt",
#         "gis_enabled",
#         "gis_multichem",
#     ]

#     for attribute in attributes:

#         assert getattr(fbhardware,attribute) == True, f"attribute: {attribute} does not match" 

#         if "manipulator" in attribute or "electron" in attribute:

#             assert getattr(from_dict_hardware,attribute) == False, f"attribute: {attribute} does not match" 
#         else:
#             assert getattr(from_dict_hardware,attribute) == True, f"attribute: {attribute} does not match" 


    
#     bad_hardware_dict = {

#         "electron":{"enabled":False},
#         "ion":{"enabled":4},
#         "stage":{"enabled":True,"rotation":"hello","tilt":True},
#         "manipulator":{"enabled":False,"rotation":False,"tilt":False},
#         "gis":{"enabled":True,"multichem":7}
#     }

#     with pytest.raises(Exception):

#         bad_hardware = structures.FibsemHardware.__from_dict__(bad_hardware_dict)
#         bad_hardware_2 = structures.FibsemHardware(electron_beam=3,ion_beam="hello",stage_enabled=[1,2,3])


# def test_manipulator_position():

#     fmp = structures.FibsemManipulatorPosition(1,2,3,4,5)

#     properties = ["x","y","z","r","t","coordinate_system"]
#     values = [1,2,3,4,5,None]
#     output_dict = fmp.__to_dict__()

#     for property, value, dict_value in zip(properties,values,output_dict):

#         output = getattr(fmp,property)
#         assert output == value, f'output {output} does not match value {value}'
#         assert output_dict[property] == value, f'output dict value {output_dict[property]} does not match value {value}'

#     new_dict = {
#         "x": 0.2,
#         "y": 4,
#         "z": 3,
#         "r": 0.0012,
#         "t": 0,
#         "coordinate_system": "RAW"
#     }

#     fmp_from_dict = structures.FibsemManipulatorPosition.__from_dict__(new_dict)

#     new_dict_values = [0.2,4,3,0.0012,0,None]

#     for property,answer in zip(properties,new_dict_values):

#         output = getattr(fmp_from_dict,property)
#         assert output == answer, f'Dict output {output} does not match answer {answer}'

#     bad_dict = {
#         "x": 0,
#         "y": "a",
#         "z": structures.Point(),
#         "r": 0.123,
#         "t": "hello"
#     }

#     with pytest.raises(Exception):

#         bad_position = structures.FibsemManipulatorPosition.__from_dict__(bad_dict)
#         bad_position_2 = structures.FibsemManipulatorPosition(1,2,3,"HELLO",(1,2),"GOODBYE")


# def test_fibsem_rectangle():

#     new_rect = structures.FibsemRectangle(
#         left=2.31,
#         top=4.5,
#         width=5,
#         height=1.1
#     )

#     assert new_rect.left == 2.31
#     assert new_rect.top == 4.5
#     assert new_rect.width == 5
#     assert new_rect.height == 1.1


#     rect_dict = {"left":1,"top":2,"width":3,"height":4}

#     points = ["left","top","width","height"]

#     rectangle_from_dict = structures.FibsemRectangle.__from_dict__(rect_dict)

#     for point in points:
        
#         output = getattr(rectangle_from_dict,point)
#         assert output == rect_dict[point], f"point: {point} -- output {output} does not match expected {rect_dict[point]} "

#     to_dict = new_rect.__to_dict__()

#     assert to_dict["left"] == 2.31
#     assert to_dict["top"] == 4.5
#     assert to_dict["width"] == 5
#     assert to_dict["height"] == 1.1

#     bad_dict = {"left":1,"top":"a","width":34.4,"height": [1]}

#     with pytest.raises(Exception):

#         bad_rect = structures.FibsemRectangle.__from_dict__(bad_dict)
#         bad_rect_2 = structures.FibsemRectangle(1,2,"hello",[1,2,3])
#         bad_rect_3 = structures.FibsemRectangle([1,2,3,4])


# def test_image_settings(fake_fibsem_image):

    

#     attributes_dict = {
#         "resolution":[1000,2000],
#         "dwell_time":2.0e-6,
#         "hfw":100e-6,
#         "autocontrast":False,
#         "beam_type":"electron",
#         "save":False,
#         "label":"Fake_image",
#         "gamma_enabled":False,
#         "save_path":os.getcwd(),
#         "reduced_area":{"left":1,"top":2,"width":3,"height":4}
#     }


#     new_image_settings = structures.ImageSettings.__from_dict__(attributes_dict)


#     for attribute in attributes_dict:

#         output = getattr(new_image_settings,attribute)
#         if attribute == "beam_type":
#             answer = structures.BeamType.ELECTRON
#         elif attribute == "reduced_area":
#             answer = structures.FibsemRectangle.__from_dict__(attributes_dict[attribute])
#         else:    
#             answer = attributes_dict[attribute]
#         assert output == answer, f"output: {output} does not match answer: {answer}"

#     from_fb_image = fake_fibsem_image.metadata.image_settings

#     assert from_fb_image.resolution == [1234,1234]

    
#     image_settings_2 = structures.ImageSettings(
#         resolution=[100,100],
#         dwell_time=1.23e-6,
#         hfw=200e-6,
#         autocontrast=True,
#         beam_type=structures.BeamType.ION,
#         save=False,
#         label="my_image",
#         gamma_enabled=True,
#         save_path=None,
#         reduced_area=None
#     )

#     answers_dict ={
#         "resolution":[100,100],
#         "dwell_time":1.23e-6,
#         "hfw":200e-6,
#         "autocontrast":True,
#         "beam_type":structures.BeamType.ION.name,
#         "save":False,
#         "label":"my_image",
#         "gamma_enabled":True,
#         "save_path":None,
#         "reduced_area":None
#     }

#     output_dict = image_settings_2.__to_dict__()

#     for item in output_dict:

#         output = output_dict[item]
#         answer = answers_dict[item]

#         assert output == answer, f"output: {output} does not match answer {answer}"

#     assert image_settings_2.autocontrast is True
#     assert image_settings_2.label == "my_image"
#     assert image_settings_2.save_path is None

#     with pytest.raises(Exception):
#         bad_image_settings = structures.ImageSettings(
#             resolution=[200,200],
#             dwell_time=2.3,
#             hfw=50,
#             autocontrast="hello",
#             beam_type="ion",
#             reduced_area=[1,2,3,4],
#             save=False)


# def test_beam_settings():

#     attributes = {
#         "beam_type":"ION",
#         "working_distance": 1.2343,
#         "beam_current": 34.545,
#         "voltage":2.33,
#         "hfw":1.23e-5,
#         "resolution":[100,200],
#         "dwell_time":1.23e-6,
#         "stigmation":None,
#         "shift":None
#     }

#     answers = {
#         "beam_type":structures.BeamType.ION,
#         "working_distance": 1.2343,
#         "beam_current": 34.545,
#         "voltage":2.33,
#         "hfw":1.23e-5,
#         "resolution":[100,200],
#         "dwell_time":1.23e-6,
#         "stigmation":structures.Point(),
#         "shift":structures.Point()
#     }

#     new_BeamSettings = structures.BeamSettings.__from_dict__(attributes)

#     assert isinstance(new_BeamSettings,structures.BeamSettings)
#     assert new_BeamSettings.beam_current == 34.545
#     assert new_BeamSettings.working_distance == 1.2343


#     to_dict = new_rect.__to_dict__()

#     assert to_dict["left"] == 2.31
#     assert to_dict["top"] == 4.5
#     assert to_dict["width"] == 5
#     assert to_dict["height"] == 1.1

#     bad_dict = {"left":1,"top":"a","width":34.4,"height": [1]}

#     with pytest.raises(Exception):

#         bad_rect = structures.FibsemRectangle.__from_dict__(bad_dict)
#         bad_rect_2 = structures.FibsemRectangle(1,2,"hello",[1,2,3])
#         bad_rect_3 = structures.FibsemRectangle([1,2,3,4])


# def test_image_settings(fake_fibsem_image):

#     from_dict(new_BeamSettings,attributes=attributes,answers=answers)

#     beamsettings_ = structures.BeamSettings(
#         beam_type=structures.BeamType.ION,
#         working_distance=1.23,
#         beam_current=1.44,
#         voltage=5.4,
#         hfw=1.3e-6,
#         resolution=[100,100],
#         dwell_time=1.3e-5,
#         stigmation=None,
#         shift=None
#     )

#     beamsettings_to_dict = beamsettings_.__to_dict__()

#     answers_to_dict = {
#         "beam_type":"ION",
#         "working_distance": 1.23,
#         "beam_current": 1.44,
#         "voltage":5.4,
#         "hfw":1.3e-6,
#         "resolution":[100,100],
#         "dwell_time":1.3e-5,
#         "stigmation":None,
#         "shift":None
#     }
    
#     to_dict(beamsettings_to_dict,answers_to_dict)

#     good_beam_settings = structures.BeamSettings(
#         beam_type=structures.BeamType.ELECTRON,
#         hfw=12,
#         resolution=[100,100],
#         dwell_time=1.11,
#         stigmation=structures.Point(1,2)
#     )
    


#     assert good_beam_settings.hfw == 12
#     assert good_beam_settings.beam_type == structures.BeamType.ELECTRON
#     assert good_beam_settings.dwell_time == 1.11
#     assert good_beam_settings.stigmation.x == 1

#     with pytest.raises(Exception):
#         bad_beam_settings = structures.BeamSettings(
#             beam_type=2,
#             working_distance=23,
#             voltage=[1,2],
#             stigmation=structures.Point(1,2),
#         )


# def test_MicroscopeState():
#     from datetime import datetime


    
#     stage_position_dict = {
#         "x": 1,
#         "y": 2,
#         "z": 3,
#         "r": 4,
#         "t": 5,
#         "coordinate_system": "RAW"
#     }

#     new_microscopeState = structures.MicroscopeState.__from_dict__(attributes)
#     time_now = datetime.timestamp(datetime.now())
#     assert new_microscopeState.ib_settings.beam_type.name == "ION"
#     assert new_microscopeState.absolute_position.x == 0
#     assert isinstance(new_microscopeState.timestamp,float)
#     assert abs(time_now-new_microscopeState.timestamp) < 10
#     assert isinstance(new_microscopeState.eb_settings,structures.BeamSettings)

#     new_microscopeState.absolute_position.y += 7    

#     assert new_microscopeState.absolute_position.y == 7

#     dict_microscopeState = new_microscopeState.__to_dict__()

#     assert dict_microscopeState["absolute_position"]["x"]==0
#     assert dict_microscopeState["absolute_position"]["y"]==7
#     assert dict_microscopeState["eb_settings"]["beam_type"]=="ELECTRON"

#     good_microscopeState = structures.MicroscopeState(absolute_position=structures.FibsemStagePosition(1,2,3,4,5))

#     assert good_microscopeState.absolute_position.x == 1
#     assert good_microscopeState.absolute_position.y == 2

#     with pytest.raises(Exception):

#         bad_microscopeState = structures.MicroscopeState(timestamp="a",absolute_position=[1,2,3],ib_settings="b")



# def test_FibsemPattern():

#     fbPattern = structures.FibsemPattern

#     assert fbPattern.Rectangle.value == 1
#     assert fbPattern.Line.value == 2
#     assert fbPattern.Circle.value == 3


# def test_fibsemMillingSettings():

#     attributes = {
#         "milling_current":20.0e-12,
#         "spot_size":5.0e-8,
#         "rate":3.0e-3,
#         "dwell_time":1.0e-6,
#         "hfw":150e-6,
#         "patterning_mode":"Serial",
#         "application_file":"Si"
#     }

#     new_fbMillingSettings = structures.FibsemMillingSettings()

#     from_dict(new_fbMillingSettings,attributes,attributes)

#     fbMillingSettings_dict = new_fbMillingSettings.__to_dict__()

#     to_dict(fbMillingSettings_dict,attributes)

#     new_fbMillingSettings.spot_size = 4.0e-8
#     new_fbMillingSettings.hfw = 54

#     assert new_fbMillingSettings.spot_size == 4.0e-8
#     assert new_fbMillingSettings.hfw == 54

#     with pytest.raises(Exception):
#         bad_fbMillingSettings = structures.FibsemMillingSettings(milling_current=3,spot_size=[1],dwell_time=True)


# def test_stage_position_to_dict():

#     fake_stage_position = structures.FibsemStagePosition(1,2,3,4,5,"RAW")

#     fake_stage_position_dict = structures.stage_position_to_dict(fake_stage_position)

#     answers = {
#         "x": 1,
#         "y":2,
#         "z":3,
#         "r":4,
#         "t":5,
#         "coordinate_system":"RAW"
#     }

#     to_dict(fake_stage_position_dict,answers)


# def test_images(fake_fibsem_image, fake_image_settings, fake_eb_settings, fake_ib_settings):
    
#     assert fake_fibsem_image.metadata.compare_image_settings(fake_image_settings)
#     assert fake_fibsem_image.metadata.version == "v1"
#     assert fake_fibsem_image.metadata.microscope_state.eb_settings == fake_eb_settings
#     assert fake_fibsem_image.metadata.microscope_state.ib_settings == fake_ib_settings
#     assert fake_fibsem_image.metadata.microscope_state.absolute_position == structures.FibsemStagePosition(0,0,0,0,0)

#     fake_fibsem_image.save()
#     path = os.path.join(fake_fibsem_image.metadata.image_settings.save_path, fake_fibsem_image.metadata.image_settings.label)
#     save_path = Path(path).with_suffix(".tif")
#     assert os.path.exists(save_path)

#     loaded = structures.FibsemImage.load(save_path)

#     assert loaded.metadata.compare_image_settings(fake_image_settings)
#     assert loaded.metadata.version == "v1"
#     assert loaded.metadata.microscope_state.eb_settings == fake_eb_settings
#     assert loaded.metadata.microscope_state.ib_settings == fake_ib_settings
#     assert loaded.metadata.microscope_state.absolute_position == structures.FibsemStagePosition(0,0,0,0,0)


# def test_stage_position_from_dict():

#     fake_stage_position_dict = {
#         "x": 1,
#         "y":2,
#         "z":3,
#         "r":4,
#         "t":5,
#         "coordinate_system": None
#     }

#     fake_stage_position = structures.stage_position_from_dict(fake_stage_position_dict)

#     answers = {
#         "x": 1,
#         "y":2,
#         "z":3,
#         "r":4,
#         "t":5,
#         "coordinate_system":None
#     }

#     from_dict(fake_stage_position,attributes=answers,answers=answers)


# def test_detector_settings(fake_detector_settings):
#     dict = {
#         "type": "BSE",
#         "mode": "NORMAL",
#         "contrast": 1,
#         "brightness": 1,
#     }

#     assert fake_detector_settings.type == "BSE"
#     assert fake_detector_settings.mode == "NORMAL"
#     assert fake_detector_settings.contrast == 1
#     assert fake_detector_settings.brightness == 1
#     # assert fake_detector_settings.to_tescan() == (100, 100)

#     assert fake_detector_settings.__to_dict__() == dict
#     assert structures.FibsemDetectorSettings.__from_dict__(dict) == fake_detector_settings

#     with pytest.raises(Exception):

#         bad_fb_detector_settings = structures.FibsemDetectorSettings(type=True,brightness="1.234",contrast=[1,23])


#     fake_stage_position = structures.FibsemStagePosition(1,2,3,4,5,"RAW")

#     fake_stage_position_dict = structures.stage_position_to_dict(fake_stage_position)

# @pytest.fixture
# def fibsem_stage_position() -> structures.FibsemStagePosition:

#     stage_position = structures.FibsemStagePosition(
#         x=1.0,
#         y=2.0,
#         z=3.0,
#         r=4.0,
#         t=5.0,
#         coordinate_system="RAW"
#     )


#     to_dict(fake_stage_position_dict,answers)

#     assert to_dict["stage"] == "Base"
#     assert to_dict["microscope_state"]["timestamp"] == 0.0
#     assert to_dict["microscope_state"]["absolute_position"]["x"] == 0.0
#     assert to_dict["microscope_state"]["absolute_position"]["y"] == 0.0
#     assert to_dict["microscope_state"]["absolute_position"]["z"] == 0.0
#     assert to_dict["microscope_state"]["absolute_position"]["r"] == 0.0
#     assert to_dict["microscope_state"]["absolute_position"]["t"] == 0.0
#     assert to_dict["microscope_state"]["absolute_position"]["coordinate_system"] == None
#     assert to_dict["microscope_state"]["eb_settings"]["beam_type"] == "ELECTRON"
#     assert to_dict["microscope_state"]["eb_settings"]["working_distance"] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["beam_current"] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["voltage"] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["hfw"] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["resolution"] == [1536,1024]
#     assert to_dict["microscope_state"]["eb_settings"]["dwell_time"] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["stigmation"]["x"] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["stigmation"]["y"] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["shift"]['x'] == 0
#     assert to_dict["microscope_state"]["eb_settings"]["shift"]['y'] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["beam_type"] == "ION"
#     assert to_dict["microscope_state"]["ib_settings"]["working_distance"] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["beam_current"] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["voltage"] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["hfw"] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["resolution"] == [1536,1024]
#     assert to_dict["microscope_state"]["ib_settings"]["dwell_time"] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["stigmation"]["x"] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["stigmation"]["y"] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["shift"]['x'] == 0
#     assert to_dict["microscope_state"]["ib_settings"]["shift"]['y'] == 0

#     from_dict = structures.FibsemState.__from_dict__(to_dict)
#     assert state == from_dict



# def test_microscope_settings(fake_image_settings):
#     settings = structures.MicroscopeSettings(
#         system = structures.SystemSettings(
#             ip_address="localhost",
#             stage=structures.StageSettings(
#                 rotation_flat_to_electron=0,
#                 rotation_flat_to_ion=0,
#                 tilt_flat_to_electron=0,
#                 tilt_flat_to_ion=0,
#                 pre_tilt=0,
#                 needle_stage_height_limit=0,
#             ),
#             ion = structures.BeamSystemSettings(
#                 beam_type=structures.BeamType.ION,
#                 voltage=0,
#                 current=0,
#                 detector_type= 'BSE',
#                 detector_mode= 'NORMAL',
#                 eucentric_height=0,
#                 plasma_gas = "Ar",
#             ),
#             electron = structures.BeamSystemSettings(
#                 beam_type=structures.BeamType.ELECTRON,
#                 voltage=0,
#                 current=0,
#                 detector_type= 'BSE',
#                 detector_mode= 'NORMAL',
#                 eucentric_height=0,
#                 plasma_gas = "Ar",
#             ),
#             manufacturer="Tescan",
#         ), 
#         image =  fake_image_settings,
#         protocol = None,
#         milling = structures.FibsemMillingSettings(
#             milling_current=0,
#             spot_size=0,
#             rate=0,
#             dwell_time=0,
#             hfw=0,
#             patterning_mode="Serial",
#             application_file="Si",
#         ),
#         hardware=structures.FibsemHardware(
#             electron_beam=True,
#             ion_beam=True,
#             stage_enabled=True,
#             stage_rotation=True,
#             stage_tilt=True,
#             manipulator_enabled=True,
#             manipulator_rotation=True,
#             manipulator_tilt=True,
#             gis_enabled=True,
#             gis_multichem=True,
#         )
#     )

#     to_dict = settings.__to_dict__()
#     from_dict = structures.MicroscopeSettings.__from_dict__(to_dict)
#     assert settings == from_dict

# def test_system_settings():
#     system = structures.SystemSettings(
#             ip_address="localhost",
#             stage=structures.StageSettings(
#                 rotation_flat_to_electron=0,
#                 rotation_flat_to_ion=0,
#                 tilt_flat_to_electron=0,
#                 tilt_flat_to_ion=0,
#                 pre_tilt=0,
#                 needle_stage_height_limit=0,
#             ),
#             ion = structures.BeamSystemSettings(
#                 beam_type=structures.BeamType.ION,
#                 voltage=0,
#                 current=0,
#                 detector_type= 'BSE',
#                 detector_mode= 'NORMAL',
#                 eucentric_height=0,
#                 plasma_gas = "Ar",
#             ),
#             electron = structures.BeamSystemSettings(
#                 beam_type=structures.BeamType.ELECTRON,
#                 voltage=0,
#                 current=0,
#                 detector_type= 'BSE',
#                 detector_mode= 'NORMAL',
#                 eucentric_height=0,
#                 plasma_gas = "Ar",
#             ),
#             manufacturer="Tescan",
#         )
#     to_dict = system.__to_dict__()
#     assert system.ip_address == to_dict["ip_address"]
#     assert system.stage.rotation_flat_to_electron == to_dict["stage"]["rotation_flat_to_electron"]
#     assert system.stage.rotation_flat_to_ion == to_dict["stage"]["rotation_flat_to_ion"]
#     assert system.stage.tilt_flat_to_electron == to_dict["stage"]["tilt_flat_to_electron"]
#     assert system.stage.tilt_flat_to_ion == to_dict["stage"]["tilt_flat_to_ion"]
#     assert system.stage.pre_tilt == to_dict["stage"]["pre_tilt"]
#     assert system.stage.needle_stage_height_limit == to_dict["stage"]["needle_stage_height_limit"]
#     assert system.ion.voltage == to_dict["ion"]["voltage"]
#     assert system.ion.current == to_dict["ion"]["current"]
#     assert system.ion.detector_type == to_dict["ion"]["detector_type"]
#     assert system.ion.detector_mode == to_dict["ion"]["detector_mode"]
#     assert system.ion.eucentric_height == to_dict["ion"]["eucentric_height"]
#     assert system.ion.plasma_gas == to_dict["ion"]["plasma_gas"]
#     assert system.electron.voltage == to_dict["electron"]["voltage"]
#     assert system.electron.current == to_dict["electron"]["current"]
#     assert system.electron.detector_type == to_dict["electron"]["detector_type"]
#     assert system.electron.detector_mode == to_dict["electron"]["detector_mode"]
#     assert system.electron.eucentric_height == to_dict["electron"]["eucentric_height"]
#     assert system.electron.plasma_gas == to_dict["electron"]["plasma_gas"]
#     assert system.manufacturer == to_dict["manufacturer"]

#     from_dict = structures.SystemSettings.__from_dict__(to_dict)
#     assert system == from_dict

# def test_stage_settings():
#     settings = structures.StageSettings(
#         rotation_flat_to_electron=0,
#         rotation_flat_to_ion=0,
#         tilt_flat_to_electron=0,
#         tilt_flat_to_ion=0,
#         pre_tilt=0,
#         needle_stage_height_limit=0,
#     )
#     to_dict = settings.__to_dict__()
#     assert settings.rotation_flat_to_electron == to_dict["rotation_flat_to_electron"]
#     assert settings.rotation_flat_to_ion == to_dict["rotation_flat_to_ion"]
#     assert settings.tilt_flat_to_electron == to_dict["tilt_flat_to_electron"]
#     assert settings.tilt_flat_to_ion == to_dict["tilt_flat_to_ion"]
#     assert settings.pre_tilt == to_dict["pre_tilt"]
#     assert settings.needle_stage_height_limit == to_dict["needle_stage_height_limit"]

#     from_dict = structures.StageSettings.__from_dict__(to_dict)
#     assert settings == from_dict


# def test_image_metadata(fake_image_settings, fake_eb_settings, fake_ib_settings, fake_detector_settings):
#     metadata = structures.FibsemImageMetadata(
#         image_settings = fake_image_settings,
#         pixel_size = structures.Point(1, 1),
#         microscope_state = structures.MicroscopeState(
#             absolute_position= structures.FibsemStagePosition(0,0,0,0,0),
#             eb_settings= fake_eb_settings,
#             ib_settings= fake_ib_settings,
#         ),
#         detector_settings = fake_detector_settings, 
#         version = "v1", 
#         )

#     to_dict = metadata.__to_dict__()

#     assert metadata.compare_image_settings(fake_image_settings)
#     assert metadata.pixel_size == structures.Point(1, 1)
#     assert metadata.microscope_state.absolute_position == structures.FibsemStagePosition(0,0,0,0,0)
#     assert metadata.microscope_state.eb_settings  == fake_eb_settings
#     assert metadata.microscope_state.ib_settings  == fake_ib_settings
#     assert metadata.detector_settings == fake_detector_settings
#     assert metadata.version == "v1"

#     from_dict = structures.FibsemImageMetadata.__from_dict__(to_dict)
#     assert metadata == from_dict


# def test_reference_images():

#     pic_size = np.random.randint(200,1000,size=2)

#     low_eb = np.random.randint(0,255,size=pic_size)
#     high_eb = np.random.randint(0,255,size=pic_size)
#     low_ib = np.random.randint(0,255,size=pic_size)
#     high_ib = np.random.randint(0,255,size=pic_size)

#     ref_images = structures.ReferenceImages(low_res_eb=low_eb, high_res_eb=high_eb, low_res_ib=low_ib, high_res_ib=high_ib)

#     assert (ref_images.low_res_eb == low_eb).all()
#     assert (ref_images.high_res_eb == high_eb).all()
#     assert (ref_images.low_res_ib == low_ib).all()
#     assert (ref_images.high_res_ib == high_ib).all()


# def test_pattern_settings():
#     circle = structures.FibsemPatternSettings(
#         pattern = structures.FibsemPattern.Circle,
#         centre_x = 0,
#         centre_y = 0,
#         radius = 0,
#         rotation = 0,
#         depth = 0,
#         start_angle = 0,
#         end_angle = 0,
#         scan_direction = "TopToBottom",
#         cleaning_cross_section = False,
#         )
    
#     to_dict = circle.__to_dict__()
#     assert circle.pattern.name == to_dict["pattern"]
#     assert circle.centre_x == to_dict["centre_x"]
#     assert circle.centre_y == to_dict["centre_y"]
#     assert circle.radius == to_dict["radius"]
#     assert circle.rotation == to_dict["rotation"]
#     assert circle.depth == to_dict["depth"]
#     assert circle.start_angle == to_dict["start_angle"]
#     assert circle.end_angle == to_dict["end_angle"]
#     assert circle.scan_direction == to_dict["scan_direction"]
#     assert circle.cleaning_cross_section == to_dict["cleaning_cross_section"]
#     from_dict = structures.FibsemPatternSettings.__from_dict__(to_dict)
#     assert circle.pattern == from_dict.pattern
#     assert circle.centre_x == from_dict.centre_x
#     assert circle.centre_y == from_dict.centre_y
#     assert circle.radius == from_dict.radius
#     assert circle.rotation == from_dict.rotation
#     assert circle.depth == from_dict.depth
#     assert circle.start_angle == from_dict.start_angle
#     assert circle.end_angle == from_dict.end_angle
#     assert circle.scan_direction == from_dict.scan_direction
#     assert circle.cleaning_cross_section == from_dict.cleaning_cross_section


#     line = structures.FibsemPatternSettings(
#         pattern = structures.FibsemPattern.Line,
#         start_x = 0,
#         start_y = 0,
#         end_x =0,
#         end_y =0, 
#         depth = 0,
#         rotation = 0,
#         scan_direction=  "TopToBottom",
#         cleaning_cross_section= False,
#     ) 

#     to_dict = line.__to_dict__()
#     assert line.pattern.name == to_dict["pattern"]
#     assert line.start_x == to_dict["start_x"]
#     assert line.start_y == to_dict["start_y"]
#     assert line.end_x == to_dict["end_x"]
#     assert line.end_y == to_dict["end_y"]
#     assert line.depth == to_dict["depth"]
#     assert line.rotation == to_dict["rotation"]
#     assert line.scan_direction == to_dict["scan_direction"]
#     assert line.cleaning_cross_section == to_dict["cleaning_cross_section"]
#     from_dict = structures.FibsemPatternSettings.__from_dict__(to_dict)
#     assert line.pattern == from_dict.pattern
#     assert line.start_x == from_dict.start_x
#     assert line.start_y == from_dict.start_y
#     assert line.end_x == from_dict.end_x
#     assert line.end_y == from_dict.end_y
#     assert line.depth == from_dict.depth
#     assert line.rotation == from_dict.rotation
#     assert line.scan_direction == from_dict.scan_direction
#     assert line.cleaning_cross_section == from_dict.cleaning_cross_section


#     rectangle = structures.FibsemPatternSettings(
#         pattern = structures.FibsemPattern.Rectangle,
#         centre_x = 0,
#         centre_y = 0,
#         width = 0,
#         height = 0,
#         rotation = 0,
#         depth = 0,
#         scan_direction = "TopToBottom",
#         cleaning_cross_section = False,
#     )


#     to_dict = rectangle.__to_dict__()
#     assert rectangle.pattern.name == to_dict["pattern"]
#     assert rectangle.centre_x == to_dict["centre_x"]
#     assert rectangle.centre_y == to_dict["centre_y"]
#     assert rectangle.width == to_dict["width"]
#     assert rectangle.height == to_dict["height"]
#     assert rectangle.rotation == to_dict["rotation"]
#     assert rectangle.depth == to_dict["depth"]
#     assert rectangle.scan_direction == to_dict["scan_direction"]
#     assert rectangle.cleaning_cross_section == to_dict["cleaning_cross_section"]
#     from_dict = structures.FibsemPatternSettings.__from_dict__(to_dict)
#     assert rectangle.pattern == from_dict.pattern
#     assert rectangle.centre_x == from_dict.centre_x
#     assert rectangle.centre_y == from_dict.centre_y
#     assert rectangle.width == from_dict.width
#     assert rectangle.height == from_dict.height
#     assert rectangle.rotation == from_dict.rotation
#     assert rectangle.depth == from_dict.depth
#     assert rectangle.scan_direction == from_dict.scan_direction
#     assert rectangle.cleaning_cross_section == from_dict.cleaning_cross_section


    