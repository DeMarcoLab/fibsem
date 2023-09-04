
from fibsem.structures import (BeamType, FibsemMillingSettings,
                               FibsemPatternSettings, MicroscopeSettings,
                               Point, FibsemPattern,FibsemImage,FibsemImageMetadata)
from fibsem.patterning import FibsemMillingStage    
from fibsem.ui.utils import import_milling_stages_yaml_file                          

milling_stages = import_milling_stages_yaml_file(r'C:\Users\rkan0039\Documents\codeFIBSEM\fibsem\fibsem\log\test3.yaml')
test_image = FibsemImage.load(r'C:\Users\rkan0039\Documents\codeFIBSEM\fibsem\fibsem\log\test_image_1_ib.tif')

print(test_image.metadata)