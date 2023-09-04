
from fibsem.structures import (BeamType, FibsemMillingSettings,
                               FibsemPatternSettings, MicroscopeSettings,
                               Point, FibsemPattern,FibsemImage,FibsemImageMetadata)
from fibsem.patterning import FibsemMillingStage    
from fibsem.ui.utils import import_milling_stages_yaml_file, _draw_patterns_in_napari
import napari                   
import matplotlib.pyplot as plt       

milling_stages = import_milling_stages_yaml_file(r'C:\Users\rkan0039\Documents\codeFIBSEM\fibsem\fibsem\log\test3.yaml')
test_image = FibsemImage.load(r'C:\Users\rkan0039\Documents\codeFIBSEM\fibsem\fibsem\log\test_image_1_ib.tif')

viewer = napari.Viewer()
viewer.add_image(test_image.data, name='test_image')
_draw_patterns_in_napari(viewer=viewer,ib_image=test_image,eb_image=None,milling_stages=milling_stages)
# viewer.save_layers(r'C:\Users\rkan0039\Documents\codeFIBSEM\fibsem\fibsem\log\sc_test.png')
# napari.save_layers(r'C:\Users\rkan0039\Documents\codeFIBSEM\fibsem\fibsem\log\sc_test.png',viewer.layers)
# viewer.camera.zoom = 6.4
# viewer.camera.center = (63, 100, 105)
# viewer.camera.angles = (17, -50, 73)

screenshot = viewer.screenshot()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(screenshot)
ax.axis('off')
plt.show()
# fig.savefig(r'C:\Users\rkan0039\Documents\codeFIBSEM\fibsem\fibsem\log\sc_test.png', dpi=300, bbox_inches='tight', pad_inches=0)