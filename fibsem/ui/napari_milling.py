import logging
import sys
from enum import Enum

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment, utils, milling
from fibsem.structures import BeamType, MicroscopeSettings, MillingSettings
from fibsem.ui.qtdesigner_files import NapariMilling
from PyQt5 import QtCore, QtWidgets

import napari.utils.notifications
import napari

# TODO: maybe have to change to dialog?


from fibsem.structures import ImageSettings

class NapariMillingUI(NapariMilling.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self, viewer: napari.Viewer, parent = None
    ):
        super(NapariMillingUI, self).__init__(parent = parent)
        self.setupUi(self)

        self.microscope, self.settings = utils.setup_session()
        self.settings.image.beam_type = BeamType.ION

        self.setup_connections()

        self.viewer = viewer
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        self.take_image()
    
    def setup_connections(self):

        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_take_image.clicked.connect(self.take_image)
        self.pushButton_update_milling_pattern.clicked.connect(self.update_patterns)

        # combobox
        self.comboBox_milling_current.addItems([f"{current:.2e}" for current in self.microscope.beams.ion_beam.beam_current.available_values])
    
    def take_image(self):
        
        # update settings, take image
        self.settings.image.hfw = float(self.doubleSpinBox_imaging_hfw.value() * constants.MICRON_TO_METRE)
        self.ib_image = acquire.new_image(self.microscope, self.settings.image)

        # refresh viewer
        self.viewer.layers.clear()
        self.viewer.add_image(self.ib_image.data, name="IB Image")
        self.viewer.add_shapes(None, name="Patterns")

    def update_patterns(self):
        logging.info(f"update patterns")

        patterns = []
        pname = "Patterns"
        depth = float(self.doubleSpinBox_milling_depth.value() * constants.MICRON_TO_METRE)

        self.microscope.patterning.clear_patterns()

        if pname in self.viewer.layers:
            for shape, arr in zip(self.viewer.layers[pname].shape_type, self.viewer.layers[pname].data):
                if shape == "rectangle":
                    
                    pixelsize = self.ib_image.metadata.binary_result.pixel_size.x
                    mill_settings = convert_napari_rect_to_mill_settings(arr, image=self.ib_image.data, pixelsize=pixelsize, depth=depth)

                    try:
                        pattern = milling._draw_rectangle_pattern_v2(self.microscope, mill_settings)
                        patterns.append(pattern)
                    except Exception as e:
                        napari.utils.notifications.show_info(f"Exception: {e}")
                    
                    
            estimated_time = utils._format_time_seconds(sum([pattern.time for pattern in patterns]))
            self.label_milling_estimated_time.setText(f"Estimated Time: {estimated_time}") 

            # TODO: convert pattern to napari shape..

        pixelsize = self.ib_image.metadata.binary_result.pixel_size.x

        shape_patterns = []
        if patterns:
            for pattern in patterns:

                shape = convert_pattern_to_napari_rect(pattern, self.ib_image.data, pixelsize)
                shape_patterns.append(shape)

        self.viewer.add_shapes(shape_patterns, name="Reverse", shape_type='rectangle', edge_width=1,
                                edge_color='royalblue', face_color='royalblue')

        # TODO: get currently selected pattern
        print(self.viewer.layers[pname].selected_data)

    def run_milling(self):
        milling_current = float(self.comboBox_milling_current.currentText())
        milling.run_milling(self.microscope, milling_current)

def convert_pattern_to_napari_rect(pattern, image: np.ndarray, pixelsize: float) -> np.ndarray:
    
    # image centre
    icy, icx = image.shape[0] // 2, image.shape[1]// 2

    # pattern to pixel coords
    w = int(pattern.width / pixelsize)
    h = int(pattern.height  / pixelsize)
    cx = int(icx + (pattern.center_x / pixelsize))
    cy = int(icy - (pattern.center_y / pixelsize))

    xmin, xmax = cx-w/2, cx+w/2
    ymin, ymax = cy-h/2, cy+h/2

    # napari shape format
    shape = [[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]]

    return shape

def convert_napari_rect_to_mill_settings(arr: np.array, image: np.array, pixelsize: float, depth: float = 10e-6) -> MillingSettings:
    # convert napari rect to milling pattern

    # get centre of image
    cy_mid, cx_mid = image.data.shape[0] // 2, image.shape[1] // 2

    # get rect dimensions in px
    ymin, xmin = arr[0]
    ymax, xmax = arr[2]
    
    width = int(xmax - xmin)
    height = int(ymax - ymin)

    cx = int(xmin + width / 2)
    cy = int(ymin + height / 2)


    # get rect dimensions in real space 
    cy_real = (cy_mid - cy) * pixelsize
    cx_real = -(cx_mid - cx) * pixelsize
    width = width * pixelsize
    height = height * pixelsize

    # set milling settings
    mill_settings = MillingSettings(width=width, height=height, depth=depth, centre_x=cx_real, centre_y=cy_real)

    return mill_settings



# TODO: override enter


def main():
    # from liftout import utils
    # from fibsem.ui import windows as fibsem_ui_windows
    # microscope, settings= utils.quick_setup()


    app = QtWidgets.QApplication([])
    viewer = napari.Viewer(ndisplay=2)
    napari_milling_ui = NapariMillingUI(viewer=viewer)
    viewer.window.add_dock_widget(napari_milling_ui, area='right')  

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
