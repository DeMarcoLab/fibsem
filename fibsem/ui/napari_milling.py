import logging
from multiprocessing import reduction
import sys
from enum import Enum

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment, utils, milling, conversions
from fibsem.structures import BeamType, MicroscopeSettings, MillingSettings
from fibsem.ui.qtdesigner_files import NapariMilling
from PyQt5 import QtCore, QtWidgets

import napari.utils.notifications
import napari

# TODO: maybe have to change to dialog?

class MovementMode(Enum):
    Stable = 1
    Eucentric = 2

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

        self.viewer: napari.Viewer = viewer
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        self.take_image()

        # key bindings
        self.viewer.bind_key("f", self._func)
    
    def setup_connections(self):

        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_take_image.clicked.connect(self.take_image)
        self.pushButton_update_milling_pattern.clicked.connect(self.update_patterns)

        # combobox
        self.comboBox_milling_current.addItems([f"{current:.2e}" for current in self.microscope.beams.ion_beam.beam_current.available_values])

    def _func(self, viewer):
        napari.utils.notifications.show_info("Taking a new image.")
        self.take_image()


    def _double_click(self, layer, event):
        print(f"Layer: {layer}")
        print(f"Event: {event.position},  {layer.world_to_data(event.position)}, {event.type}")


        # get coords
        coords = layer.world_to_data(event.position)

        # check inside image dimensions, (y, x)
        eb_shape = self.image.data.shape[0], self.image.data.shape[1] // 2
        ib_shape = self.image.data.shape[0], self.image.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (coords[1] > 0 and coords[1] < eb_shape[1]):
            adorned_image = self.eb_image
            beam_type = BeamType.ELECTRON
        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (coords[1] > eb_shape[0] and coords[1] < ib_shape[1]):
            adorned_image = self.ib_image
            coords = (coords[0], coords[1] - ib_shape[1])
            beam_type = BeamType.ION
        else:
            napari.utils.notifications.show_info(f"Clicked outside image dimensions")
            return


        point = conversions.image_to_microscope_image_coordinates(
                (coords[1], coords[0]), adorned_image, adorned_image.metadata.binary_result.pixel_size.x
            )
        napari.utils.notifications.show_info(f"Inside {beam_type.name} Image Dimensions - centre: {point.x}, {point.y}")

        # move

        self.movement_mode = MovementMode.Stable
        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            logging.info(f"moving eucentricly in {beam_type}")

            movement.move_stage_eucentric_correction(
                microscope=self.microscope, 
                settings=self.settings,
                dy=-point.y
            )

        else:
            # corrected stage movement
            movement.move_stage_relative_with_corrected_movement(
                microscope=self.microscope,
                settings=self.settings,
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )



    def take_image(self):
        
        # update settings, take image
        self.settings.image.hfw = float(self.doubleSpinBox_imaging_hfw.value() * constants.MICRON_TO_METRE)
        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.settings.image)

        self.image = np.concatenate([self.eb_image.data, self.ib_image.data], axis=1)

        # refresh viewer
        self.viewer.layers.clear()
        # self.ion_image = self.viewer.add_image(self.ib_image.data, name="IB Image")
        # self.pattern_layer = self.viewer.add_shapes(None, name="Patterns")


        # crosshair
        cy, cx_eb = self.image.shape[0] // 2, self.image.shape[1] // 4 
        cx_ib = cx_eb + self.image.shape[1] // 2 

        self.points_layer = self.viewer.add_points(
            data=[[cy, cx_eb], [cy, cx_ib]], 
            symbol="cross", size=50,
            edge_color="yellow", face_color="yellow",
        )
        self.points_layer.editable = False

        self.image_layer = self.viewer.add_image(self.image, name="Images", opacity=0.9)
        self.image_layer.mouse_double_click_callbacks.append(self._double_click)

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

            # convert pattern to napari shape..

        pixelsize = self.ib_image.metadata.binary_result.pixel_size.x

        shape_patterns = []
        if patterns:
            for pattern in patterns:

                shape = convert_pattern_to_napari_rect(pattern, self.ib_image.data, pixelsize)
                shape_patterns.append(shape)

        self.viewer.add_shapes(shape_patterns, name="Reverse", shape_type='rectangle', edge_width=1,
                                edge_color='royalblue', face_color='royalblue')

        # get currently selected pattern
        selected_data_idx = list(self.viewer.layers[pname].selected_data)
        data = self.viewer.layers[pname].data 
        selected_data = [data[i] for i in selected_data_idx]
        print(selected_data)

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


def main():
    
    app = QtWidgets.QApplication([])
    viewer = napari.Viewer(ndisplay=2)
    napari_milling_ui = NapariMillingUI(viewer=viewer)
    viewer.window.add_dock_widget(napari_milling_ui, area='right')  

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
