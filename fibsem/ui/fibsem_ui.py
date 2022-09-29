import logging
import sys
from enum import Enum
import traceback

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment, utils, milling
from fibsem.structures import BeamType, MicroscopeSettings, MillingSettings
from fibsem.ui.qtdesigner_files import FibsemUI
from PyQt5 import QtCore, QtWidgets

import napari.utils.notifications
import napari

from pprint import pprint


class MovementMode(Enum):
    Stable = 1
    Eucentric = 2

class FibsemUI(FibsemUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self, viewer: napari.Viewer, parent = None
    ):
        super(FibsemUI, self).__init__(parent = parent)
        self.setupUi(self)

        # connect to microscope
        self.microscope, self.settings = utils.setup_session()
    
        # flags
        self.MOVEMENT_ENABLED = False
        self.MILLING_ENABLED = False

        self.setup_connections()
        self.update_ui_from_settings()

        self.viewer: napari.Viewer = viewer
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        
        self.take_image()

    def setup_connections(self):       

        # imaging
        self.comboBox_imaging_beam_type.addItems([beam.name for beam in BeamType])
        self.pushButton_update_imaging_settings.clicked.connect(self.update_image_settings)
        self.pushButton_take_image.clicked.connect(self.take_image)

        # movement
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.pushButton_enable_movement.clicked.connect(self.toggle_enable_movement)
        self.comboBox_movement_mode.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_mode.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_instructions.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_header.setVisible(self.MOVEMENT_ENABLED)

        # milling
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_update_milling_pattern.clicked.connect(self.update_patterns)
        self.comboBox_milling_current.addItems([f"{current:.2e}" for current in self.microscope.beams.ion_beam.beam_current.available_values])
        self.pushButton_enable_milling.clicked.connect(self.toggle_enable_milling)
        self.label_milling_header.setVisible(self.MILLING_ENABLED)
        self.label_milling_pattern.setVisible(self.MILLING_ENABLED)
        self.label_milling_depth.setVisible(self.MILLING_ENABLED)
        self.label_milling_current.setVisible(self.MILLING_ENABLED)
        self.comboBox_milling_pattern.setVisible(self.MILLING_ENABLED)
        self.comboBox_milling_current.setVisible(self.MILLING_ENABLED)
        self.doubleSpinBox_milling_depth.setVisible(self.MILLING_ENABLED)
        self.pushButton_run_milling.setVisible(self.MILLING_ENABLED)
        self.pushButton_update_milling_pattern.setVisible(self.MILLING_ENABLED)
        self.label_milling_estimated_time.setVisible(self.MILLING_ENABLED)
        

    def update_ui_from_settings(self):

        logging.info(f"updating ui")

        # imaging
        self.comboBox_imaging_beam_type.setCurrentText(self.settings.image.beam_type.name)
        self.lineEdit_imaging_resolution.setText(self.settings.image.resolution)
        self.doubleSpinBox_imaging_dwell_time.setValue(self.settings.image.dwell_time * constants.METRE_TO_MICRON)
        self.doubleSpinBox_imaging_hfw.setValue(self.settings.image.hfw * constants.METRE_TO_MICRON)
        self.checkBox_imaging_use_autocontrast.setChecked(self.settings.image.autocontrast)
        self.checkBox_imaging_use_autogamma.setChecked(self.settings.image.gamma.enabled)
        self.checkBox_imaging_save_image.setChecked(self.settings.image.save)
        self.lineEdit_imaging_save_path.setText(self.settings.image.save_path)
        self.lineEdit_imaging_label.setText(self.settings.image.label)

    ####### IMAGING
    def update_image_settings(self):

        logging.info(f"Updating image settings")

        try:
            self.settings.image.beam_type = BeamType[self.comboBox_imaging_beam_type.currentText()]
            self.settings.image.resolution = str(self.lineEdit_imaging_resolution.text())
            self.settings.image.dwell_time = float(self.doubleSpinBox_imaging_dwell_time.value()) * constants.MICRON_TO_METRE
            self.settings.image.hfw = float(self.doubleSpinBox_imaging_hfw.value()) * constants.MICRON_TO_METRE
            self.settings.image.autocontrast = bool(self.checkBox_imaging_use_autocontrast.isChecked())
            self.settings.image.gamma.enabled = bool(self.checkBox_imaging_use_autocontrast.isChecked())
            self.settings.image.save = bool(self.checkBox_imaging_save_image.isChecked())
            self.settings.image.save_path = str(self.lineEdit_imaging_save_path)
            self.settings.image.label = str(self.lineEdit_imaging_label)

        except:
            napari.utils.notifications.show_info(f"Unable to update image settings: {traceback.format_exc()}")

        pprint(self.settings.image)

    def take_image(self):
        
        self.viewer.layers.clear()
        self.image = acquire.new_image(self.microscope, self.settings.image)
        self.image_layer = self.viewer.add_image(self.image.data, name=f"{self.settings.image.beam_type.name} Image")


    ####### MOVEMENT
    def toggle_enable_movement(self):

        self.MOVEMENT_ENABLED = not self.MOVEMENT_ENABLED

        # show options
        self.comboBox_movement_mode.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_mode.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_instructions.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_header.setVisible(self.MOVEMENT_ENABLED)

        # disable other tabs while moving
        self.tabWidget.setTabEnabled(0, not self.MOVEMENT_ENABLED)
        self.tabWidget.setTabEnabled(2, not self.MOVEMENT_ENABLED)
        self.tabWidget.setTabEnabled(3, not self.MOVEMENT_ENABLED)

        if self.MOVEMENT_ENABLED:
            self.pushButton_enable_movement.setText(f"Disable Movement")
            self.take_movement_image()
        else:
            self.pushButton_enable_movement.setText(f"Enable Movement")
            self.take_image()
            
    def _double_click(self, layer, event):
        # print(f"Layer: {layer}")
        # print(f"Event: {event.position},  {layer.world_to_data(event.position)}, {event.type}")

        # get coords
        coords = layer.world_to_data(event.position)

        # check inside image dimensions, (y, x)
        eb_shape = self.image.data.shape[0], self.image.data.shape[1] // 2
        ib_shape = self.image.data.shape[0], self.image.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (coords[1] > 0 and coords[1] < eb_shape[1]):
            napari.utils.notifications.show_info("Inside EB Image Dimensions")
            adorned_image = self.eb_image
            beam_type = BeamType.ELECTRON

        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (coords[1] > eb_shape[0] and coords[1] < ib_shape[1]):
            napari.utils.notifications.show_info("Inside IB Image Dimensions")
            adorned_image = self.ib_image
            coords = (coords[0], coords[1] - ib_shape[1])
            beam_type = BeamType.ION
        else:
            napari.utils.notifications.show_info(f"Outside Image Dimensions")
            return

        center_x, center_y = conversions.pixel_to_realspace_coordinate(
                (coords[1], coords[0]), adorned_image
            )


        logging.info(f"movement: x={center_x:.2e}, y={center_y:.2e}")

        # move

        self.movement_mode = MovementMode.Stable
        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            logging.info(f"moving eucentricly in {beam_type}")

            # movement.move_stage_eucentric_correction(
            #     microscope=self.microscope, 
            #     dy=-center_y
            # )

        else:
            logging.info(f"moving stably in {beam_type}")
            # corrected stage movement
            # movement.move_stage_relative_with_corrected_movement(
            #     microscope=self.microscope,
            #     settings=self.settings,
            #     dx=center_x,
            #     dy=center_y,
            #     beam_type=beam_type,
            # )


        # take new image
        self.take_movement_image()


    def take_movement_image(self):
        
        # update settings, take image
        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.settings.image)
        self.image = np.concatenate([self.eb_image.data, self.ib_image.data], axis=1) # stack both images together

        # crosshair
        # cy, cx_eb = self.image.shape[0] // 2, self.image.shape[1] // 4 
        # cx_ib = cx_eb + self.image.shape[1] // 2 
        
        # # refresh viewer
        self.viewer.layers.clear()
        # self.points_layer = self.viewer.add_points(
        #     data=[[cy, cx_eb], [cy, cx_ib]], 
        #     symbol="cross", size=50,
        #     edge_color="yellow", face_color="yellow",
        # )
        # self.points_layer.editable = False

        self.image_layer = self.viewer.add_image(self.image, name="Images", opacity=0.9)
        self.image_layer.mouse_double_click_callbacks.append(self._double_click) # append callback

        # TODO: if you attach the callback to all image layers, can enable movement everywhere

    ####### MILLING

    def toggle_enable_milling(self):

        self.MILLING_ENABLED = not self.MILLING_ENABLED

        # show options
        self.label_milling_header.setVisible(self.MILLING_ENABLED)
        self.label_milling_pattern.setVisible(self.MILLING_ENABLED)
        self.label_milling_depth.setVisible(self.MILLING_ENABLED)
        self.label_milling_current.setVisible(self.MILLING_ENABLED)
        self.comboBox_milling_pattern.setVisible(self.MILLING_ENABLED)
        self.comboBox_milling_current.setVisible(self.MILLING_ENABLED)
        self.doubleSpinBox_milling_depth.setVisible(self.MILLING_ENABLED)
        self.pushButton_run_milling.setVisible(self.MILLING_ENABLED)
        self.pushButton_update_milling_pattern.setVisible(self.MILLING_ENABLED)
        self.label_milling_estimated_time.setVisible(self.MILLING_ENABLED)
        

        # disable other tabs while moving
        self.tabWidget.setTabEnabled(0, not self.MILLING_ENABLED)
        self.tabWidget.setTabEnabled(1, not self.MILLING_ENABLED)
        self.tabWidget.setTabEnabled(3, not self.MILLING_ENABLED)

        if self.MILLING_ENABLED:
            self.pushButton_enable_milling.setText(f"Disable Milling")
            self.take_milling_image()
        else:
            self.pushButton_enable_milling.setText(f"Enable Milling")
            self.take_image()



    def take_milling_image(self):

        self.viewer.layers.clear()
        self.ib_image = acquire.last_image(self.microscope, beam_type=BeamType.ION)
        self.ion_image = self.viewer.add_image(self.ib_image.data, name=f"{BeamType.ION.name} Image")
        self.pattern_layer = self.viewer.add_shapes(None, name="Patterns")

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

        
        # getting napari rect from patterns...
        # pixelsize = self.ib_image.metadata.binary_result.pixel_size.x

        # shape_patterns = []
        # if patterns:
        #     for pattern in patterns:

        #         shape = convert_pattern_to_napari_rect(pattern, self.ib_image.data, pixelsize)
        #         shape_patterns.append(shape)

        #     self.viewer.add_shapes(shape_patterns, name="Reverse", shape_type='rectangle', edge_width=1,
        #                             edge_color='royalblue', face_color='royalblue')

        # # TODO: get currently selected pattern
        # selected_data_idx = list(self.viewer.layers[pname].selected_data)
        # data = self.viewer.layers[pname].data 
        # selected_data = [data[i] for i in selected_data_idx]
        # print(selected_data)


    def run_milling(self):
        milling_current = float(self.comboBox_milling_current.currentText())
        try:
            milling.run_milling(self.microscope, milling_current)
        except:
            napari.utils.notifications.show_info(f"Unable to mill: {traceback.format_exc()}")

# TODO: detection
# TODO: tools, settings, changing settings
# TODO: liftout,
# TODO: piescope

# TODO: live streaming
# TODO: move somewhere

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
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, area='right')  

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
