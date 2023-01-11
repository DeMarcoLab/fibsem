import logging
import sys
from enum import Enum
import traceback

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment, utils, milling, calibration
from fibsem.structures import BeamType, MicroscopeSettings, MillingSettings, Point
from fibsem.ui.qtdesigner_files import FibsemUI
from PyQt5 import QtCore, QtWidgets
from fibsem.detection.detection import FeatureType, Feature
from fibsem.detection import detection
from fibsem.detection import utils as det_utils

import napari.utils.notifications
import napari

from pprint import pprint


class MovementMode(Enum):
    Stable = 1
    Eucentric = 2

class FibsemUI(FibsemUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self, viewer: napari.Viewer, microscope: SdbMicroscopeClient = None, 
        settings: MicroscopeSettings = None, parent = None
    ):
        super(FibsemUI, self).__init__(parent = parent)
        self.setupUi(self)

        # connect to microscope, if required
        if microscope is None:
            self.microscope, self.settings = utils.setup_session()
    
        # flags
        self.MOVEMENT_ENABLED = False
        self.MILLING_ENABLED = False
        self.DETECTION_ENABLED = False

        self.setup_connections()
        self.update_ui_from_settings()
        self.update_ui_element_visibility()

        self.viewer: napari.Viewer = viewer
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True
        
        self.take_image()

    def setup_connections(self):       

        # imaging
        self.comboBox_imaging_beam_type.addItems([beam.name for beam in BeamType])
        self.comboBox_imaging_resolution.addItems([str(res) for res in self.microscope.beams.electron_beam.scanning.resolution.available_values])
        self.pushButton_update_imaging_settings.clicked.connect(self.update_image_settings)
        self.pushButton_take_image.clicked.connect(self.take_image)

        # movement
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.pushButton_enable_movement.clicked.connect(self.toggle_enable_movement)


        # milling
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        self.pushButton_update_milling_pattern.clicked.connect(self.update_patterns)
        self.comboBox_milling_current.addItems([f"{current:.2e}" for current in self.microscope.beams.ion_beam.beam_current.available_values])
        self.pushButton_enable_milling.clicked.connect(self.toggle_enable_milling)


        # detection
        self.pushButton_enable_detection.clicked.connect(self.toggle_enable_detection)
        self.pushButton_detection_move.clicked.connect(self.detection_based_movement)
        self.pushButton_detection_update_features.clicked.connect(self.update_detection_features)
        self.comboBox_detection_beam_type.addItems([beam.name for beam in BeamType])
        self.comboBox_detection_feature_1.currentTextChanged.connect(self.update_available_features)
        available_feature_1 = [FeatureType.LamellaCentre, FeatureType.LamellaRightEdge, FeatureType.NeedleTip]
        self.comboBox_detection_feature_1.addItems([feature.name for feature in available_feature_1])
        

        # tools
        self.pushButton_sputter_platinum.clicked.connect(self.run_tools)
        self.pushButton_move_stage_out.clicked.connect(self.run_tools)
        self.pushButton_auto_charge_neutralisation.clicked.connect(self.run_tools)
        self.pushButton_auto_focus_beam.clicked.connect(self.run_tools)
        self.pushButton_auto_home_stage.clicked.connect(self.run_tools)
        self.pushButton_auto_link_stage.clicked.connect(self.run_tools)
        self.pushButton_auto_needle_calibration.clicked.connect(self.run_tools)
        self.pushButton_validate_microscope_settings.clicked.connect(self.run_tools)




    def update_ui_element_visibility(self):

        # movement
        self.comboBox_movement_mode.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_mode.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_instructions.setVisible(self.MOVEMENT_ENABLED)
        self.label_movement_header.setVisible(self.MOVEMENT_ENABLED)

        # milling
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

        # detection
        self.comboBox_detection_beam_type.setVisible(self.DETECTION_ENABLED)
        self.label_detection_beam_type.setVisible(self.DETECTION_ENABLED)
        self.comboBox_detection_feature_1.setVisible(self.DETECTION_ENABLED)
        self.comboBox_detection_feature_2.setVisible(self.DETECTION_ENABLED)
        self.label_detection_feature_1.setVisible(self.DETECTION_ENABLED)
        self.label_detection_feature_2.setVisible(self.DETECTION_ENABLED)
        self.comboBox_detection_current_feature.setVisible(self.DETECTION_ENABLED)
        self.label_detection_current_feature.setVisible(self.DETECTION_ENABLED)
        self.label_detection_info.setVisible(self.DETECTION_ENABLED)
        self.pushButton_detection_move.setVisible(self.DETECTION_ENABLED)
        self.pushButton_detection_update_features.setVisible(self.DETECTION_ENABLED)
        self.checkBox_detection_move_x_axis.setVisible(self.DETECTION_ENABLED)
        self.checkBox_detection_move_y_axis.setVisible(self.DETECTION_ENABLED)
        self.label_detection_settings_header.setVisible(self.DETECTION_ENABLED)
        self.label_detection_movement_header.setVisible(self.DETECTION_ENABLED)
        

    def update_ui_from_settings(self):

        logging.info(f"updating ui")

        # imaging
        self.comboBox_imaging_beam_type.setCurrentText(self.settings.image.beam_type.name)
        self.comboBox_imaging_resolution.setCurrentText(self.settings.image.resolution)
        self.doubleSpinBox_imaging_dwell_time.setValue(self.settings.image.dwell_time * constants.METRE_TO_MICRON)
        self.doubleSpinBox_imaging_hfw.setValue(self.settings.image.hfw * constants.METRE_TO_MICRON)
        self.checkBox_imaging_use_autocontrast.setChecked(self.settings.image.autocontrast)
        self.checkBox_imaging_use_autogamma.setChecked(self.settings.image.gamma.enabled)
        self.checkBox_imaging_save_image.setChecked(self.settings.image.save)
        self.lineEdit_imaging_save_path.setText(self.settings.image.save_path)
        self.lineEdit_imaging_label.setText(self.settings.image.label)

    ####### IMAGING
    def update_image_settings(self):

        try:
            self.settings.image.beam_type = BeamType[self.comboBox_imaging_beam_type.currentText()]
            self.settings.image.resolution = str(self.comboBox_imaging_resolution.currentText())
            self.settings.image.dwell_time = float(self.doubleSpinBox_imaging_dwell_time.value()) * constants.MICRON_TO_METRE
            self.settings.image.hfw = float(self.doubleSpinBox_imaging_hfw.value()) * constants.MICRON_TO_METRE
            self.settings.image.autocontrast = bool(self.checkBox_imaging_use_autocontrast.isChecked())
            self.settings.image.gamma.enabled = bool(self.checkBox_imaging_use_autocontrast.isChecked())
            self.settings.image.save = bool(self.checkBox_imaging_save_image.isChecked())
            self.settings.image.save_path = str(self.lineEdit_imaging_save_path)
            self.settings.image.label = str(self.lineEdit_imaging_label)

        except:
            napari.utils.notifications.show_info(f"Unable to update image settings: {traceback.format_exc()}")


    def take_image(self):
        
        try:
            self.viewer.layers.clear()
            self.image = acquire.new_image(self.microscope, self.settings.image)
            self.image_layer = self.viewer.add_image(self.image.data, name=f"{self.settings.image.beam_type.name} Image")
        except:
            napari.utils.notifications.show_info(f"Unable to update image: {traceback.format_exc()}")

    ####### MOVEMENT
    def toggle_enable_movement(self):

        self.MOVEMENT_ENABLED = not self.MOVEMENT_ENABLED

        # show options
        self.update_ui_element_visibility()

        # disable other tabs while moving
        self.tabWidget.setTabEnabled(0, not self.MOVEMENT_ENABLED)
        self.tabWidget.setTabEnabled(2, not self.MOVEMENT_ENABLED)
        self.tabWidget.setTabEnabled(3, not self.MOVEMENT_ENABLED)
        self.tabWidget.setTabEnabled(4, not self.MOVEMENT_ENABLED)


        if self.MOVEMENT_ENABLED:
            self.pushButton_enable_movement.setText(f"Disable Movement")
            self.take_movement_image()
        else:
            self.pushButton_enable_movement.setText(f"Enable Movement")
            self.take_image()
            
    def _double_click(self, layer, event):

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve

        # check inside image dimensions, (y, x)
        eb_shape = self.image.data.shape[0], self.image.data.shape[1] // 2
        ib_shape = self.image.data.shape[0], self.image.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (coords[1] > 0 and coords[1] < eb_shape[1]):
            adorned_image = self.eb_image
            beam_type = BeamType.ELECTRON

        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (coords[1] > eb_shape[0] and coords[1] < ib_shape[1]):
            adorned_image = self.ib_image
            coords = (coords[0], coords[1] - ib_shape[1] // 2)
            beam_type = BeamType.ION
        else:
            napari.utils.notifications.show_info(f"Clicked outside image dimensions. Please click inside the image to move.")
            return

        point = conversions.image_to_microscope_image_coordinates(Point(x=coords[1], y=coords[0]), 
                adorned_image.data, adorned_image.metadata.binary_result.pixel_size.x)  

        logging.info(f"coords: {coords}, beam_type: {beam_type}")
        logging.info(f"movement: x={point.x:.2e}, y={point.y:.2e}")

        # move

        self.movement_mode = MovementMode[self.comboBox_movement_mode.currentText()]

        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            logging.info(f"moving eucentricly in {beam_type}")

            movement.move_stage_eucentric_correction(
                microscope=self.microscope, 
                settings=self.settings,
                dy=-point.y
            )

        else:
            logging.info(f"moving stably in {beam_type}")
            # corrected stage movement
            movement.move_stage_relative_with_corrected_movement(
                microscope=self.microscope,
                settings=self.settings,
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )

        # take new image
        self.take_movement_image()


    def take_movement_image(self):

        try:
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
        except:
            napari.utils.notifications.show_info(f"Unable to update movement image: {traceback.format_exc()}")
        # TODO: if you attach the callback to all image layers, can enable movement everywhere

    ####### MILLING

    def toggle_enable_milling(self):

        self.MILLING_ENABLED = not self.MILLING_ENABLED

        # show options
        self.update_ui_element_visibility()

        # disable other tabs while moving
        self.tabWidget.setTabEnabled(0, not self.MILLING_ENABLED)
        self.tabWidget.setTabEnabled(1, not self.MILLING_ENABLED)
        self.tabWidget.setTabEnabled(3, not self.MILLING_ENABLED)
        self.tabWidget.setTabEnabled(4, not self.MILLING_ENABLED)

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
            
            
            # TODO: maybe just scale this by the current factor and see how close it gets?
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


    #### DETECTION

    def toggle_enable_detection(self):

        self.DETECTION_ENABLED = not self.DETECTION_ENABLED

        # show options
        self.update_ui_element_visibility()

        # disable other tabs while moving
        self.tabWidget.setTabEnabled(0, not self.DETECTION_ENABLED)
        self.tabWidget.setTabEnabled(1, not self.DETECTION_ENABLED)
        self.tabWidget.setTabEnabled(2, not self.DETECTION_ENABLED)
        self.tabWidget.setTabEnabled(4, not self.DETECTION_ENABLED)

        if self.DETECTION_ENABLED:
            self.pushButton_enable_detection.setText(f"Disable Detection")
            self.update_detection_features()

        else:
            self.pushButton_enable_detection.setText(f"Enable Detection")
            self.take_image()

    def update_available_features(self):

        feature_1 = FeatureType[self.comboBox_detection_feature_1.currentText()]

        if feature_1 is FeatureType.NeedleTip:
            available_feature_2 = [FeatureType.LamellaCentre, FeatureType.ImageCentre]

        if feature_1 is FeatureType.LamellaRightEdge:
            available_feature_2 = [FeatureType.LandingPost]

        if feature_1 is FeatureType.LamellaCentre:
            available_feature_2 = [FeatureType.ImageCentre]

        self.comboBox_detection_feature_2.clear()
        self.comboBox_detection_feature_2.addItems([feature.name for feature in available_feature_2])

    def update_detection_features(self):
        logging.info("update detection features")

        features_names = [self.comboBox_detection_feature_1.currentText(),
                    self.comboBox_detection_feature_2.currentText()]
        self.comboBox_detection_current_feature.clear()
        self.comboBox_detection_current_feature.addItems(features_names)

        self.take_detection_image()
        features = [
            Feature(FeatureType[features_names[0]], Point(0, 0)),
            Feature(FeatureType[features_names[1]], Point(0, 0))
            ]
        self.detection = detection.locate_shift_between_features(self.image, features=features)
        
        self.update_detection_image()

    def take_detection_image(self):
        
        self.viewer.layers.clear()
        image_settings = self.settings.image
        image_settings.beam_type = BeamType[self.comboBox_detection_beam_type.currentText()] 
        self.image = acquire.new_image(self.microscope, image_settings)
        
        self.det_image_layer = self.viewer.add_image(self.image.data, name=f"{image_settings.beam_type.name} Image", opacity=0.7)
        self.det_image_layer.mouse_drag_callbacks.append(self._update_detection_features_from_click) # append callback

        self.det_image_layer.mouse_wheel_callbacks.append(self._experimental_mouse_wheel_callback) # append callback

    def _experimental_mouse_wheel_callback(self, layer, event):

        logging.info(f"Layer: {layer}, Event: {event}, {event.type}")

    def _update_detection_features_from_click(self, layer, event):
    
        if event.type == "mouse_press":
            # get coords
            coords = layer.world_to_data(event.position)

            logging.info(f"Layer: {layer}, Event: {event}")
            logging.info(f"Coords: {coords}")

            current_idx = self.comboBox_detection_current_feature.currentIndex()
            self.detection.features[current_idx].feature_px = Point(x=coords[1], y=coords[0])

            x_distance_m, y_distance_m = det_utils.convert_pixel_distance_to_metres(
                self.detection.features[0].feature_px, 
                self.detection.features[1].feature_px, 
                self.image
            )
            self.detection.distance_metres = Point(
                x_distance_m, y_distance_m
            )  # move from 1 to 2 (reverse direction)

            self.update_detection_image()

    def update_detection_image(self):
        
        # plot detection features as points
        f1 = self.detection.features[0] 
        f2 = self.detection.features[1] 
            
        feature_points = [
            (f1.feature_px.y, f1.feature_px.x),
             (f2.feature_px.y, f2.feature_px.x)]

        features = {"label": [f1.type.name, 
                f2.type.name],
                "choice": [True, False]}

        face_color_cycle = [
            det_utils.DETECTION_TYPE_COLOURS_v2[f1.type], 
            det_utils.DETECTION_TYPE_COLOURS_v2[f2.type]
            ]

        # TODO: why doesnt the text label work??
        text = {"string": '{label}', "size": 20, "translation": np.array([-5, -5]), "color":"white"}

        try:
            self.viewer.layers["Detection"].data = feature_points
        except:
            self.viewer.add_points(feature_points, 
                    name="Detection",
                    features=features, 
                    text=text,
                    size=25, 
                    face_color="choice",
                    face_color_cycle=face_color_cycle,
            )

        # update ui label
        det_metres = self.detection.distance_metres
        self.label_detection_info.setText(f"Moving: x={det_metres.x:.2e}m, y={det_metres.y:.2e}m")

        # set image layer activate after updating points
        self.viewer.layers.selection.active = self.det_image_layer



    def detection_based_movement(self):
        logging.info("detection based movement")

        move_x = self.checkBox_detection_move_x_axis.isChecked()
        move_y = self.checkBox_detection_move_y_axis.isChecked()

        beam_type = BeamType[self.comboBox_detection_beam_type.currentText()]
        movement.move_based_on_detection(self.microscope, self.settings, self.detection, beam_type, move_x, move_y)


    ##### TOOLS
    def run_tools(self):

        sender = self.sender()
        logging.info(f"Sender: {sender}")

        if sender == self.pushButton_sputter_platinum:
            logging.info(f"Sputtering Platinum")

        if sender == self.pushButton_move_stage_out:
            logging.info(f"Moving Stage Out")

        if sender == self.pushButton_auto_focus_beam:
            logging.info(f"Auto Focus")

        if sender == self.pushButton_auto_charge_neutralisation:
            logging.info(f"Auto Discharge Beam")
            calibration.auto_charge_neutralisation(self.microscope, self.settings.image)

        if sender == self.pushButton_auto_home_stage:
            logging.info(f"Auto Home Stage")

        if sender == self.pushButton_auto_link_stage:
            logging.info(f"Auto Link Stage")
        
        if sender == self.pushButton_auto_needle_calibration:
            logging.info(f"Auto Needle Calibration")

        if sender == self.pushButton_validate_microscope_settings:
            logging.info(f"Validating Microscope Settings")
        
        self.take_image()

# TODO: START HERE
# TODO: test detection + movement
# TODO: tools, settings, changing settings, validate settings page
# TODO: piescope_v2

# TODO: live streaming

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
