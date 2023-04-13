import os
from copy import deepcopy
from pathlib import Path

import napari
import napari.utils.notifications
import numpy as np
import tifffile as tff
from PyQt5 import QtWidgets

from fibsem import config as cfg
from fibsem import utils
from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.microscope import FibsemMicroscope
from fibsem.segmentation import model as fibsem_model
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    BeamType,
    FibsemImage,
    ImageSettings,
    MicroscopeSettings,
    Point,
)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemDetectionWidget
import logging

class FibsemDetectionWidgetUI(FibsemDetectionWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        detected_features: DetectedFeatures = None,
        image: FibsemImage = None,
        parent=None,
    ):
        super(FibsemDetectionWidgetUI, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image = image

        self.setup_connections()

        self._USER_CORRECTED = False

        # set detected features
        if detected_features is not None:
            self.set_detected_features(detected_features)

    def setup_connections(self):
        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press continue when finished."""
        )
        self.pushButton_continue.clicked.connect(self.continue_button_clicked)

        self.pushButton_run_feature_detection.clicked.connect(
            self.run_feature_detection
        )

        self.pushButton_test_function.setText(f"Move Feature Positions")
        self.pushButton_test_function.setVisible(False)
        self.checkBox_move_features.clicked.connect(self.toggle_feature_interaction)
        self.checkBox_move_features.setEnabled(False)  # disabled until features are loaded

        self.checkBox_show_mask.setVisible(False)

        self.pushButton_load_model.clicked.connect(self.load_model)

        self.comboBox_feature_1.addItems(
            [feature.name for feature in detection.__FEATURES__]
        )
        self.comboBox_feature_2.addItems(
            [feature.name for feature in detection.__FEATURES__]
        )

        self.comboBox_beam_type.addItems([beam_type.name for beam_type in BeamType])

        self.lineEdit_encoder.setText("resnet34")
        self.lineEdit_checkpoint.setText(
            os.path.join(os.path.dirname(fibsem_model.__file__), "models", "model4.pt")
        )
        self.spinBox_num_classes.setValue(3)

    def toggle_feature_interaction(self):
        if "features" in self.viewer.layers:
            self.checkBox_move_features.setEnabled(True)
            
        if self.checkBox_move_features.isChecked():
            self.viewer.layers.selection.active = self.viewer.layers["features"]

        else:
            self.viewer.layers.selection.active = self.viewer.layers["image"]

    def run_feature_detection(self):

        self._USER_CORRECTED = False # reset user corrected flag

        image = self.image

        features = (
            detection.__FEATURES__[self.comboBox_feature_1.currentIndex()](),
            detection.__FEATURES__[self.comboBox_feature_2.currentIndex()](),
        )

        # detect features
        pixelsize = 25e-9 # TODO: get from metadata
        det = detection.locate_shift_between_features_v2(
            deepcopy(image.data), self.model, features=features, pixelsize=pixelsize
        )

        self.set_detected_features(det)

    def load_model(self):
        checkpoint = self.lineEdit_checkpoint.text()
        encoder = self.lineEdit_encoder.text()
        num_classes = self.spinBox_num_classes.value()
        logging.info(f"loading checkpoint: {checkpoint}...")
        self.model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)
        napari.utils.notifications.show_info(
            f"Model loaded: {os.path.basename(checkpoint)}"
        )

    def load_feature_detection(self, det: DetectedFeatures):

        # load feature detection

        # update ui elements
        self.comboBox_feature_1.setCurrentText(det.features[0].name)
        self.comboBox_feature_2.setCurrentText(det.features[1].name)

        # update detected features
        self.set_detected_features(det)

    def test_function(self):
        print("test function....")

    def continue_button_clicked(self):
        # save all images and coordinates for testing
        det_utils.save_data(det = self.detected_features, corrected=self._USER_CORRECTED)

        # move based on detected features
        # TODO: add x, y limits to UI
        detection.move_based_on_detection(
            self.microscope,
            self.settings,
            det=self.detected_features,
            beam_type=BeamType[self.comboBox_beam_type.currentText()],
        )

    def set_detected_features(self, det_features: DetectedFeatures):
        self.detected_features = det_features

        self.update_features_ui()

    def update_features_ui(self):
        # update combo box
        self.comboBox_feature_1.setCurrentText(self.detected_features.features[0].name)
        self.comboBox_feature_2.setCurrentText(self.detected_features.features[1].name)

        try:
            self.viewer.layers["image"].data = self.detected_features.image
        except:
            self.viewer.add_image(
                self.detected_features.image, name="image", opacity=0.3
            )

        # add mask to viewer
        try:
            self.viewer.layers["mask"].data = self.detected_features.mask
        except:
            self.viewer.add_image(self.detected_features.mask, name="mask", opacity=0.3)

        # add points to viewer
        data = []
        for feature in self.detected_features.features:
            x, y = feature.feature_px
            data.append([y, x])

        # if the features layer already exists, remove the layer
        if "features" in self.viewer.layers:
            self.viewer.layers.remove("features")

        text = {
            "string": [feature.name for feature in self.detected_features.features],
            "color": "white",
            "translation": np.array([-30, 0]),
        }

        self.viewer.add_points(
            data,
            name="features",
            text=text,
            size=20,
            edge_width=7,
            edge_width_is_relative=False,
            edge_color="transparent",
            face_color=[feature.color for feature in self.detected_features.features],
            blending="translucent",
        )

        # set points layer to select mode and active
        self.viewer.layers["features"].mode = "select"

        # when the point is moved update the feature
        self.viewer.layers["features"].mouse_drag_callbacks.append(self.point_moved)

        self.update_info()
        self.checkBox_move_features.setChecked(True)
        self.toggle_feature_interaction()

        napari.utils.notifications.show_info(f"Features Detected")

    def update_info(self):
        self.label_info.setText(
            f"""Moving {self.detected_features.features[0].name} to {self.detected_features.features[1].name}
        \n{self.detected_features.features[0].name}: {self.detected_features.features[0].feature_px}
        \n{self.detected_features.features[1].name}: {self.detected_features.features[1].feature_px}
        \ndx={self.detected_features.distance.x*1e6:.2f}um, dy={self.detected_features.distance.y*1e6:.2f}um
        User Corrected: {self._USER_CORRECTED}
        """
        )

    def point_moved(self, layer, event):
        dragged = False
        yield

        # on move
        while event.type == "mouse_move":
            dragged = True
            yield

        # on release
        if not dragged:
            return

        # get the data
        data = layer.data

        # get which point was moved
        index: list[int] = list(layer.selected_data)  
        for idx in index:
            
            logging.info(f"point moved: {self.detected_features.features[idx].name} to {data[idx]}") # TODO: fix for logging statistics

            # update the feature
            self.detected_features.features[idx].feature_px = Point(
                x=data[idx][1], y=data[idx][0]
            )

        # recalculate the distance
        self.detected_features.distance = self.detected_features.features[0].feature_px._distance_to(
            self.detected_features.features[1].feature_px
        )
        self.detected_features.distance = self.detected_features.distance._to_metres(pixel_size = self.detected_features.pixelsize) # TODO: get from metadata)

        self._USER_CORRECTED = True
        self.update_info()

    def _get_detected_features(self):
        return self.detected_features


def main():
    from fibsem import utils

    microscope, settings = utils.setup_session()

    viewer = napari.Viewer(ndisplay=2)

    # load image
    image = FibsemImage.load(os.path.join(os.path.dirname(detection.__file__), "test_image.tif"))
    
    # TODO: START_HERE
    # add load detected feature
    # convert / add fibsem image and binary masks for detections

    det_widget_ui = FibsemDetectionWidgetUI(
        microscope=microscope, 
        settings=settings, 
        viewer=viewer, 
        image = image)
    viewer.window.add_dock_widget(
        det_widget_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
