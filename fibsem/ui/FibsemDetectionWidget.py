import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem.microscope import FibsemMicroscope
from fibsem import constants, acquire
from fibsem.structures import (
    MicroscopeSettings,
    Point,
    BeamType,
    ImageSettings,
    FibsemImage,
)
from fibsem.ui import utils as ui_utils
from fibsem.ui.qtdesigner_files import FibsemDetectionWidget
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
import numpy as np
from pathlib import Path

from fibsem.detection import detection
from fibsem.detection.detection import DetectedFeatures

import tifffile as tff
import os

from fibsem.detection import detection
from copy import deepcopy
import tifffile as tff
import os


class FibsemDetectionWidgetUI(FibsemDetectionWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        image_widget: FibsemImageSettingsWidget = None,
        detected_features: DetectedFeatures = None,
        parent=None,
    ):
        super(FibsemDetectionWidgetUI, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget

        self.setup_connections()

        self._USER_CORRECTED = False

        # set detected features
        if detected_features is not None:
            self.set_detected_features(detected_features)
        else:
            self.test_function()

    def setup_connections(self):

        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press continue when finished."""
        )
        self.pushButton_continue.clicked.connect(self.continue_button_clicked)

        self.pushButton_run_feature_detection.clicked.connect(
            self.run_feature_detection
        )

        self.pushButton_test_function.clicked.connect(self.test_function)
        self.pushButton_load_model.clicked.connect(self.load_model)

        self.comboBox_feature_1.addItems(
            [feature.name for feature in detection.__FEATURES__]
        )
        self.comboBox_feature_2.addItems(
            [feature.name for feature in detection.__FEATURES__]
        )

        self.comboBox_beam_type.addItems([beam_type.name for beam_type in BeamType])

        from fibsem.segmentation import model as fibsem_model
        import os

        self.lineEdit_encoder.setText("resnet34")
        self.lineEdit_checkpoint.setText(
            os.path.join(os.path.dirname(fibsem_model.__file__), "models", "model4.pt")
        )
        self.spinBox_num_classes.setValue(3)

    def run_feature_detection(self):

        print("running feature detection...")

        # det = self.load_feature_detection()

        image = self.load_test_image()

        features = (
            detection.__FEATURES__[self.comboBox_feature_1.currentIndex()](),
            detection.__FEATURES__[self.comboBox_feature_2.currentIndex()](),
        )

        # detect features
        pixelsize = 10e-9
        det = detection.locate_shift_between_features_v2(
            deepcopy(image.data), self.model, features=features, pixelsize=pixelsize
        )

        beam_type = BeamType[self.comboBox_beam_type.currentText()]

        self.set_detected_features(det, beam_type)

    def load_test_image(self):
        # load image
        image = tff.imread(
            os.path.join(os.path.dirname(detection.__file__), "test_image.tif")
        )
        image = FibsemImage(image, None)

        return image

    def load_model(self):
        from fibsem.segmentation.model import load_model

        print("loading model...")
        checkpoint = self.lineEdit_checkpoint.text()
        encoder = self.lineEdit_encoder.text()
        num_classes = self.spinBox_num_classes.value()
        self.model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)
        print("model loaded")

    def load_feature_detection(self):
        return

    def test_function(self):

        print("test function....")

    def continue_button_clicked(self):
        print("continue button clicked")

    def set_detected_features(
        self, det_features: DetectedFeatures, beam_type: BeamType = None
    ):

        self.detected_features = det_features

        # update combo box
        self.comboBox_feature_1.setCurrentText(self.detected_features.features[0].name)
        self.comboBox_feature_2.setCurrentText(self.detected_features.features[1].name)

        # TODO: read the image metadata properly

        # add image to viewer (Should be handled by the image widget)
        try:
            self.viewer.layers["image"].data = self.detected_features.image
        except:
            self.viewer.add_image(
                self.detected_features.image, name="image", opacity=0.7
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
            edge_width=1,
            # edge_width_is_relative=False,
            edge_color="transparent",
            face_color=[
                feature.color for feature in self.detected_features.features
            ],
        )

        # translate the image and mask layers if the beam type is ion
        if beam_type is BeamType.ION:
            translation = [0, self.detected_features.image.shape[1]]
        else:
            translation = [0, 0]

        self.viewer.layers["image"].translate = translation
        self.viewer.layers["mask"].translate = translation
        self.viewer.layers["features"].translate = translation

        # set points layer to select mode and active
        self.viewer.layers["features"].mode = "select"
        self.viewer.layers.selection.active = self.viewer.layers["features"]

        # when the point is moved update the feature
        self.viewer.layers["features"].mouse_drag_callbacks.append(self.point_moved)

        self.update_info()

    def update_info(self):
        self.label_info.setText(
            f""" Moving {self.detected_features.features[0].name} to {self.detected_features.features[1].name}
        \n{self.detected_features.features[0].name} is at {self.detected_features.features[0].feature_px}
        \n{self.detected_features.features[1].name} is at {self.detected_features.features[1].feature_px}
        \n\nx distance: {self.detected_features.distance.x*1e6:.2f}um 
        \ny distance: {self.detected_features.distance.y*1e6:.2f}um
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

        # update the feature
        self.detected_features.features[0].feature_px = Point(
            x=data[0][1], y=data[0][0]
        )
        self.detected_features.features[1].feature_px = Point(
            x=data[1][1], y=data[1][0]
        )

        # update the point
        print("point moved: ", self.detected_features.features[0].feature_px)
        print("point moved: ", self.detected_features.features[1].feature_px)

        point_diff_px = self.detected_features.features[0].feature_px._distance_to(
            self.detected_features.features[1].feature_px
        )
        self.detected_features.distance = point_diff_px._to_metres(
            self.detected_features.pixelsize
        )

        self._USER_CORRECTED = True
        self.update_info()

    def _get_detected_features(self):
        return self.detected_features


def main():
    viewer = napari.Viewer(ndisplay=2)
    det_widget_ui = FibsemDetectionWidgetUI(viewer=viewer)
    viewer.window.add_dock_widget(
        det_widget_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
