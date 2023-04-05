import os
from copy import deepcopy
from pathlib import Path

import napari
import napari.utils.notifications
import numpy as np
import tifffile as tff
from PyQt5 import QtWidgets

from fibsem.detection import detection
from fibsem.detection.detection import DetectedFeatures
from fibsem.microscope import FibsemMicroscope
from fibsem.segmentation import model as fibsem_model
from fibsem.segmentation.model import load_model
from fibsem.structures import (BeamType, FibsemImage, ImageSettings,
                               MicroscopeSettings, Point)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemDetectionWidget


class FibsemDetectionWidgetUI(FibsemDetectionWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        image_widget: FibsemImageSettingsWidget,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
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

    def setup_connections(self):

        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press continue when finished."""
        )
        self.pushButton_continue.clicked.connect(self.continue_button_clicked)

        self.pushButton_run_feature_detection.clicked.connect(
            self.run_feature_detection
        )

        self.pushButton_test_function.setText(f"Move Feature Positions")
        self.pushButton_test_function.clicked.connect(self.toggle_feature_interaction)
        self.pushButton_test_function.setEnabled(False) # disabled until features are loaded 


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
            self.pushButton_test_function.setEnabled(True)

        # change colour of test button between orange and green
        if self.pushButton_test_function.text() == "Move Feature Positions":
            self.pushButton_test_function.setText(f"Disable Feature Interaction")
            self.pushButton_test_function.setStyleSheet("background-color: orange")
            self.viewer.layers.selection.active = self.viewer.layers["features"]

        else:
            self.pushButton_test_function.setText(f"Move Feature Positions")
            self.pushButton_test_function.setStyleSheet("background-color: gray")
            self.viewer.layers.selection.active = self.viewer.layers[self.image_layer_name]

    def run_feature_detection(self):

        print("running feature detection...")

        image = self.load_test_image()

        features = (
            detection.__FEATURES__[self.comboBox_feature_1.currentIndex()](),
            detection.__FEATURES__[self.comboBox_feature_2.currentIndex()](),
        )

        # detect features
        pixelsize = 25e-9
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

        print("loading model...")
        checkpoint = self.lineEdit_checkpoint.text()
        encoder = self.lineEdit_encoder.text()
        num_classes = self.spinBox_num_classes.value()
        self.model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)
        napari.utils.notifications.show_info(f"Model loaded: {os.path.basename(checkpoint)}")

    def load_feature_detection(self):
        return

    def test_function(self):

        print("test function....")

    def continue_button_clicked(self):
        print("continue button clicked")


        # TODO:
        # save image if corrected
        if self._USER_CORRECTED:
            
            if self.image_layer_name == BeamType.ELECTRON.name:
                image = self.image_widget.eb_image
            else:
                image = self.image_widget.ib_image

            # TODO: these should be fibsem images so we have metatdata
            from fibsem import utils
            tff.imsave(f"{utils.current_timestamp()}.tif", image)
        
        # save coordinates for testing




        # move based on detected features
        from fibsem.detection import detection

        # TODO: add x, y limtis
        detection.move_based_on_detection(self.microscope, self.settings, det=self.detected_features, beam_type=BeamType[self.comboBox_beam_type.currentText()])

    def set_detected_features(
        self, det_features: DetectedFeatures, beam_type: BeamType = None
    ):

        self.detected_features = det_features

        self.update_features_ui(beam_type)


    def update_features_ui(self, beam_type: BeamType = None):

        # update combo box
        self.comboBox_feature_1.setCurrentText(self.detected_features.features[0].name)
        self.comboBox_feature_2.setCurrentText(self.detected_features.features[1].name)

        # add image to viewer (Should be handled by the image widget)
        self.image_layer_name = beam_type.name
        self.image_widget.update_viewer(self.detected_features.image, self.image_layer_name)

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
            # edge_width_is_relative=False,
            edge_color="transparent",
            face_color=[
                feature.color for feature in self.detected_features.features
            ],
            blending="translucent",
        )

        # translate the image and mask layers if the beam type is ion
        if beam_type is BeamType.ION:
            translation = [0, self.detected_features.image.shape[1]]
        else:
            translation = [0, 0]

        self.viewer.layers["mask"].translate = translation
        self.viewer.layers["features"].translate = translation

        # set points layer to select mode and active
        self.viewer.layers["features"].mode = "select"

        # when the point is moved update the feature
        self.viewer.layers["features"].mouse_drag_callbacks.append(self.point_moved)

        self.update_info()
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

    from fibsem import utils
    microscope, settings = utils.setup_session()

    viewer = napari.Viewer(ndisplay=2)

    image_widget = FibsemImageSettingsWidget(microscope=microscope, image_settings=settings.image, viewer=viewer)
    det_widget_ui = FibsemDetectionWidgetUI(microscope=microscope, settings=settings, viewer=viewer, image_widget=image_widget)
    viewer.window.add_dock_widget(
        det_widget_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
