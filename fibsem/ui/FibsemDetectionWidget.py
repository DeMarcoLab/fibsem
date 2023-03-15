import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem.microscope import FibsemMicroscope
from fibsem import constants, acquire
from fibsem.structures import MicroscopeSettings, Point, BeamType, ImageSettings, FibsemImage
from fibsem.ui import utils as ui_utils
from fibsem.ui.qtdesigner_files import FibsemDetectionWidget
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
import numpy as np
from pathlib import Path

from fibsem.detection import detection
from fibsem.detection.detection import DetectedFeatures


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

        # set detected features
        if detected_features is not None:
            self.set_detected_features(detected_features)
        else:
            self.test_function()


    def setup_connections(self):

        self.label_instructions.setText("""Drag the detected feature positions to move them. Press continue when finished.""")
        self.pushButton_continue.clicked.connect(self.continue_button_clicked)

        self.pushButton_test_function.clicked.connect(self.test_function)

        self.comboBox_feature.addItems(
            [feature.name for feature in detection.__FEATURES__]
        )
        self.comboBox_feature.currentIndexChanged.connect(self.feature_changed)

    def test_function(self):
        
        from fibsem.detection import detection
        from fibsem.segmentation.model import load_model
        from copy import deepcopy
        import tifffile as tff

        image = tff.imread("/home/patrick/github/fibsem/fibsem/detection/test_image.tif")
        image = FibsemImage(image, None)
        checkpoint = "/home/patrick/github/fibsem/fibsem/segmentation/models/model4.pt"
        encoder = "resnet34"
        num_classes = 3
        cuda = True
        model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)

        # detect features
        pixelsize = 10e-9
        features = (detection.NeedleTip(), detection.LamellaRightEdge())
        det = detection.locate_shift_between_features_v2(deepcopy(image.data), model, 
                                                         features=features, pixelsize=pixelsize)

        # calculate features in microscope image coords
        # det.features[0].feature_m = conversions.image_to_microscope_image_coordinates(det.features[0].feature_px, image.data, pixelsize)
        # det.features[1].feature_m = conversions.image_to_microscope_image_coordinates(det.features[1].feature_px, image.data, pixelsize)

        self.set_detected_features(det)

    def feature_changed(self):
        self.current_idx = self.comboBox_feature.currentIndex()
        if self.detected_features is not None:
            print("current feature: ", self.detected_features.features[self.current_idx])

    def continue_button_clicked(self):
        print("continue button clicked")

    def set_detected_features(self, det_features: DetectedFeatures):
        self.detected_features = det_features

        # update combo box
        self.comboBox_feature.clear()
        self.comboBox_feature.addItems(
            [feature.name for feature in self.detected_features.features]
        )


        # TODO: read the image metadata properly


        # add image to viewer (Should be handled by the image widget)
        try:
            self.viewer.layers["image"].data = self.detected_features.image
        except:
            self.viewer.add_image(self.detected_features.image, name="image", opacity=0.7)

        # add mask to viewer
        try:
            self.viewer.layers["mask"].data = self.detected_features.mask
        except:
            self.viewer.add_image(self.detected_features.mask, name="mask", opacity=0.3)

        IB_IMAGE = False
        if IB_IMAGE:
            self.viewer.layers["image"].translate = [0.0, self.detected_features.image.shape[1]]
            self.viewer.layers["mask"].translate = [0.0, self.detected_features.image.shape[1]]  

        # add points to viewer

        data = []
        for feature in self.detected_features.features:
            x, y = feature.feature_px
            data.append([y, x])

        try:
            self.viewer.layers["features"].data = data
        except:
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
            edge_color='transparent',
            face_color=[feature.color for feature in self.detected_features.features]

            )
        if IB_IMAGE:
            self.viewer.layers["features"].translate = [0.0, self.detected_features.image.shape[1]]  

        # set points layer to select mode
        self.viewer.layers["features"].mode = "select"

        # when the point is moved update the feature
        self.viewer.layers["features"].mouse_drag_callbacks.append(self.point_moved)

        self.update_info()


    def update_info(self):
        self.label_info.setText(f""" Moving {self.detected_features.features[0].name} to {self.detected_features.features[1].name}
        \n{self.detected_features.features[0].name} is at {self.detected_features.features[0].feature_px}
        \n{self.detected_features.features[1].name} is at {self.detected_features.features[1].feature_px}
        \n\nx distance: {self.detected_features.distance.x*1e6:.2f}um 
        \ny distance: {self.detected_features.distance.y*1e6:.2f}um
        """)

    def point_moved(self, layer, event):

        dragged = False
        yield
        
        # on move
        while event.type == 'mouse_move':
            dragged = True
            yield

        # on release
        if not dragged:
            return
        
        # get the data
        data = layer.data

        # update the feature
        self.detected_features.features[0].feature_px = Point(x=data[0][1], y=data[0][0])
        self.detected_features.features[1].feature_px = Point(x=data[1][1], y=data[1][0])

        # update the point
        print("point moved: ", self.detected_features.features[0].feature_px)
        print("point moved: ", self.detected_features.features[1].feature_px)

        point_diff_px  = self.detected_features.features[0].feature_px._distance_to(self.detected_features.features[1].feature_px)
        self.detected_features.distance = point_diff_px._to_metres(self.detected_features.pixelsize)

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
