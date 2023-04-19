import os
from copy import deepcopy
from pathlib import Path

import napari
import napari.utils.notifications
import numpy as np
import tifffile as tff
from PyQt5 import QtWidgets

from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation import model as fibsem_model
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    BeamType,
    FibsemImage,
    Point,
)
from PyQt5.QtCore import pyqtSignal
from fibsem.ui.qtdesigner_files import FibsemDetectionWidget
import logging

CHECKPOINT_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "model4.pt")

class FibsemDetectionWidgetUI(FibsemDetectionWidget.Ui_Form, QtWidgets.QDialog):
    def __init__(
        self,
        viewer: napari.Viewer,
        image: FibsemImage,
        detected_features: DetectedFeatures = None,
        model: fibsem_model.SegmentationModel = None,
        _eval_mode: bool = False,
        parent=None,
    ):
        super(FibsemDetectionWidgetUI, self).__init__(parent=parent)
        self.setupUi(self)

        self.viewer = viewer
        self.image = image
        self.model = model
        self._eval_mode = _eval_mode

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

        self.checkBox_show_mask.clicked.connect(self._toggle_ui)
        self.checkBox_show_mask.setEnabled(False)
        self.checkBox_show_mask.setChecked(True)
        self.checkBox_move_features.clicked.connect(self._toggle_ui)
        self.checkBox_move_features.setEnabled(False)  # disabled until features are loaded

        # features
        self.comboBox_feature_1.addItems([f.name for f in detection.__FEATURES__])
        self.comboBox_feature_2.addItems([f.name for f in detection.__FEATURES__])
        self.comboBox_beam_type.addItems([beam_type.name for beam_type in BeamType])

        # model
        self.pushButton_load_model.clicked.connect(self.load_model)
        self.lineEdit_encoder.setText("resnet34")
        self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
        self.spinBox_num_classes.setValue(3)

        # # setup continue signal
        # self.continue_signal = pyqtSignal(str)

        self.lineEdit_image_path_folder.setText(f"/home/patrick/github/data/liftout/training/train/images")
        self.label_image_path_num.setText("")

        self.label_header_evalulation.setVisible(self._eval_mode)
        self.lineEdit_image_path_folder.setVisible(self._eval_mode)
        self.pushButton_load_images.setVisible(self._eval_mode)
        self.pushButton_load_images.clicked.connect(self.load_image_folder)
        self.pushButton_previous_image.clicked.connect(self.update_image)
        self.pushButton_next_image.clicked.connect(self.update_image)
        self.pushButton_previous_image.setVisible(self._eval_mode)
        self.pushButton_next_image.setVisible(self._eval_mode)
        self.label_image_path_num.setVisible(self._eval_mode)

    def load_image_folder(self):
        folder = self.lineEdit_image_path_folder.text()
        self.image_paths = sorted(list(Path(folder).glob("*.tif")))
        self.idx = 0
        self.update_image()
    
    def update_image(self):
        if self.sender() == self.pushButton_previous_image:
            self.idx -= 1
        if self.sender() == self.pushButton_next_image:
            self.idx += 1

        self.idx = np.clip(self.idx, 0, len(self.image_paths)-1)
        self.label_image_path_num.setText(f"{self.idx+1}/{len(self.image_paths)}")

        self.image = FibsemImage.load(self.image_paths[self.idx])

        self.run_feature_detection()


    def do_something(self, msg):
        print("do_something called")
        print(msg)
        

    def _toggle_ui(self):
        if "features" in self.viewer.layers:
            self.checkBox_move_features.setEnabled(True)
            
        if self.checkBox_move_features.isChecked():
            self.viewer.layers.selection.active = self.viewer.layers["features"]

        else:
            self.viewer.layers.selection.active = self.viewer.layers["image"]

        if "mask" in self.viewer.layers:
            self.checkBox_show_mask.setEnabled(True)
            self.viewer.layers["mask"].visible = self.checkBox_show_mask.isChecked()

    def run_feature_detection(self):

        self._USER_CORRECTED = False # reset user corrected flag

        if self.model is None:
            self.load_model()

        features = (
            detection.__FEATURES__[self.comboBox_feature_1.currentIndex()](),
            detection.__FEATURES__[self.comboBox_feature_2.currentIndex()](),
        )

        # detect features
        pixelsize = 25e-9 # TODO: get from metadata
        det = detection.locate_shift_between_features_v2(
            deepcopy(self.image.data), self.model, features=features, pixelsize=pixelsize
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

    def continue_button_clicked(self):
        
        # save all images and coordinates for testing
        det_utils.save_data(det = self.detected_features, corrected=self._USER_CORRECTED)

        # emit signal
        # self.continue_signal.emit("continue_signal")
        # print("continue signal emitted")

        if not self._eval_mode:
            self.close()
            self.viewer.close()

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
        self._toggle_ui()

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


from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.detection.detection import Feature, DetectedFeatures
from fibsem import acquire
def detection_ui(image: FibsemImage, model: fibsem_model.SegmentationModel, features: list[Feature], validate: bool = True) -> DetectedFeatures:

    pixelsize = image.metadata.pixel_size.x if image.metadata is not None else 25e-9

    # detect features
    det = detection.locate_shift_between_features_v2(
        deepcopy(image.data), model, features=features, pixelsize=pixelsize
    )

    if validate:
        viewer = napari.Viewer(ndisplay=2)
        det_widget_ui = FibsemDetectionWidgetUI(
            viewer=viewer, 
            image = image, 
            _eval_mode=True)
        
        det_widget_ui.set_detected_features(det)

        viewer.window.add_dock_widget(
            det_widget_ui, area="right", add_vertical_stretch=False
        )
        det_widget_ui.exec_()
        # napari.run()

        det = det_widget_ui.detected_features
    
    return det



def detection_movement(microscope: FibsemMicroscope, settings: MicroscopeSettings, validate: bool = True, _load_image: bool = False) -> DetectedFeatures:

    if _load_image:
        # load image
        image = FibsemImage.load(os.path.join(os.path.dirname(detection.__file__), "test_image.tif"))
    else:
        image = acquire.new_image(microscope, settings.image)

    # TODO: read from config?
    checkpoint = str(CHECKPOINT_PATH)
    encoder="resnet34"
    num_classes = 3
    model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)
 
    features = [detection.NeedleTip(), detection.LamellaCentre()]
    det = detection_ui(image, model, features, validate=validate, _load_image=True)

    return det



def main():

    # TODO: START_HERE
    # convert / add fibsem image and binary masks for detections
    # convert detections to use binary masks instead of rgb

    validate = True
    _load_image = True

    from fibsem import utils
    microscope, settings = utils.setup_session()

    det = detection_movement(microscope, settings, validate=validate, _load_image=_load_image)

    detection.move_based_on_detection(
    microscope,
    settings,
    det=det,
    beam_type=settings.image.beam_type,
)


if __name__ == "__main__":
    main()


# TODO:
# - convert to use binary masks instead of rgb
# - convert detected features / detection to take in Union[FibsemImage, np.ndarray]
# - add mask, rgb to detected features + save to file
# - convert mask layer to label not image
# - save detected features to file on prev / save image
# - edittable mask -> rerun detection
# - abstract segmentation model widget
# - add n detections, not just two.. if no features are passed... use all?
# - toggle show info checkbox
# - maybe integrate as labelling ui? -> assisted labelling