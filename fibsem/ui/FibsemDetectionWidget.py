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
from fibsem.ui.utils import message_box_ui
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    BeamType,
    FibsemImage,
    Point,
)
from fibsem.ui import _stylesheets
from PyQt5.QtCore import pyqtSignal
from fibsem.ui.qtdesigner_files import FibsemDetectionWidget
import logging

CHECKPOINT_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "model4.pt")
CLASS_COLORS = {0: "black", 1: "red", 2: "green", 3: "cyan", 4: "yellow", 5: "magenta", 6: "blue"}

class FibsemDetectionWidgetUI(FibsemDetectionWidget.Ui_Form, QtWidgets.QDialog):
    continue_signal = pyqtSignal(DetectedFeatures)

    def __init__(
        self,
        viewer: napari.Viewer,
        image: FibsemImage,
        detected_features: DetectedFeatures = None,
        model: fibsem_model.SegmentationModel = None,
        _SEG_MODE: bool = True,
        _DET_MODE: bool = True,
        _EVAL_MODE: bool = False,
        parent=None,
    ):
        super(FibsemDetectionWidgetUI, self).__init__(parent=parent)
        self.setupUi(self)

        self.viewer = viewer
        self.image = image
        self.model = model
        self.image_paths = None

        self._EVAL_MODE = _EVAL_MODE
        self._USER_CORRECTED = False
        self._USE_ALL_FEATURES = False

        self.USE_SEGMENTATION = True
        self.USE_FEATURE_DETECTION = _DET_MODE
        self.USE_EVALUATION = _EVAL_MODE

        self._image_layer = None
        self._mask_layer = None
        self._features_layer = None

        self.checkBox_use_segmentation.setChecked(self.USE_SEGMENTATION)
        self.checkBox_use_segmentation.setVisible(False)
        self.checkBox_use_feature_detection.setChecked(self.USE_FEATURE_DETECTION)
        self.checkBox_use_evaluation.setChecked(self.USE_EVALUATION)
        


        self.setup_connections()

        # set detected features
        if detected_features is not None:
            self.set_detected_features(detected_features)

        # message_box_ui(title="Feature Detection Validation",text="Ensure that the features detected are located accurately, if not, please manually correct the locations by dragging. Once locations are valid, click continue",buttons=QtWidgets.QMessageBox.Ok)

    def setup_connections(self):
        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press continue when finished."""
        )
        self.pushButton_continue.clicked.connect(self.continue_button_clicked)

        self.pushButton_run_feature_detection.clicked.connect(
            self.run_feature_detection
        )
        self.pushButton_run_feature_detection.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        # ui
        # self.checkBox_use_segmentation.stateChanged.connect(self._toggle_ui)
        self.checkBox_use_feature_detection.stateChanged.connect(self._toggle_ui)
        self.checkBox_use_evaluation.stateChanged.connect(self._toggle_ui)
        self.checkBox_move_features.stateChanged.connect(self._toggle_ui)
        self.checkBox_move_features.setEnabled(False)  # disabled until features are loaded
        self.checkBox_all_features.stateChanged.connect(self._toggle_ui)
        self.checkBox_show_info.stateChanged.connect(self._toggle_ui)

        # features
        self.comboBox_feature_1.addItems([f.name for f in detection.__FEATURES__])
        self.comboBox_feature_2.addItems([f.name for f in detection.__FEATURES__])

        # model
        self.pushButton_load_model.clicked.connect(self.load_model)
        self.pushButton_load_model.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.lineEdit_encoder.setText("resnet34")
        self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
        self.spinBox_num_classes.setValue(3)

        self.continue_signal.connect(self.do_something)

        self.lineEdit_image_path_folder.setText(f"/home/patrick/github/data/liftout/training/train/images")
        self.label_image_path_num.setText("")

        self.label_header_evalulation.setVisible(self._EVAL_MODE)
        self.lineEdit_image_path_folder.setVisible(self._EVAL_MODE)
        self.pushButton_load_images.setVisible(self._EVAL_MODE)
        self.pushButton_load_images.clicked.connect(self.load_image_folder)
        self.pushButton_load_images.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
        self.pushButton_previous_image.clicked.connect(self.update_image)
        self.pushButton_next_image.clicked.connect(self.update_image)
        self.pushButton_previous_image.setVisible(self._EVAL_MODE)
        self.pushButton_next_image.setVisible(self._EVAL_MODE)
        self.label_image_path_num.setVisible(self._EVAL_MODE)
        self.pushButton_next_image.setVisible(False)
        self.pushButton_previous_image.setVisible(False)
        self.label_image_path_num.setVisible(False)

        # TODO: add button to go to idx

    def load_image_folder(self):
        folder = self.lineEdit_image_path_folder.text()
        self.image_paths = sorted(list(Path(folder).glob("*.tif")))
        self.idx = 0
        self.update_image()
        self.pushButton_next_image.setVisible(True)
        self.pushButton_previous_image.setVisible(True)
        self.label_image_path_num.setVisible(True)
    
    def save_data(self):

        
        # get the updated mask
        self.detected_features.mask = self._mask_layer.data.astype(np.uint8) # type: ignore
        
        # save current data
        # fname = os.path.basename(self.image_paths[self.idx])
        det_utils.save_data(det = self.detected_features, corrected=self._USER_CORRECTED, fname=None)


    def update_image(self):

        if self.image_paths is None:
            return

        # save current data
        self.save_data()

        # update image view
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
        # print(msg)
        

    def _toggle_ui(self):

        # self.USE_SEGMENTATION = self.checkBox_use_segmentation.isChecked()

        # if self.USE_SEGMENTATION is False:
            # self.checkBox_use_feature_detection.setChecked(False)

        self.USE_FEATURE_DETECTION = self.checkBox_use_feature_detection.isChecked()
        self.USE_EVALUATION = self.checkBox_use_evaluation.isChecked()

        # # hide segmentation if not used
        # self.label_header_model.setVisible(self.USE_SEGMENTATION)
        # self.label_encoder.setVisible(self.USE_SEGMENTATION)
        # self.lineEdit_encoder.setVisible(self.USE_SEGMENTATION)
        # self.label_checkpoint.setVisible(self.USE_SEGMENTATION)
        # self.lineEdit_checkpoint.setVisible(self.USE_SEGMENTATION)
        # self.label_num_classes.setVisible(self.USE_SEGMENTATION)
        # self.spinBox_num_classes.setVisible(self.USE_SEGMENTATION)
        # self.pushButton_load_model.setVisible(self.USE_SEGMENTATION)

        # hide feature detection if not used
        self.label_header_features.setVisible(self.USE_FEATURE_DETECTION)
        self.label_feature.setVisible(self.USE_FEATURE_DETECTION)
        self.comboBox_feature_1.setVisible(self.USE_FEATURE_DETECTION)
        self.comboBox_feature_2.setVisible(self.USE_FEATURE_DETECTION)
        self.checkBox_all_features.setVisible(self.USE_FEATURE_DETECTION)
        self.checkBox_move_features.setVisible(self.USE_FEATURE_DETECTION)
        self.checkBox_show_info.setVisible(self.USE_FEATURE_DETECTION)
        # self.pushButton_run_feature_detection.setVisible(self.USE_FEATURE_DETECTION)


        # hide evaluation if not used
        self.label_header_evalulation.setVisible(self.USE_EVALUATION)
        self.label_dataset.setVisible(self.USE_EVALUATION)
        self.lineEdit_image_path_folder.setVisible(self.USE_EVALUATION)
        self.pushButton_load_images.setVisible(self.USE_EVALUATION)
        self.pushButton_previous_image.setVisible(self.USE_EVALUATION)
        self.pushButton_next_image.setVisible(self.USE_EVALUATION)
        self.label_image_path_num.setVisible(self.USE_EVALUATION)

        if self.USE_FEATURE_DETECTION:
            self.pushButton_run_feature_detection.setText("Run Feature Detection")
        else:
            self.pushButton_run_feature_detection.setText("Run Segmentation")

    

        # features
        if not self.USE_FEATURE_DETECTION:
            if "features" in self.viewer.layers:
                self.viewer.layers.remove(self._features_layer)
                self._features_layer = None

            return
        
        self._USE_ALL_FEATURES = self.checkBox_all_features.isChecked()
        self.label_feature.setVisible(not self._USE_ALL_FEATURES)
        self.comboBox_feature_1.setVisible(not self._USE_ALL_FEATURES)
        self.comboBox_feature_2.setVisible(not self._USE_ALL_FEATURES)
        
        # info
        self.label_info.setVisible(self.checkBox_show_info.isChecked())
        
        if "features" in self.viewer.layers:
            self.checkBox_move_features.setEnabled(True)
            
            if self.checkBox_move_features.isChecked():
                self.viewer.layers.selection.active = self._features_layer
            else:
                self.viewer.layers.selection.active = self._image_layer

    def run_feature_detection(self):

        self._USER_CORRECTED = False # reset user corrected flag

        if self.model is None:
            self.load_model()

        if self._USE_ALL_FEATURES or self.USE_FEATURE_DETECTION is False:
            features = [f() for f in detection.__FEATURES__]
        else:
            features = (
                detection.__FEATURES__[self.comboBox_feature_1.currentIndex()](),
                detection.__FEATURES__[self.comboBox_feature_2.currentIndex()](),
            )

        # detect features
        pixelsize = 25e-9 # TODO: get from metadata
        det = detection.detect_features(

            deepcopy(self.image.data), self.model, 
            features=features, 
            pixelsize=pixelsize,
            filter=True, point=None
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
        
        # save current data
        self.save_data()

        # emit signal
        self.continue_signal.emit(self.detected_features)
        print("continue signal emitted")

        

        if not self._EVAL_MODE:
            self.close()
            self.viewer.close()

    def set_detected_features(self, det_features: DetectedFeatures):
        self.detected_features = det_features

        self.update_features_ui()

    def update_features_ui(self):
        
        # update combo box
        if not self._USE_ALL_FEATURES:
            self.comboBox_feature_1.setCurrentText(self.detected_features.features[0].name)
            self.comboBox_feature_2.setCurrentText(self.detected_features.features[1].name)

        try:
            self._image_layer.data = self.detected_features.image
        except:
            self._image_layer = self.viewer.add_image(
                self.detected_features.image, name="image", opacity=0.7, blending="additive",
            )

        # add mask to viewer
        try:
            self._mask_layer.data = self.detected_features.mask
        except:
            self._mask_layer = self.viewer.add_labels(self.detected_features.mask, 
                                                      name="mask", 
                                                      opacity=0.7,
                                                      blending="additive", 
                                                      color=CLASS_COLORS)

        # if the features layer already exists, remove the layer
        if "features" in self.viewer.layers:
            self.viewer.layers.remove(self._features_layer)
            self._features_layer = None

        if not self.USE_FEATURE_DETECTION:
            napari.utils.notifications.show_info(f"Segmentation finished.")
            return

        # add points to viewer
        data = []
        for feature in self.detected_features.features:
            x, y = feature.px
            data.append([y, x])


        text = {
            "string": [feature.name for feature in self.detected_features.features],
            "color": "white",
            "translation": np.array([-30, 0]),
        }

        self._features_layer = self.viewer.add_points(
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
        self._features_layer.mode = "select"

        # when the point is moved update the feature
        self._features_layer.events.data.connect(self.update_point)

        self.update_info()
        self.checkBox_move_features.setChecked(True)
        self._toggle_ui()

        napari.utils.notifications.show_info(f"Features Detected")

    def update_info(self):
        
        if len(self.detected_features.features) != 2:
            self.label_info.setText("Please select two features for info.")
            return
        
        self.label_info.setText(
            f"""Moving {self.detected_features.features[0].name} to {self.detected_features.features[1].name}
        \n{self.detected_features.features[0].name}: {self.detected_features.features[0].px}
        \n{self.detected_features.features[1].name}: {self.detected_features.features[1].px}
        \ndx={self.detected_features.distance.x*1e6:.2f}um, dy={self.detected_features.distance.y*1e6:.2f}um
        User Corrected: {self._USER_CORRECTED}
        """
        )

    def update_point(self, event):
        logging.info(f"{event.source.name} changed its data!")

        layer = self.viewer.layers[f"{event.source.name}"]  # type: ignore

        # get the data
        data = layer.data

        # get which point was moved
        index: list[int] = list(layer.selected_data)  
                
        if len(data) != len(self.detected_features.features):
            # loop backwards to remove the features
            for idx in index[::-1]:
                logging.info(f"point deleted: {self.detected_features.features[idx].name}")
                self.detected_features.features.pop(idx)

        else: 
            for idx in index:
                
                logging.info(f"point moved: {self.detected_features.features[idx].name} to {data[idx]}") # TODO: fix for logging statistics
                
                # update the feature
                self.detected_features.features[idx].px = Point(
                    x=data[idx][1], y=data[idx][0]
                )

        self._USER_CORRECTED = True
        self.update_info()



    def _get_detected_features(self):
        return self.detected_features


from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.detection.detection import Feature, DetectedFeatures
from fibsem import acquire
def detection_ui(image: FibsemImage, model: fibsem_model.SegmentationModel, features: list[Feature], validate: bool = True) -> DetectedFeatures:
    print("GOT TO DETECTION UI")
    pixelsize = image.metadata.pixel_size.x if image.metadata is not None else 25e-9

    # detect features
    det = detection.detect_features(
        deepcopy(image.data), model, features=features, pixelsize=pixelsize
    )

    if validate:
        viewer = napari.Viewer(ndisplay=2)
        det_widget_ui = FibsemDetectionWidgetUI(
            viewer=viewer, 
            image = image, 
            _EVAL_MODE=False,)
        
        det_widget_ui.set_detected_features(det)

        viewer.window.add_dock_widget(
            det_widget_ui, area="right", add_vertical_stretch=False, name="Fibsem Feature Detection"
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
    det = detection_ui(image, model, features, validate=validate)

    return det



def main():

    validate = True
    _load_image = True

    from fibsem import utils
    microscope, settings = utils.setup_session(manufacturer="Demo")

    det = detection_movement(microscope, settings, validate=validate, _load_image=_load_image)

    detection.move_based_on_detection(
    microscope,
    settings,
    det=det,
    beam_type=settings.image.beam_type,
)


if __name__ == "__main__":
    main()


# DONE
# - convert to use binary masks instead of rgb - DOne
# - add mask, rgb to detected features + save to file  # DONE
# - convert mask layer to label not image # DONE
# - save detected features to file on prev / save image # DONE
# - add n detections, not just two.. if no features are passed... use all?
# - add toggles for seg / feature detection / eval
# - maybe integrate as labelling ui? -> assisted labelling
# - toggle show info checkbox

# TODO:
# - convert detected features / detection to take in Union[FibsemImage, np.ndarray]
# - edittable mask -> rerun detection 
# - abstract segmentation model widget
# - need to ensure feature det is only enabled if seg is enabled
# - need seg to be enabled if feature det is enabled same for eval