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
from PyQt5.QtCore import pyqtSignal
from fibsem.ui.qtdesigner_files import FibsemEmbeddedDetectionWidget
import logging

CHECKPOINT_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "model4.pt")
CLASS_COLORS = {0: "black", 1: "red", 2: "green", 3: "cyan", 4: "yellow", 5: "magenta", 6: "blue"}

class FibsemEmbeddedDetectionUI(FibsemEmbeddedDetectionWidget.Ui_Form, QtWidgets.QWidget):
    continue_signal = pyqtSignal(DetectedFeatures)

    def __init__(
        self,
        viewer: napari.Viewer,
        det: DetectedFeatures = None,
        model: fibsem_model.SegmentationModel = None,
        parent=None,
    ):
        super(FibsemEmbeddedDetectionUI, self).__init__(parent=parent)
        self.setupUi(self)

        self.viewer = viewer
        self.model = model

        self._USER_CORRECTED = False


        self._image_layer = None
        self._mask_layer = None
        self._features_layer = None

        self.setup_connections()

        # set detected features
        if det is not None:
            self.set_detected_features(det)

        if model is not None:
            self._set_model(model)

    def setup_connections(self):
        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press continue when finished."""
        )
        self.pushButton_continue.clicked.connect(self.confirm_button_clicked)

    def save_data(self):
        
        # get the updated mask
        self.det.mask = self._mask_layer.data.astype(np.uint8) # type: ignore
        
        # save current data
        det_utils.save_data(det = self.det, corrected=self._USER_CORRECTED, fname=None)
    

    def confirm_button_clicked(self):
        
        # save current data
        self.save_data()

        # emit signal
        self.continue_signal.emit(self.det)
        print("continue signal emitted")        

        # remove det layers
        if self._image_layer in self.viewer.layers:
            self.viewer.layers.remove(self._image_layer)
        if self._mask_layer in self.viewer.layers:
            self.viewer.layers.remove(self._mask_layer)
        if self._features_layer in self.viewer.layers:
            self.viewer.layers.remove(self._features_layer)
        
        # reshow all other layers
        for layer in self.viewer.layers:
            layer.visible = True

        # reset camera
        self.viewer.camera.center = self.prev_camera.center
        self.viewer.camera.zoom = self.prev_camera.zoom

    def set_detected_features(self, det_features: DetectedFeatures):
        self.det = det_features
        self._USER_CORRECTED = False

        self.update_features_ui()

    def update_features_ui(self):
        
        # hide all other layers?
        for layer in self.viewer.layers:
            layer.visible = False

        self._image_layer = self.viewer.add_image(
            self.det.image, name="image", opacity=0.7, blending="additive",
        )

        # add mask to viewer
        self._mask_layer = self.viewer.add_labels(self.det.mask, 
                                                    name="mask", 
                                                    opacity=0.7,
                                                    blending="additive", 
                                                    color=CLASS_COLORS)

        # add points to viewer
        data = []
        for feature in self.det.features:
            x, y = feature.px
            data.append([y, x])


        text = {
            "string": [feature.name for feature in self.det.features],
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
            face_color=[feature.color for feature in self.det.features],
            blending="translucent",
        )

        # set points layer to select mode and active
        self._features_layer.mode = "select"

        # when the point is moved update the feature
        self._features_layer.events.data.connect(self.update_point)

        self.update_info()
        
        # set camera
        self.prev_camera = deepcopy(self.viewer.camera)
        self.viewer.camera.center = [
            0.0,
            self._image_layer.data.shape[0] / 2,
            self._image_layer.data.shape[1] / 2,
        ]
        self.viewer.camera.zoom = 0.7

        napari.utils.notifications.show_info(f"Features Detected")

    def update_info(self):
        
        if len(self.det.features) != 2:
            self.label_info.setText("Info only available for 2 features")
            return
        
        self.label_info.setText(
            f"""Moving {self.det.features[0].name} to {self.det.features[1].name}
        \n{self.det.features[0].name}: {self.det.features[0].px}
        \n{self.det.features[1].name}: {self.det.features[1].px}
        \ndx={self.det.distance.x*1e6:.2f}um, dy={self.det.distance.y*1e6:.2f}um
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
                
        if len(data) != len(self.det.features):
            # loop backwards to remove the features
            for idx in index[::-1]:
                logging.info(f"point deleted: {self.det.features[idx].name}")
                self.det.features.pop(idx)

        else: 
            for idx in index:
                
                logging.info(f"point moved: {self.det.features[idx].name} to {data[idx]}") # TODO: fix for logging statistics
                
                # update the feature
                self.det.features[idx].px = Point(
                    x=data[idx][1], y=data[idx][0]
                )

        self._USER_CORRECTED = True
        self.update_info()


    def _set_model(self, model: fibsem_model.SegmentationModel):
        self.model = model
        # update model info
        self.label_model.setText(f"Model: {self.model.model.name}, Checkpont: {self.model.checkpoint}")

    def _get_detected_features(self):
        return self.det


from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.detection.detection import Feature, DetectedFeatures
from fibsem import acquire

def main():
    # load model
    checkpoint = str(CHECKPOINT_PATH)
    encoder="resnet34"
    num_classes = 3
    model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)
    
    # load image
    image = FibsemImage.load(os.path.join(os.path.dirname(detection.__file__), "test_image.tif"))

    pixelsize = image.metadata.pixel_size.x if image.metadata is not None else 25e-9

    # detect features
    features = [detection.NeedleTip(), detection.LamellaCentre()]
    det = detection.detect_features(
        deepcopy(image.data), model, features=features, pixelsize=pixelsize
    )

    viewer = napari.Viewer(ndisplay=2)
    det_widget_ui = FibsemEmbeddedDetectionUI(
        viewer=viewer, 
        model=model,
        )
    
    det_widget_ui.set_detected_features(det)

    viewer.window.add_dock_widget(
        det_widget_ui, area="right", add_vertical_stretch=False, name="Fibsem Feature Detection"
    )
    napari.run()

    det = det_widget_ui.det

    # print(det)


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