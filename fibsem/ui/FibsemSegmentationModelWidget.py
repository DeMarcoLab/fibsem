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
from fibsem.ui.qtdesigner_files import FibsemSegmentationModelWidget
from fibsem.segmentation.model import SegmentationModel
import logging

CHECKPOINT_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "model4.pt")
SEGMENT_ANYTHING_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "sam_vit_h_4b8939.pth")
CLASS_COLORS = {0: "black", 1: "red", 2: "green", 3: "cyan", 4: "yellow", 5: "magenta", 6: "blue"}

class FibsemSegmentationModelWidget(FibsemSegmentationModelWidget.Ui_Form, QtWidgets.QDialog):
    continue_signal = pyqtSignal(DetectedFeatures)

    def __init__(
        self,
        model: SegmentationModel = None,
        parent=None,
    ):
        super(FibsemSegmentationModelWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.model = model
        self.setup_connections()

    def setup_connections(self):

        # model
        self.pushButton_load_model.clicked.connect(self.load_model)
        self.lineEdit_encoder.setText("resnet34")
        self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
        self.spinBox_num_classes.setValue(3)
        self.comboBox_model_type.addItems(["SegmentationModel", "SegmentAnythingModel"])
        self.comboBox_model_type.currentIndexChanged.connect(self.update_model_type)

    def update_model_type(self):

        model_type = self.comboBox_model_type.currentText()

        if model_type == "SegmentationModel":
            self.spinBox_num_classes.setValue(3)
            self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
            self.lineEdit_encoder.setText("resnet34")
            self.lineEdit_encoder.setEnabled(True)
        elif model_type == "SegmentAnythingModel":
            self.spinBox_num_classes.setValue(7)
            self.lineEdit_checkpoint.setText(SEGMENT_ANYTHING_PATH)
            self.lineEdit_encoder.setText("default")
            self.lineEdit_encoder.setEnabled(False)


    def load_model(self) -> SegmentationModel:

        model_type = self.comboBox_model_type.currentText()
        encoder = self.lineEdit_encoder.text()
        checkpoint = self.lineEdit_checkpoint.text()
        num_classes = self.spinBox_num_classes.value()

        if model_type == "SegmentationModel":

            self.model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)

        if model_type == "SegmentAnythingModel":
            self.model = load_sam_model(encoder=encoder, checkpoint=checkpoint)

        print(f"Loaded: ", self.model)

        # TODO abstract this properly
        self.model_type = model_type
        self.model.checkpoint = checkpoint
        self.model.num_classes = num_classes

        return self.model



# TODO: abstract this so can use same interface as fibsem?
def load_sam_model(encoder: str, checkpoint: str):
    """Load the SAM predictor model from a checkpoint file."""
    from segment_anything import sam_model_registry, SamPredictor
    import torch

    sam = sam_model_registry[encoder](checkpoint=checkpoint)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    sam.to(device=device)

    model = SamPredictor(sam)
    return model


def main():

    viewer  = napari.Viewer()
    widget = FibsemSegmentationModelWidget()
    viewer.window.add_dock_widget(widget, area="right", name="Fibsem Segmentation Model")

    napari.run()
    


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
# - abstract segmentation model widget

# TODO:
# - convert detected features / detection to take in Union[FibsemImage, np.ndarray]
# - edittable mask -> rerun detection 
# - need to ensure feature det is only enabled if seg is enabled
# - need seg to be enabled if feature det is enabled same for eval