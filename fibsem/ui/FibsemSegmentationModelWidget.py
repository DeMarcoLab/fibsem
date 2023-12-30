import os
from copy import deepcopy
from pathlib import Path

import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation import model as fibsem_model
from fibsem.segmentation.model import load_model

from PyQt5.QtCore import pyqtSignal
from fibsem.ui.qtdesigner_files import FibsemSegmentationModelWidget
from fibsem.segmentation.model import SegmentationModel
import logging
import torch

CHECKPOINT_PATH = "autolamella-mega-latest.pt"
SEGMENT_ANYTHING_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "sam_vit_h_4b8939.pth")
MOBILE_SAM_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "weight", "mobile_sam.pt")

SEGMENT_ANYTHING_AVAIABLE = False
MOBILE_SAM_AVAIABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SEGMENT_ANYTHING_AVAIABLE = True
except ImportError:
    pass
try:
    from mobile_sam import sam_model_registry, SamPredictor
    MOBILE_SAM_AVAIABLE = True
except ImportError:
    pass

AVAILABLE_MODELS = ["SegmentationModel"]
if SEGMENT_ANYTHING_AVAIABLE:
    AVAILABLE_MODELS.append("SegmentAnythingModel")
if MOBILE_SAM_AVAIABLE:
    AVAILABLE_MODELS.append("MobileSAMModel")

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
        self.lineEdit_encoder.setText("default")
        self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
        # self.spinBox_num_classes.setValue(3)
        self.comboBox_model_type.addItems(AVAILABLE_MODELS)
        self.comboBox_model_type.currentIndexChanged.connect(self.update_model_type)

        # only for labelling
        self.lineEdit_encoder.setEnabled(False)
        self.spinBox_num_classes.setEnabled(False)

        # set the hover toolTip 
        self.lineEdit_encoder.setToolTip("Please use a pretrained model.")
        self.lineEdit_checkpoint.setToolTip("Please use a pretrained model.")
        self.spinBox_num_classes.setToolTip("Please set using the configuration (.yaml) file.")

    def update_model_type(self):

        model_type = self.comboBox_model_type.currentText()

        if model_type == "SegmentationModel":
            # self.spinBox_num_classes.setValue(3)
            self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
            # self.lineEdit_encoder.setText("resnet34")
            # self.lineEdit_encoder.setEnabled(True)
        elif model_type == "SegmentAnythingModel":
            # self.spinBox_num_classes.setValue(7)
            self.lineEdit_checkpoint.setText(SEGMENT_ANYTHING_PATH)
            # self.lineEdit_encoder.setText("default")
            # self.lineEdit_encoder.setEnabled(False)
        elif model_type == "MobileSAMModel":
            # self.spinBox_num_classes.setValue(7)
            self.lineEdit_checkpoint.setText(MOBILE_SAM_PATH)



    def load_model(self) -> SegmentationModel:

        model_type = self.comboBox_model_type.currentText()
        encoder = self.lineEdit_encoder.text()
        checkpoint = self.lineEdit_checkpoint.text()
        num_classes = self.spinBox_num_classes.value()

        if model_type == "SegmentationModel":

            self.model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)

        if model_type == "SegmentAnythingModel":
            self.model = load_sam_model(encoder="sam", checkpoint=checkpoint)

        if model_type == "MobileSAMModel":
            self.model = load_sam_model(encoder="mobile_sam", checkpoint=checkpoint)

        print(f"Loaded: ", self.model)

        # TODO abstract this properly
        self.model_type = model_type
        self.model.checkpoint = checkpoint
        self.model.num_classes = num_classes

        return self.model



# TODO: abstract this so can use same interface as fibsem?
def load_sam_model(encoder: str, checkpoint: str):
    """Load the SAM predictor model from a checkpoint file."""

    if encoder == "sam":
        model_type = "default"
    elif encoder == "mobile_sam":
        model_type = "vit_t"

    sam = sam_model_registry[model_type](checkpoint=checkpoint)

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
