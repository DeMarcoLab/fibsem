
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation.model import load_model

from PyQt5.QtCore import pyqtSignal
from fibsem.ui.qtdesigner_files import FibsemSegmentationModelWidget
from fibsem.segmentation.model import SegmentationModel

CHECKPOINT_PATH = "autolamella-mega-latest.pt"
SEGMENT_ANYTHING_AVAIABLE = False
try:
    SEGMENT_ANYTHING_AVAIABLE = True
    from fibsem.segmentation.sam_model import SamModelWrapper
except ImportError:
    pass


AVAILABLE_MODELS = ["SegmentationModel"]

if SEGMENT_ANYTHING_AVAIABLE:
    AVAILABLE_MODELS.append("SegmentAnythingModel")
RECOMMENDED_SAM_CHECKPOINTS = ["facebook/sam-vit-base", "Zigeng/SlimSAM-uniform-50"]

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
        self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
        self.comboBox_model_type.addItems(AVAILABLE_MODELS)
        self.comboBox_model_type.currentIndexChanged.connect(self.update_model_type)

    def update_model_type(self):

        model_type = self.comboBox_model_type.currentText()

        if model_type == "SegmentationModel":
            self.lineEdit_checkpoint.setText(CHECKPOINT_PATH)
            self.lineEdit_checkpoint.setToolTip("Please use a pretrained model.")
            self.checkpoint_seg_button.setEnabled(True)
        elif model_type == "SegmentAnythingModel":

            self.lineEdit_checkpoint.setText(RECOMMENDED_SAM_CHECKPOINTS[0])
            self.lineEdit_checkpoint.setToolTip(f"""Any supported transformer SAM model from HuggingFace can be used. 
                                                \nRecommended: 
                                                \nLarge GPU: {RECOMMENDED_SAM_CHECKPOINTS[0]}
                                                \nSmall GPU / CPU: {RECOMMENDED_SAM_CHECKPOINTS[1]}""")
            self.checkpoint_seg_button.setEnabled(False)

    # thread this, as it can take a long time to download the models
    def load_model(self) -> SegmentationModel:

        model_type = self.comboBox_model_type.currentText()
        checkpoint = self.lineEdit_checkpoint.text()

        if model_type == "SegmentationModel":
            self.model = load_model(checkpoint=checkpoint)

        if model_type == "SegmentAnythingModel":
            self.model = SamModelWrapper(checkpoint=checkpoint)

        print(f"Loaded: {self.model}, {self.model.device}, {self.model.checkpoint}")

        # TODO abstract this properly
        self.model_type = model_type
        self.model.checkpoint = checkpoint

        return self.model

def main():

    viewer  = napari.Viewer()
    widget = FibsemSegmentationModelWidget()
    viewer.window.add_dock_widget(widget, 
                                  area="right", 
                                  name="Fibsem Segmentation Model")

    napari.run()
    


if __name__ == "__main__":
    main()
