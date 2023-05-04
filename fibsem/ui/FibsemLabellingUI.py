import logging
import os
import sys
import glob

# import fibsem
import napari
import napari.utils.notifications
import numpy as np
import tifffile as tff
import zarr
from PIL import Image
from fibsem.ui.qtdesigner_files import FibsemLabellingUI
from PyQt5 import QtWidgets
import dask.array as da
from fibsem.segmentation.config import CLASS_COLORS
from fibsem.segmentation import utils as seg_utils

from fibsem.ui.FibsemSegmentationModelWidget import FibsemSegmentationModelWidget

# BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")


class FibsemLabellingUI(FibsemLabellingUI.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        viewer: napari.Viewer,
        parent=None,
    ):
        super(FibsemLabellingUI, self).__init__(parent=parent)
        self.setupUi(self)
        self.viewer = viewer
        self.idx = 0
        self.model = None
        self._model_assist = False
        self.DATA_LOADED = False


        self.setup_connections()

    def setup_connections(self):
        print("setup connections")

        self.pushButton_load_data.clicked.connect(self.load_data)
        self.pushButton_next.clicked.connect(self.next_image)
        self.pushButton_previous.clicked.connect(self.previous_image)

        self.model_widget = FibsemSegmentationModelWidget()
        self.tabWidget.addTab(self.model_widget, "Model")

        self.model_widget.pushButton_load_model.clicked.connect(self.load_model)

        self.checkBox_model_assist.stateChanged.connect(self.toggle_model_assist)

    def toggle_model_assist(self):
        self._model_assist = self.checkBox_model_assist.isChecked()

        if self.model is None:
            self.label_model_info.setText(f"Please load a model.")
        if self.DATA_LOADED:
            self.update_image()

    def load_model(self):
        self.model = self.model_widget.model
        self.label_model_info.setText(
            f"Model: {os.path.basename(self.model.checkpoint)}"
        )

        if self.DATA_LOADED:
            self.update_image()

    def load_data(self):

        # read raw data
        raw_path = self.lineEdit_raw_data.text()
        self.save_path = self.lineEdit_save_path.text()
        self.n_classes = self.spinBox_num_classes.value()

        # get filenames
        self.filenames = sorted(glob.glob(os.path.join(raw_path, "*.tif*")))

        # create required directories
        os.makedirs(os.path.join(self.save_path, "labels"), exist_ok=True)

        self.DATA_LOADED = True

        self.update_image()

    def next_image(self):
        self.save_image()

        # advance index
        self.idx += 1
        self.idx = np.clip(self.idx, 0, len(self.filenames) - 1)

        self.update_image()

    def previous_image(self):
        self.save_image()

        # decrement index
        self.idx -= 1
        self.idx = np.clip(self.idx, 0, len(self.filenames) - 1)

        self.update_image()

    def save_image(self):
        # save current image
        bname = os.path.basename(self.fname).split(".")[0]

        # only resave the labels...
        label = self._label_layer.data.astype(np.uint8)

        im = Image.fromarray(label)
        im.save(os.path.join(self.save_path, "labels", f"{bname}.tif"))  # or 'test.tif'

        rgb = seg_utils.decode_segmap(label, self.n_classes)
        rgb = Image.fromarray(rgb)
        rgb.save(os.path.join(self.save_path, "labels", f"{bname}.png"))  
    
    def update_image(self):
        # update progress text
        self.label_progress.setText(f"{self.idx + 1}/{len(self.filenames)}")

        # update / clear viewer
        self.viewer.layers.clear()
        self.fname = self.filenames[self.idx]
        self.img = tff.imread(self.fname)
        self._image_layer = self.viewer.add_image(
            self.img,
            name="Image",
            opacity=0.7,
            blending="additive",
        )

        label_image = self.get_label_image()
        self._label_layer = self.viewer.add_labels(
            label_image,
            name="Mask",
            opacity=0.7,
            blending="additive",
            color=CLASS_COLORS,
        )

        # disable buttons
        IS_NOT_FIRST_INDEX = bool(self.idx != 0)
        IS_NOT_LAST_INDEX = bool(self.idx != int(len(self.filenames) - 1))
        self.pushButton_previous.setEnabled(IS_NOT_FIRST_INDEX)
        self.pushButton_next.setEnabled(IS_NOT_LAST_INDEX)

    def get_label_image(self) -> np.ndarray:

        if os.path.basename(self.fname) in os.listdir(
            os.path.join(self.save_path, "labels")
        ):
            label_fname = os.path.join(
                self.save_path, "labels", os.path.basename(self.fname)
            )
            label_image = tff.imread(label_fname)
            label_image = np.array(label_image, dtype=np.uint8)

        elif self._model_assist and self.model is not None:
            label_image = self.model.inference(self.img, rgb=False)[0]
        else:
            label_image = np.zeros_like(self.img)

        return label_image

    def closeEvent(self, event):
        # try to save the current image on close
        try:
            self.save_image()
        except:
            pass
        event.accept()

    # TODO: remove use of PIl, use tf to save
    # BUG: no way to save the last image in the dataset? except go back?

    # TODO: go to index
    # TODO: hotkeys
    #

# TODO: add instructions
# TODO: add hotkeys
# TODO: add trianing


def main():
    viewer = napari.Viewer(ndisplay=2)
    fibsem_labelling_ui = FibsemLabellingUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_labelling_ui, area="right", 
        add_vertical_stretch=True,
        name="Fibsem Labelling")
    napari.run()


if __name__ == "__main__":
    main()
