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
from fibsem.ui.FibsemModelTrainingWidget import FibsemModelTrainingWidget

from fibsem.ui.FibsemSegmentationModelWidget import FibsemSegmentationModelWidget
from fibsem.ui.FibsemModelTrainingWidget import FibsemModelTrainingWidget
from fibsem.ui.utils import _get_directory_ui,_get_file_ui
# BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")

from napari.layers import Points
from typing import Any, Generator, Optional


logging.basicConfig(level=logging.INFO)

_DEBUG = True

INSTRUCTIONS = {
    "data": "Select the folder containing the raw data\n",
    "model": "Select the model to use for segmentation\n",
    "ready": "Use the paintbrush to label the image.\n",
    "model_loaded": "The Segmentation Model will assist with labelling images.\n",
    "sam": "Click a point to segment the image.\nConfirm adds the SAM mask to the mask for the selected class.\nPress C to confirm, X to clear, D/F to increment the class\n",
}


# ref: https://github.com/JoOkuma/napari-segment-anything (Apache License 2.0)
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
        self.model_type = None
        self._model_assist = False
        self.DATA_LOADED = False

        self.SAM_IMAGE_SET = False
        self._sam_mask_layer = None
        self._sam_pts_layer = None

        self.setup_connections()

        if _DEBUG:
            self.lineEdit_raw_data.setText(
                "/home/patrick/github/data/liftout/active-learning/dm-embryo/data"
            )
            self.lineEdit_save_path.setText(
                "/home/patrick/github/data/liftout/active-learning/dm-embryo/data"
            )
            self.spinBox_num_classes.setValue(3)

    def setup_connections(self):
        self.pushButton_load_data.clicked.connect(self.load_data)
        self.pushButton_next.clicked.connect(self.next_image)
        self.pushButton_previous.clicked.connect(self.previous_image)

        self.model_widget = FibsemSegmentationModelWidget()
        self.train_widget = FibsemModelTrainingWidget(viewer=self.viewer)

         # save path buttons

        self.rawData_button.clicked.connect(self.select_filepath)
        self.savePath_button.clicked.connect(self.select_filepath)
        self.train_widget.dataPath_button.clicked.connect(self.select_filepath)
        self.train_widget.labelPath_button.clicked.connect(self.select_filepath)
        self.train_widget.outputPath_button.clicked.connect(self.select_filepath)
        self.model_widget.checkpoint_seg_button.clicked.connect(self.select_filepath)



        self.tabWidget.addTab(self.model_widget, "Model")
        self.tabWidget.addTab(self.train_widget, "Train")

        self.model_widget.pushButton_load_model.clicked.connect(self.load_model)

        self.checkBox_model_assist.stateChanged.connect(self.toggle_model_assist)

        self.viewer.bind_key("R", self.next_image)
        self.viewer.bind_key("E", self.previous_image)

        # SAM
        self.pushButton_model_confirm.setVisible(False)
        self.comboBox_model_class_index.setVisible(False)
        self.label_model_class_index.setVisible(False)
        self.pushButton_model_clear.setVisible(False)

        self._update_instructions()

    def select_filepath(self):

        if self.sender() == self.rawData_button:
            path = _get_directory_ui(msg="Select Raw Data Directory")
            if path is not None and path != "":
                self.lineEdit_raw_data.setText(path)
        elif self.sender() == self.savePath_button:
            path = _get_directory_ui(msg="Select Labels Directory")
            if path is not None and path != "":
                self.lineEdit_save_path.setText(path)

        elif self.sender() == self.train_widget.dataPath_button:
            path = _get_directory_ui(msg="Select Data Path Directory")
            if path is not None and path != "":
                self.train_widget.lineEdit_data_path.setText(path)

        elif self.sender() == self.train_widget.labelPath_button:
            path = _get_directory_ui(msg="Select Label Path Directory")
            if path is not None and path != "":
                self.train_widget.lineEdit_label_path.setText(path)

        elif self.sender() == self.train_widget.outputPath_button:
            path = _get_directory_ui(msg="Select Output Path Directory")
            if path is not None and path != "":
                self.train_widget.lineEdit_save_path.setText(path)

        elif self.sender() == self.model_widget.checkpoint_seg_button:
            path = _get_file_ui(msg="Select Checkpoint File")
            if path is not None and path != "":
                self.model_widget.lineEdit_checkpoint.setText(path)

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

    def next_image(self, _: Optional[Any] = None) -> None:
        self.save_image()

        # advance index
        self.idx += 1
        self.idx = np.clip(self.idx, 0, len(self.filenames) - 1)

        self.update_image()

    def previous_image(self, _: Optional[Any] = None) -> None:
        self.save_image()

        # decrement index
        self.idx -= 1
        self.idx = np.clip(self.idx, 0, len(self.filenames) - 1)

        self.update_image()

    def save_image(self):
        # save current image
        bname = os.path.basename(self.fname).split(".")[0]

        # only resave the labels...
        label = self._mask_layer.data.astype(np.uint8)

        im = Image.fromarray(label)
        im.save(os.path.join(self.save_path, "labels", f"{bname}.tif"))  # or 'test.tif'

        rgb = seg_utils.decode_segmap(label, self.n_classes)
        rgb = Image.fromarray(rgb)
        rgb.save(os.path.join(self.save_path, "labels", f"{bname}.png"))

    def update_image(self):
        # update progress text
        self.label_progress.setText(f"{self.idx + 1}/{len(self.filenames)}")

        # update / clear viewer
        self.fname = self.filenames[self.idx]
        self.img = tff.imread(self.fname)
        self.SAM_IMAGE_SET = False

        try:
            self._image_layer.data = self.img
        except:
            self._image_layer = self.viewer.add_image(
                self.img,
                name="Image",
                opacity=0.7,
                blending="additive",
            )

        label_image = self.get_label_image()

        try:
            self._mask_layer.data = label_image
        except:
            self._mask_layer = self.viewer.add_labels(
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

        self._update_instructions()
        self._set_sam_active()
        self.viewer.status = f"Loaded image {self.idx + 1}/{len(self.filenames)}"

    def _update_instructions(self):
        if not self.DATA_LOADED:
            self.label_instructions.setText(INSTRUCTIONS["data"])
            return

        if self.model is None:
            msg = INSTRUCTIONS["ready"] + INSTRUCTIONS["model"]
            self.label_instructions.setText(msg)
            return

        if self.model is not None:
            msg = INSTRUCTIONS["ready"] + INSTRUCTIONS["model_loaded"]
            self.label_instructions.setText(msg)

        if self.model_type == "SegmentAnythingModel":
            msg = INSTRUCTIONS["ready"] + INSTRUCTIONS["sam"]
            self.label_instructions.setText(msg)

    def get_label_image(self) -> np.ndarray:
        if os.path.basename(self.fname) in os.listdir(
            os.path.join(self.save_path, "labels")
        ):
            label_fname = os.path.join(
                self.save_path, "labels", os.path.basename(self.fname)
            )
            label_image = tff.imread(label_fname)
            label_image = np.array(label_image, dtype=np.uint8)

            msg = f"Loaded label image from {label_fname}"

        elif self._model_assist and self.model is not None:
            if self.model_type == "SegmentationModel":
                label_image = self.model.inference(self.img, rgb=False)[0]
            if self.model_type == "SegmentAnythingModel":
                label_image = np.zeros_like(self.img)

            msg = f"Generated label image using {self.model_type}"

        else:
            label_image = np.zeros_like(self.img)

            msg = "No label image found, and no model loaded"

        napari.utils.notifications.show_info(msg)

        return label_image

    ####### MODEL
    def toggle_model_assist(self):
        self._model_assist = self.checkBox_model_assist.isChecked()

        if self.model is None:
            self.label_model_info.setText(f"Please load a model.")
        if self.DATA_LOADED:
            self.update_image()

    def load_model(self):
        self.model = self.model_widget.model
        self.model_type = self.model_widget.model_type
        self.label_model_info.setText(
            f"Model: {os.path.basename(self.model.checkpoint)}"
        )

        # specific layers for SAM model
        if self.model_type == "SegmentAnythingModel":
            self._add_sam_pts_layer()

            self.comboBox_model_class_index.addItems(
                [f"Class {i:02d}" for i in range(self.model.num_classes)]
            )
            self.pushButton_model_confirm.clicked.connect(self._confirm_model_class)
            self.pushButton_model_clear.clicked.connect(self._clear_sam_data)
            self.pushButton_model_confirm.setVisible(True)
            self.comboBox_model_class_index.setVisible(True)
            self.label_model_class_index.setVisible(True)
            self.pushButton_model_clear.setVisible(True)

            self.viewer.bind_key("C", self._confirm_model_class)
            self.viewer.bind_key("X", self._clear_sam_data)
            self.viewer.bind_key("D", self._decrement_sam_class)
            self.viewer.bind_key("F", self._increment_sam_class)

        napari.utils.notifications.show_info(
            f"Loaded {self.model_type}: {os.path.basename(self.model.checkpoint)}"
        )

        if self.DATA_LOADED:
            self.update_image()

    ####### SAM
    def _add_sam_pts_layer(self):
        # add SAM points layer
        self._sam_pts_layer = self.viewer.add_points(name="SAM Points")
        self._sam_pts_layer.events.data.connect(self._update_sam_mask)
        self._sam_pts_layer.mouse_drag_callbacks.append(self._mouse_button_modifier)
        self._sam_pts_layer.mode = "add"

    def _mouse_button_modifier(self, _: Points, event) -> None:
        self._sam_pts_layer.selected_data = []
        if event.button == 1:
            self._sam_pts_layer.current_face_color = "blue"
        else:
            self._sam_pts_layer.current_face_color = "red"

        logging.info(f"Color: {self._sam_pts_layer.current_face_color}")

    def _update_sam_mask(self, _: Optional[Any] = None) -> None:
        points = self._sam_pts_layer.data
        colors = self._sam_pts_layer.face_color

        logging.info(f"POINTS: {len(points)}")

        if len(points) > 0:
            points = np.flip(points, axis=-1)
            colors = self._sam_pts_layer.face_color
            blue = [0, 0, 1, 1]
            labels = np.all(colors == blue, axis=1)
        else:
            points = None
            labels = None

        # get the points from a click
        # convert the current mask to a label image
        # clear the current mask and point

        # this only needs to happen once when the image is loaded
        # double chcek when the image is saved that it is grayscale
        if not self.SAM_IMAGE_SET:
            import cv2

            logging.info("setting image embedding")
            gray_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
            self.model.set_image(gray_img)
            self.SAM_IMAGE_SET = True
            self._logits = None

        mask, _, self._logits = self.model.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            mask_input=self._logits,
            multimask_output=False,
        )

        # add layer for mask
        try:
            self._sam_mask_layer.data = mask[0]
        except:
            self._sam_mask_layer = self.viewer.add_labels(
                mask[0],
                name="SAM Mask",
                opacity=0.7,
                blending="additive",
                color={0: "black", 1: "white"},
            )

        self._set_sam_active()

    def _set_sam_active(self):
        try:
            # set sam points layer active
            self.viewer.layers.selection.active = self._sam_pts_layer
            self._sam_pts_layer.mode = "add"
        except:
            pass

    def _confirm_model_class(self, _: Optional[Any] = None) -> None:
        if self._sam_mask_layer is None:
            return

        # take the current sam mask and add it to the Mask
        sam_mask = self._sam_mask_layer.data

        # get the current class index
        class_index = self.comboBox_model_class_index.currentIndex()

        # set all values of sam mask to class index
        sam_mask[sam_mask > 0] = class_index

        # add to, and update mask layer
        mask = self._mask_layer.data
        mask[sam_mask > 0] = sam_mask[sam_mask > 0]
        self._mask_layer.data = mask

        self._clear_sam_data()

    def _increment_sam_class(self, _: Optional[Any] = None) -> None:
        self._change_sam_class(1)

    def _decrement_sam_class(self, _: Optional[Any] = None) -> None:
        self._change_sam_class(-1)

    def _change_sam_class(self, val: int = 0) -> None:
        # get the current class index
        self._sam_class_index = self.comboBox_model_class_index.currentIndex()

        self._sam_class_index += val

        # clip to 0,number of classes
        n_classes = self.comboBox_model_class_index.count()
        self._sam_class_index = np.clip(self._sam_class_index, 0, n_classes - 1)

        self.comboBox_model_class_index.setCurrentIndex(self._sam_class_index)
        self.viewer.status = f"Current Class: {self._sam_class_index}"

    def _clear_sam_data(self, _: Optional[Any] = None) -> None:
        # clear points
        # TODO: find the proper way to do this, not doing this way causses issues with face colour
        self.viewer.layers.remove(self._sam_pts_layer)
        self._add_sam_pts_layer()

        # clear mask
        if self._sam_mask_layer is not None:
            self._sam_mask_layer.data = np.zeros_like(self._sam_mask_layer.data)

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
        fibsem_labelling_ui,
        area="right",
        add_vertical_stretch=True,
        name="Fibsem Labelling",
    )
    napari.run()


if __name__ == "__main__":
    main()
