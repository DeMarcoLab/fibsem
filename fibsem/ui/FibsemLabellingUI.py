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
from fibsem.segmentation.config import CLASS_COLORS, CLASS_LABELS, convert_color_names_to_rgb
from fibsem.segmentation import utils as seg_utils

from fibsem.ui.FibsemSegmentationModelWidget import FibsemSegmentationModelWidget

from fibsem.ui.utils import _get_directory_ui,_get_file_ui
# BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")

from napari.layers import Points
from typing import Any, Generator, Optional


logging.basicConfig(level=logging.INFO)

_DEBUG = True

INSTRUCTIONS = {
    "data": "Select the folder containing the raw data\n",
    "model": "Select the model to use for segmentation\n",
    "ready": "Use the paintbrush to label the image.\n Press R/E to go to the next/previous image.\n ",
    "model_loaded": "The Segmentation Model will assist with labelling images.\n",
    "sam": "Click a point to segment the image.\nConfirm adds the SAM mask to the mask for the selected class.\nPress C to confirm, X to clear, D/F to increment the class\n",
}

CONFIG = {
    "IMAGES": {
        "FILE_EXT": ".tif",
    },
    "LABELS": {
        "COLOR_MAP": CLASS_COLORS,
        "LABEL_MAP": CLASS_LABELS,
    },
    "SAVE": {
        "SAVE_RGB": True,
    },
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
        self.last_idx = 0
        self.model = None
        self.model_type = None
        self.model_assist = False
        self.DATA_LOADED = False

        self.SAM_IMAGE_SET = False
        self.sam_mask_layer = None
        self.sam_pts_layer = None

        self.setup_connections()

        if _DEBUG:
            self.lineEdit_data_path.setText("/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/train")
            self.lineEdit_labels_path.setText("/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/train/labels/test")

    def setup_connections(self):
        self.pushButton_load_data.clicked.connect(self.load_data)

        self.model_widget = FibsemSegmentationModelWidget()
        
        # save path buttons

        self.pushButton_data_path.clicked.connect(self.select_filepath)
        self.pushButton_labels_path.clicked.connect(self.select_filepath)
        self.model_widget.checkpoint_seg_button.clicked.connect(self.select_filepath)


        self.tabWidget.addTab(self.model_widget, "Model")
        
        # set tab 3 invisible (until fixed)
        self.tabWidget.setTabEnabled(2, False)
        self.tabWidget.setTabVisible(2, False)        

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

        if self.sender() == self.pushButton_data_path:
            path = _get_directory_ui(msg="Select Image Data Directory")
            if path is not None and path != "":
                self.lineEdit_data_path.setText(path)
        elif self.sender() == self.pushButton_labels_path:
            path = _get_directory_ui(msg="Select Labels Directory")
            if path is not None and path != "":
                self.lineEdit_labels_path.setText(path)

        elif self.sender() == self.model_widget.checkpoint_seg_button:
            path = _get_file_ui(msg="Select Checkpoint File")
            if path is not None and path != "":
                self.model_widget.lineEdit_checkpoint.setText(path)

    def load_data(self):
        # read raw data
        data_path = self.lineEdit_data_path.text()
        labels_path = self.lineEdit_labels_path.text()
        
        # create required directories
        os.makedirs(labels_path, exist_ok=True)
        
        # check if save and raw are the same 
        if data_path == labels_path:
            napari.utils.notifications.show_error(f"Save and Raw directories cannot be the same")
            return
            
        # get filenames
        FILE_EXT = CONFIG["IMAGES"]["FILE_EXT"]
        filenames = sorted(glob.glob(os.path.join(data_path, f"*{FILE_EXT}*")))
        if len(filenames) == 0:
            napari.utils.notifications.show_error(f"No images found in {data_path}")
            return

        # assign data
        self.data_path = data_path
        self.labels_path = labels_path
        self.filenames = filenames
        self.DATA_LOADED = True


        # load filenames as single layer
        self.img_layer = self.viewer.open(filenames, name="Image", stack=True)[0]
        self.img_layer.opacity = 0.7
        self.img_layer.blending="additive"

        # label_image = self.get_label_image()
        arr = np.zeros_like(np.asarray(self.img_layer.data[0]))
        self.mask_layer = self.viewer.add_labels(
            arr,
            name="Mask",
            opacity=0.7,
            blending="additive",
            color=CONFIG["LABELS"]["COLOR_MAP"],
        )

        self.viewer.dims.events.current_step.connect(self.update_image)
        self.update_viewer_to_image(0)
        

        # this event just doesn't work?
        # self.mask_layer.events.data.connect(self.save_image) 


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

    def save_image(self, event: Optional[Any] = None) -> None:

        idx = self.last_idx
        fname = self.filenames[idx]
        print(f"IDX: {idx}, Saving image to {os.path.basename(fname)}")
        print(event)

        # save current image
        FILE_EXT = CONFIG["IMAGES"]["FILE_EXT"]
        bname = os.path.basename(fname).removesuffix(FILE_EXT)

        # only resave the labels...
        label = np.asarray(self.mask_layer.data).astype(np.uint8)

        im = Image.fromarray(label) # TODO: replace with tifffile?
        im.save(os.path.join(self.labels_path, f"{bname}{FILE_EXT}"))

        if CONFIG["SAVE"]["SAVE_RGB"]:
            
            colormap = CONFIG["LABELS"]["COLOR_MAP"]
            colormap_rgb = convert_color_names_to_rgb(colormap)

            rgb = seg_utils.decode_segmap_v2(label, colormap_rgb)
            rgb = Image.fromarray(rgb)

            RGB_PATH = os.path.join(self.labels_path, "rgb")
            os.makedirs(RGB_PATH, exist_ok=True)
            rgb.save(os.path.join(RGB_PATH, f"{bname}.png"))

    
    def update_viewer_to_image(self, idx: int):
        self.viewer.dims.set_point(0, idx)

    def update_image(self):
        
        idx = int(self.viewer.dims.point[0])
        print("UPDATING IMAGE")
        print("LAST IDX: ", self.last_idx)
        print("CURRENT IDX: ", idx)

        # save previous image
        if self.last_idx != idx:
            self.save_image()

            # update index
            self.last_idx = idx
        
        # update mask layer
        self.mask_layer.data =  self.get_label_image()

        # update SAM layers
        self.SAM_IMAGE_SET = False
        self.set_sam_active()


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

        idx = int(self.viewer.dims.point[0])
        fname = self.filenames[idx]
        image = np.asarray(self.img_layer.data[idx]) # req for lazy load

        logging.info(f"IDX: {idx}, Loading image from {os.path.basename(fname)}")

        if os.path.basename(fname) in os.listdir(os.path.join(self.labels_path)):
            label_fname = os.path.join(self.labels_path, os.path.basename(fname)
            )
            label_image = tff.imread(label_fname)
            label_image = np.array(label_image, dtype=np.uint8)

            msg = f"Loaded label image from {os.path.basename(label_fname)}"

        elif self.model_assist and self.model is not None:
            if self.model_type == "SegmentationModel":
                label_image = self.model.inference(image, rgb=False)[0]
            if self.model_type == "SegmentAnythingModel":
                label_image = np.zeros_like(image)

            msg = f"Generated label image using {self.model_type}"

        else:
            label_image = np.zeros_like(image)

            msg = "No label image found, and no model loaded"

        napari.utils.notifications.show_info(msg)

        return np.asarray(label_image).astype(np.uint8)

    ####### MODEL
    def toggle_model_assist(self):
        self.model_assist = self.checkBox_model_assist.isChecked()

        if self.model is None:
            self.label_model_info.setText(f"Please load a model.")
        if self.DATA_LOADED:
            self.update_image()

    def load_model(self):
        self.model = self.model_widget.model
        self.model_type = self.model_widget.model_type
        self.label_model_info.setText(f"Model: {os.path.basename(self.model.checkpoint)}")

        # specific layers for SAM model
        if self.model_type == "SegmentAnythingModel":
            self._add_sam_pts_layer()

            N_LABELS = self.model.num_classes
            self.comboBox_model_class_index.addItems(
                [f"{i:02d} - {CONFIG['LABELS']['LABEL_MAP'].get(i, 'Unspecified')}" for i in range(N_LABELS)]
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
        self.sam_pts_layer = self.viewer.add_points(name="SAM Points")
        self.sam_pts_layer.events.data.connect(self._update_sam_mask)
        self.sam_pts_layer.mouse_drag_callbacks.append(self._mouse_button_modifier)
        self.sam_pts_layer.mode = "add"

    def _mouse_button_modifier(self, _: Points, event) -> None:
        self.sam_pts_layer.selected_data = []
        if event.button == 1:
            self.sam_pts_layer.current_face_color = "blue"
        else:
            self.sam_pts_layer.current_face_color = "red"

        logging.info(f"Color: {self.sam_pts_layer.current_face_color}")

    def _update_sam_mask(self, _: Optional[Any] = None) -> None:
        points = self.sam_pts_layer.data
        colors = self.sam_pts_layer.face_color

        logging.info(f"POINTS: {len(points)}")

        if len(points) > 0:
            points = np.flip(points, axis=-1)
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
            # get the current image
            idx = int(self.viewer.dims.point[0])
            image = np.asarray(self.img_layer.data[idx]) # req for lazy load
            
            # convert to grayscale
            gray_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # set embedding
            self.model.set_image(gray_img)
            self.SAM_IMAGE_SET = True
            self.logits = None

        mask, scores, self.logits = self.model.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            mask_input=self.logits,
            multimask_output=False,
        )

        # add layer for mask
        try:
            self.sam_mask_layer.data = mask[0]
        except:
            self.sam_mask_layer = self.viewer.add_labels(
                mask[0],
                name="SAM Mask",
                opacity=0.7,
                blending="additive",
                color={0: "black", 1: "white"},
            )

        self.set_sam_active()

    def set_sam_active(self):
        try:
            # set sam points layer active
            self.viewer.layers.selection.active = self.sam_pts_layer
            self.sam_pts_layer.mode = "add"
        except:
            pass

    def _confirm_model_class(self, _: Optional[Any] = None) -> None:
        if self.sam_mask_layer is None:
            return

        # take the current sam mask and add it to the Mask
        sam_mask = self.sam_mask_layer.data

        # get the current class index
        class_index = self.comboBox_model_class_index.currentIndex()

        # set all values of sam mask to class index
        sam_mask[sam_mask > 0] = class_index

        # add to, and update mask layer
        mask = self.mask_layer.data
        mask[sam_mask > 0] = sam_mask[sam_mask > 0]
        self.mask_layer.data = mask

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
        self.viewer.layers.remove(self.sam_pts_layer)
        self._add_sam_pts_layer()

        # clear mask
        if self.sam_mask_layer is not None:
            self.sam_mask_layer.data = np.zeros_like(self.sam_mask_layer.data)

    def closeEvent(self, event):
        # try to save the current image on close
        try:
            self.save_image()
        except:
            pass
        event.accept()

    # BUG: no way to save the last image in the dataset? except go back?

    # TODO: go to index



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
