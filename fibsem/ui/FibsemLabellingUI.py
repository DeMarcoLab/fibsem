import logging
import os
import sys
import glob

import fibsem
import napari
import napari.utils.notifications
import numpy as np
import tifffile as tff
from PIL import Image
from fibsem.ui.qtdesigner_files import FibsemLabellingUI
from PyQt5 import QtWidgets
from fibsem.segmentation.config import CLASS_COLORS, CLASS_LABELS, convert_color_names_to_rgb
from fibsem.segmentation import utils as seg_utils

from fibsem.ui.FibsemSegmentationModelWidget import FibsemSegmentationModelWidget
from fibsem.ui import _stylesheets
from fibsem.ui.utils import _get_directory_ui,_get_file_ui

from napari.layers import Points
from typing import Any, Generator, Optional

# setup a basic logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

_DEBUG = False

# new line char in html
NL = "<br>"

START_INSTRUCTIONS = f"""
<h3>Instructions</h3>
Welcome to the OpenFIBSEM Labelling Tool.{NL}
This tool is designed to assist with labelling 2D images for segmentation.{NL}

<h4>Getting Started</h4>
Select the folder containing the image data.{NL}
Select the folder to save the labels to.{NL}
Select a custom colormap [Optional].{NL}

"""
READY_INSTRUCTIONS = f"""
<h4>Ready to Label</h4>
Use the paintbrush to label the image.{NL}
Use the eraser to remove labels.{NL}
Press R/E to go to the next/previous image.{NL}
Press D/F to change the selected class label.{NL}{NL}

<h4>Saving</h4>
The image will be saved when you move to the next image.{NL}
You can also save the image by pressing S.{NL}{NL}

"""

MODEL_NOT_LOADED_INSTRUCTIONS = f"""
<h4>Model Assisted Labelling</h4>
You can use a model to assist with labelling.{NL}
Select the model to use for segmentation.{NL}
The model will be used to generate a mask for the image.{NL}
You are then able to use the paintbrush to correct the mask.{NL}{NL}
"""

MODEL_LOADED_INSTRUCTIONS = f"""
<h4>Model Assisted Labelling</h4>
The Segmentation Model will assist with labelling images.{NL}
The model will be used to generate a mask for the image.{NL}
You are then able to use the paintbrush to correct the mask.{NL}{NL}
"""

SAM_MODEL_LOADED_INSTRUCTIONS = f"""
<h4>Segment Anything Model</h4>
The SAM Model will assist with labelling images.{NL}
Left Click to add a point to the mask.{NL}
Right Click to remove a point from the mask.{NL}
Press C to confirm the SAM mask.{NL}
Press X to clear the SAM mask.{NL}{NL}


"""

INSTRUCTIONS = {
    "START": START_INSTRUCTIONS,
    "READY": READY_INSTRUCTIONS,
    "MODEL_NOT_LOADED": MODEL_NOT_LOADED_INSTRUCTIONS,
    "MODEL_LOADED": MODEL_LOADED_INSTRUCTIONS,
    "SAM_MODEL_LOADED": SAM_MODEL_LOADED_INSTRUCTIONS,
}

CONFIGURATION = {
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
    "UI": {
        "INSTRUCTIONS": INSTRUCTIONS,
        "LOAD_DATA_BUTTON_COLOR": _stylesheets._BLUE_PUSHBUTTON_STYLE,
        "LOAD_MODEL_BUTTON_COLOR": _stylesheets._BLUE_PUSHBUTTON_STYLE,
        "CONFIRM_BUTTON_COLOR": _stylesheets._GREEN_PUSHBUTTON_STYLE,
        "CLEAR_BUTTON_COLOR": _stylesheets._RED_PUSHBUTTON_STYLE,
        "IMAGE_OPACITY": 0.7,
        "MASK_OPACITY": 0.3,
    },
    "TOOLTIPS": {
        "AUTOSAVE": "Automatically save the image when moving to the next image",
        "SAVE_RGB": "Save the RGB mask when saving the image. This is in additional to the class mask, and is only for visualisation purposes.",
    }

}

# ref (other sam labelling tools):
# https://github.com/JoOkuma/napari-segment-anything (Apache License 2.0)
# https://github.com/MIC-DKFZ/napari-sam (Apache License 2.0)
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

        self.img_layer = None
        self.mask_layer = None
        
        # sam
        self.sam_mask_layer = None
        self.sam_pts_layer = None   
        self.logits = None          # sam logits
        self.image = None           # sam image

        self.setup_connections()

        if _DEBUG:
            self.lineEdit_data_path.setText("/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/train")
            self.lineEdit_labels_path.setText("/home/patrick/github/fibsem/fibsem/log/labels2")
            self.model_widget.lineEdit_checkpoint.setText("autolamella-serial-liftout-20240107.pt")
    
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

        self.model_widget.model_loaded.connect(self.load_model)

        self.checkBox_model_assist.stateChanged.connect(self.toggle_model_assist)

        self.viewer.bind_key("R", self.next_image)
        self.viewer.bind_key("E", self.previous_image)
        self.viewer.bind_key("S", self.save_image)
        self.viewer.bind_key("D", self.decrement_class_index)
        self.viewer.bind_key("F", self.increment_class_index)

        # SAM
        self.pushButton_model_confirm.setVisible(False)
        self.pushButton_model_clear.setVisible(False)

        # style
        self.pushButton_load_data.setStyleSheet(CONFIGURATION["UI"]["LOAD_DATA_BUTTON_COLOR"])
        self.model_widget.pushButton_load_model.setStyleSheet(CONFIGURATION["UI"]["LOAD_MODEL_BUTTON_COLOR"])
        self.pushButton_model_confirm.setStyleSheet(CONFIGURATION["UI"]["CONFIRM_BUTTON_COLOR"])
        self.pushButton_model_clear.setStyleSheet(CONFIGURATION["UI"]["CLEAR_BUTTON_COLOR"])

        # tooltips
        self.checkBox_autosave.setToolTip(CONFIGURATION["TOOLTIPS"]["AUTOSAVE"])
        self.checkBox_save_rgb.setToolTip(CONFIGURATION["TOOLTIPS"]["SAVE_RGB"])

        self.update_instructions()

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
            path = _get_file_ui(msg="Select Checkpoint File", _filter=None)
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
        FILE_EXT = CONFIGURATION["IMAGES"]["FILE_EXT"]
        filenames = sorted(glob.glob(os.path.join(data_path, f"*{FILE_EXT}*")))
        if len(filenames) == 0:
            napari.utils.notifications.show_error(f"No images found in {data_path}")
            return

        # assign data
        self.data_path = data_path
        self.labels_path = labels_path
        self.filenames = filenames
        self.DATA_LOADED = True

        # remove existing layers
        try:
            self.viewer.layers.remove(self.img_layer)
            self.viewer.layers.remove(self.mask_layer)
            self.viewer.layers.remove(self.sam_pts_layer)
            self.viewer.layers.remove(self.sam_mask_layer)
        except:
            pass

        # load filenames as single layer
        self.img_layer = self.viewer.open(filenames, name="Image", stack=True)[0]
        self.img_layer.opacity = CONFIGURATION["UI"]["IMAGE_OPACITY"]
        self.img_layer.blending="additive"

        arr = np.zeros_like(np.asarray(self.img_layer.data[0]))
        self.mask_layer = self.viewer.add_labels(
            arr,
            name="Mask",
            opacity=CONFIGURATION["UI"]["MASK_OPACITY"],
            blending="additive",
            color=CONFIGURATION["LABELS"]["COLOR_MAP"],
        )

        self.viewer.dims.events.current_step.connect(self.update_image)
        self.update_viewer_to_image(0)

        self.update_ui_elements()

        self.update_instructions()

        # TODO: fix this event, it just doesn't work?
        # self.mask_layer.events.data.connect(self.save_image) 

    def update_ui_elements(self):
        # update ui elements on data loaded
        n_labels = len(CONFIGURATION["LABELS"]["LABEL_MAP"])
        ldict = CONFIGURATION['LABELS']['LABEL_MAP']
        cdict = CONFIGURATION['LABELS']['COLOR_MAP']
        label_map = [f"{i:02d} - {ldict.get(i, 'Unspecified')} ({cdict.get(i, 'Unspecified')})" for i in range(n_labels)]
        self.comboBox_model_class_index.clear()
        self.comboBox_model_class_index.addItems(label_map)

    def previous_image(self , _: Optional[Any] = None) -> None:
        idx = int(self.viewer.dims.point[0])
        idx -= 1
        idx = np.clip(idx, 0, len(self.filenames) - 1)
        self.update_viewer_to_image(idx)

    def next_image(self, _: Optional[Any] = None) -> None:
        idx = int(self.viewer.dims.point[0])
        idx += 1
        idx = np.clip(idx, 0, len(self.filenames) - 1)
        self.update_viewer_to_image(idx)

    def save_image(self, _: Optional[Any] = None) -> None:

        # save current image
        idx = self.last_idx
        fname = self.filenames[idx]
        FILE_EXT = CONFIGURATION["IMAGES"]["FILE_EXT"]
        bname = os.path.basename(fname).removesuffix(FILE_EXT)

        # only resave the labels...
        label = np.asarray(self.mask_layer.data).astype(np.uint8)

        im = Image.fromarray(label) # TODO: replace with tifffile?
        im.save(os.path.join(self.labels_path, f"{bname}{FILE_EXT}"))
        
        logging.info(f"Saving mask to {os.path.basename(fname)}")

        if CONFIGURATION["SAVE"]["SAVE_RGB"] and self.checkBox_save_rgb.isChecked():
            
            colormap = CONFIGURATION["LABELS"]["COLOR_MAP"]
            colormap_rgb = convert_color_names_to_rgb(colormap)

            rgb = seg_utils.decode_segmap_v2(label, colormap_rgb)
            rgb = Image.fromarray(rgb)

            RGB_PATH = os.path.join(self.labels_path, "rgb")
            os.makedirs(RGB_PATH, exist_ok=True)
            rgb.save(os.path.join(RGB_PATH, f"{bname}.png"))

            logging.info(f"Saving RGB mask to {os.path.join(RGB_PATH, f'{bname}.png')}")

    
    def update_viewer_to_image(self, idx: int):
        self.viewer.dims.set_point(0, idx)

    def update_image(self):
        
        idx = int(self.viewer.dims.point[0])
        logging.info(f"UPDATING IMAGE: LAST IDX: {self.last_idx}, CURRENT IDX {idx}")
        
        # save previous image
        if self.last_idx != idx:
            if self.checkBox_autosave.isChecked():
                self.save_image()
            self.last_idx = idx # update last index
        
        # update mask layer
        self.mask_layer.data = self.get_label_image()

        # set active layers
        if self.model is not None:
            if self.model_type == "SegmentAnythingModel":
                self.model.is_image_set = False # TODO: implement this, so we don't have to recompute the image embedding each time
                self.logits = None
                self.image = None
        self.set_active_layers()

    def set_active_layers(self):
        model_assist = self.checkBox_model_assist.isChecked()
        if model_assist and self.sam_pts_layer is not None:
            self.viewer.layers.selection.active = self.sam_pts_layer
            self.sam_pts_layer.mode = "add"
        else:
            self.viewer.layers.selection.active = self.mask_layer
            self.mask_layer.mode = "paint"

    def update_instructions(self):
        # display instructions
        msg=""

        if not self.DATA_LOADED:
            msg = INSTRUCTIONS["START"]
        else: 
            msg = INSTRUCTIONS["READY"]

            if self.model is None: 
                msg += INSTRUCTIONS["MODEL_NOT_LOADED"]

            elif self.model is not None:

                if self.model_type == "SegmentAnythingModel":
                    msg += INSTRUCTIONS["SAM_MODEL_LOADED"]
                else:
                    msg += INSTRUCTIONS["MODEL_LOADED"]

        self.label_instructions.setText(msg)

    def get_label_image(self) -> np.ndarray:

        idx = int(self.viewer.dims.point[0])
        fname = self.filenames[idx]
        image = np.asarray(self.img_layer.data[idx]) # req for lazy load

        logging.info(f"IDX: {idx}, Loading image from {os.path.basename(fname)}")

        if os.path.basename(fname) in os.listdir(os.path.join(self.labels_path)):
            label_fname = os.path.join(self.labels_path, os.path.basename(fname))
            label_image = tff.imread(label_fname)
            label_image = np.array(label_image, dtype=np.uint8)

            msg = f"Loaded label image from {os.path.basename(label_fname)}"

        elif self.model_assist and self.model is not None:
            if self.model_type == "SegmentationModel":
                label_image = self.model.inference(image, rgb=False)
                if label_image.ndim == 3:
                    label_image = label_image[0]
                msg = f"Generated label image using {os.path.basename(self.model.checkpoint)}"
            if self.model_type == "SegmentAnythingModel":
                label_image = np.zeros_like(image)

                msg = f"No label image found, using SAM model to generate mask."

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
        self.model_type =  self.model_widget.model_type
        self.label_model_info.setText(f"Model: {os.path.basename(self.model.checkpoint)}")

        # specific layers for SAM model
        if self.model_type == "SegmentAnythingModel":
            self._add_sam_pts_layer()
            
            self.pushButton_model_confirm.clicked.connect(self._confirm_model_class)
            self.pushButton_model_clear.clicked.connect(self._clear_sam_data)
            self.pushButton_model_confirm.setVisible(True)

            self.pushButton_model_clear.setVisible(True)

            self.viewer.bind_key("C", self._confirm_model_class, overwrite=True)
            self.viewer.bind_key("X", self._clear_sam_data, overwrite=True)

        napari.utils.notifications.show_info(
            f"Loaded {self.model_type}: {os.path.basename(self.model.checkpoint)}"
        )

        self.update_instructions()

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
            self.sam_pts_layer.current_properties = {"prompt": 1}
        else:
            self.sam_pts_layer.current_face_color = "red"
            self.sam_pts_layer.current_properties = {"prompt": 0} # 1 = pos, 0 = neg, -1 = background

    def _update_sam_mask(self, _: Optional[Any] = None) -> None:

        points = None
        points_labels = None
        # get the prompt points from sam_pts layer
        if len(self.sam_pts_layer.data) > 0:
            points = [np.flip(self.sam_pts_layer.data, axis=-1).tolist()]
            points_labels = [self.sam_pts_layer.properties["prompt"].tolist()]

        if self.image is None:
            idx = int(self.viewer.dims.point[0])
            image = np.asarray(self.img_layer.data[idx]) # req for lazy load
            self.image = Image.fromarray(image).convert("RGB")

        # TODO: dont' recompute image embedding each time
        mask, score, self.logits = self.model.predict(self.image, 
                                                points, 
                                                points_labels, 
                                                input_masks=self.logits,
                                                multimask_output=False)

        # add layer for mask
        try:
            self.sam_mask_layer.data = mask
        except:
            self.sam_mask_layer = self.viewer.add_labels(
                mask,
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
        cidx = self.comboBox_model_class_index.currentIndex()

        # set all values of sam mask to class index
        sam_mask[sam_mask > 0] = cidx

        # add to, and update mask layer
        mask = self.mask_layer.data
        mask[sam_mask > 0] = sam_mask[sam_mask > 0]
        self.mask_layer.data = mask

        self._clear_sam_data()

    def increment_class_index(self, _: Optional[Any] = None) -> None:
        self.change_class_index(1)

    def decrement_class_index(self, _: Optional[Any] = None) -> None:
        self.change_class_index(-1)

    def change_class_index(self, val: int = 0) -> None:
        # get the current class index
        cidx = self.comboBox_model_class_index.currentIndex()
        cidx += val

        # clip to 0,number of classes
        n_labels = len(CONFIGURATION["LABELS"]["LABEL_MAP"])
        cidx = np.clip(cidx, 0, n_labels - 1)

        self.comboBox_model_class_index.setCurrentIndex(cidx)
        self.viewer.status = f"Current Class: {cidx}"

        # set the label index for the mask layer
        if self.mask_layer is not None:
            self.mask_layer.selected_label = cidx

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


def main():
    viewer = napari.Viewer(ndisplay=2)
    fibsem_labelling_ui = FibsemLabellingUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_labelling_ui,
        area="right",
        add_vertical_stretch=True,
        name=f"OpenFIBSEM v{fibsem.__version__} - Image Labelling",

    )
    napari.run()


if __name__ == "__main__":
    main()
