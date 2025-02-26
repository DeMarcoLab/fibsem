import glob
import logging
import os
import sys
from typing import Optional

import napari
import napari.utils.notifications
import numpy as np
import yaml
from napari.layers import Image as NapariImageLayer
from napari.layers import Labels as NapariLabelsLayer
from napari.layers import Points as NapariPointsLayer
from napari.utils.events import Event
from PIL import Image
from PyQt5 import QtWidgets

import fibsem
from fibsem.segmentation import utils as seg_utils
from fibsem.segmentation.config import (
    CLASS_COLORS,
    CLASS_CONFIG_PATH,
    CLASS_LABELS,
    convert_color_names_to_rgb,
)
from fibsem.ui import stylesheets
from fibsem.ui.FibsemSegmentationModelWidget import FibsemSegmentationModelWidget
from fibsem.ui.qtdesigner_files import FibsemLabellingUI as FibsemLabellingDialog
from fibsem.ui.utils import open_existing_directory_dialog, open_existing_file_dialog

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

IMAGE_LAYER_PROPERTIES = {
    "name": "Image",
    "opacity": 0.7,
    "blending": "additive",
    "colormap": "gray",
    "visible": True,
}

MASK_LAYER_PROPERTIES = {
    "name": "Mask",
    "opacity": 0.3,
    "blending": "additive",
    "colormap": CLASS_COLORS,
    "visible": True,
}

SAM_MASK_LAYER_PROPERTIES = {
    "name": "sam_mask",
    "opacity": 0.7,
    "blending": "additive",
    "colormap": {0: "black", 1: "white"},
    "visible": True,
}

CONFIGURATION = {
    "IMAGES": {
        "FILE_EXT": ".tif",
        "SUPPORTED_FILE_EXT": [".tif", ".png", ".jpg", ".jpeg"]
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
        "IMAGE_OPACITY": 0.7,
        "MASK_OPACITY": 0.3,
    },
    "LAYERS": {
        "IMAGE": IMAGE_LAYER_PROPERTIES,
        "MASK": MASK_LAYER_PROPERTIES,
        "SAM_MASK": SAM_MASK_LAYER_PROPERTIES,
    },
    "TOOLTIPS": {
        "AUTOSAVE": "Automatically save the image when moving to the next image",
        "SAVE_RGB": "Save the RGB mask when saving the image. This is in additional to the class mask, and is only for visualisation purposes.",
    }
}

# ref (other sam labelling tools):
# https://github.com/JoOkuma/napari-segment-anything (Apache License 2.0)
# https://github.com/MIC-DKFZ/napari-sam (Apache License 2.0)
class FibsemLabellingUI(FibsemLabellingDialog.Ui_Dialog, QtWidgets.QDialog):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.viewer: napari.Viewer = viewer
        self.last_idx: int = 0
        self.model = None
        self.model_type: str = None
        self.model_assist: bool = False
        self.is_data_loaded: bool = False

        self.img_layer: NapariImageLayer = None
        self.mask_layer: NapariLabelsLayer = None
        
        # sam
        self.sam_mask_layer: NapariLabelsLayer = None
        self.sam_pts_layer: NapariPointsLayer = None   
        self.logits: np.ndarray = None          # sam logits
        self.image: np.ndarray = None           # sam image

        # widget
        self.model_widget = FibsemSegmentationModelWidget()

        self.setup_connections()

        if _DEBUG:
            self.lineEdit_data_path.setText("/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/train")
            self.lineEdit_labels_path.setText("/home/patrick/github/fibsem/fibsem/log/labels2")
            self.model_widget.lineEdit_checkpoint.setText("autolamella-serial-liftout-20240107.pt")
    
    def setup_connections(self):
        self.pushButton_load_data.clicked.connect(self.load_data)

        # save path buttons
        self.pushButton_data_path.clicked.connect(self.select_filepath)
        self.pushButton_labels_path.clicked.connect(self.select_filepath)
        self.model_widget.checkpoint_seg_button.clicked.connect(self.select_filepath)
        self.pushButton_data_config.clicked.connect(self.select_filepath)
        self.lineEdit_data_config.setText(CLASS_CONFIG_PATH)
        self.comboBox_data_file_ext.addItems(CONFIGURATION["IMAGES"]["SUPPORTED_FILE_EXT"])

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
        self.pushButton_load_data.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.model_widget.pushButton_load_model.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_model_confirm.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_model_clear.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)

        # tooltips
        self.checkBox_autosave.setToolTip(CONFIGURATION["TOOLTIPS"]["AUTOSAVE"])
        self.checkBox_save_rgb.setToolTip(CONFIGURATION["TOOLTIPS"]["SAVE_RGB"])

        self.update_instructions()

    def select_filepath(self):
        """Handle filepath selection"""

        if self.sender() == self.pushButton_data_path:
            path = open_existing_directory_dialog(msg="Select Image Data Directory")
            if path is not None and path != "":
                self.lineEdit_data_path.setText(path)
        elif self.sender() == self.pushButton_labels_path:
            path = open_existing_directory_dialog(msg="Select Labels Directory")
            if path is not None and path != "":
                self.lineEdit_labels_path.setText(path)

        elif self.sender() == self.model_widget.checkpoint_seg_button:
            path = open_existing_file_dialog(msg="Select Checkpoint File", _filter=None)
            if path is not None and path != "":
                self.model_widget.lineEdit_checkpoint.setText(path)
        elif self.sender() == self.pushButton_data_config:
            path = open_existing_file_dialog(msg="Select Configuration File", path=CLASS_CONFIG_PATH, _filter="*.yaml")
            if path is not None and path != "":
                self.lineEdit_data_config.setText(path)

                with open(path) as f:
                    CLASS_CONFIG = yaml.load(f, Loader=yaml.FullLoader) 

                CONFIGURATION["LABELS"]["COLOR_MAP"] = CLASS_CONFIG["CLASS_COLORS"]
                CONFIGURATION["LABELS"]["LABEL_MAP"] = CLASS_CONFIG["CLASS_LABELS"]

    def load_data(self):
        """Load images, and set up the viewer"""
        # read raw data
        data_path = self.lineEdit_data_path.text()
        labels_path = self.lineEdit_labels_path.text()
        
        # create required directories
        os.makedirs(labels_path, exist_ok=True)
        
        # check if save and raw are the same 
        if data_path == labels_path:
            napari.utils.notifications.show_error("Save and Raw directories cannot be the same")
            return
            
        # get filenames
        CONFIGURATION["IMAGES"]["FILE_EXT"] = self.comboBox_data_file_ext.currentText()
        file_ext = CONFIGURATION["IMAGES"]["FILE_EXT"]
        filenames = sorted(glob.glob(os.path.join(data_path, f"*{file_ext}*")))
        if len(filenames) == 0:
            napari.utils.notifications.show_error(f"No images found in {data_path}")
            return

        # assign data
        self.data_path = data_path
        self.labels_path = labels_path
        self.filenames = filenames
        self.is_data_loaded = True

        # remove existing layers
        try:
            self.viewer.layers.remove(self.img_layer)
            self.viewer.layers.remove(self.mask_layer)
            self.viewer.layers.remove(self.sam_pts_layer)
            self.viewer.layers.remove(self.sam_mask_layer)
        except Exception:
            pass

        # load filenames as single layer
        image_layer_config = CONFIGURATION["LAYERS"]["IMAGE"]
        self.img_layer = self.viewer.open(filenames, name=image_layer_config["name"], stack=True)[0]
        self.img_layer.opacity = image_layer_config["opacity"]
        self.img_layer.blending = image_layer_config["blending"]

        arr = np.zeros_like(np.asarray(self.img_layer.data[0]))
        mask_layer_config = CONFIGURATION["LAYERS"]["MASK"]
        self.mask_layer = self.viewer.add_labels(
            data=arr,
            name=mask_layer_config["name"],
            opacity=mask_layer_config["opacity"],
            blending=mask_layer_config["blending"],
            colormap=CONFIGURATION["LABELS"]["COLOR_MAP"],
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

    def previous_image(self , event: Optional[Event] = None) -> None:
        idx = int(self.viewer.dims.point[0])
        idx -= 1
        idx = np.clip(idx, 0, len(self.filenames) - 1)
        self.update_viewer_to_image(idx)

    def next_image(self, event: Optional[Event] = None) -> None:
        idx = int(self.viewer.dims.point[0])
        idx += 1
        idx = np.clip(idx, 0, len(self.filenames) - 1)
        self.update_viewer_to_image(idx)

    def save_image(self, event: Optional[Event] = None) -> None:
        """Save the current mask to the labels path"""

        # save current image
        idx = self.last_idx
        fname = self.filenames[idx]
        file_ext = CONFIGURATION["IMAGES"]["FILE_EXT"]
        bname = os.path.basename(fname).removesuffix(file_ext)

        # only resave the labels...
        label = np.asarray(self.mask_layer.data).astype(np.uint8)

        im = Image.fromarray(label) # TODO: replace with tifffile?
        im.save(os.path.join(self.labels_path, f"{bname}{file_ext}"))
        
        logging.info(f"Saving mask to {os.path.basename(fname)}")

        # optionally save an rgb mask (for easy visualisation)
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
        """Update the current image when the index is changed.
        Attempt to autosave the previous image"""
        
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
        """Set the active layer based on the current state of the UI"""
        model_assist = self.checkBox_model_assist.isChecked()
        if model_assist and self.sam_pts_layer is not None:
            self.viewer.layers.selection.active = self.sam_pts_layer
            self.sam_pts_layer.mode = "add"
        else:
            self.viewer.layers.selection.active = self.mask_layer
            self.mask_layer.mode = "paint"

    def update_instructions(self):
        """Update instructions based on the current state of the UI"""
        # display instructions
        msg=""

        if not self.is_data_loaded:
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
        """Get the label (mask) image assoicated with the current image.
        Multi-step process:
            1. Check if a label image exists for the current image. If so, load it.
            2. If no label image exists, check if a model is loaded. If so, use the model to generate a label image.
            3. Otherwise, return a blank image.        
        Returns:
            np.ndarray: label image
        """

        idx = int(self.viewer.dims.point[0])
        fname = self.filenames[idx]
        image = np.asarray(self.img_layer.data[idx]) # req for lazy load

        logging.info(f"IDX: {idx}, Loading image from {os.path.basename(fname)}")

        if os.path.basename(fname) in os.listdir(os.path.join(self.labels_path)):
            label_fname = os.path.join(self.labels_path, os.path.basename(fname))
            label_image = Image.open(label_fname).convert("L")
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
                msg = "No label image found, using SAM model to generate mask."

        else:
            label_image = np.zeros_like(image)

            msg = "No label image found, and no model loaded"

        napari.utils.notifications.show_info(msg)

        return np.asarray(label_image).astype(np.uint8)

    ####### MODEL
    def toggle_model_assist(self):
        """Toggle model assisted labelling on/off"""
        self.model_assist = self.checkBox_model_assist.isChecked()

        if self.model is None:
            self.label_model_info.setText("Please load a model.")
        if self.is_data_loaded:
            self.update_image()

    def load_model(self):
        """Load the model and update the UI based on the selected model"""
        self.model = self.model_widget.model
        self.model_type =  self.model_widget.model_type
        self.label_model_info.setText(f"Model: {os.path.basename(self.model.checkpoint)}")

        # specific layers for SAM model
        if self.model_type == "SegmentAnythingModel":
            self.add_sam_pts_layer()
            
            self.pushButton_model_confirm.clicked.connect(self.accept_sam_mask)
            self.pushButton_model_clear.clicked.connect(self.clear_sam_data)
            self.pushButton_model_confirm.setVisible(True)

            self.pushButton_model_clear.setVisible(True)

            self.viewer.bind_key("C", self.accept_sam_mask, overwrite=True)
            self.viewer.bind_key("X", self.clear_sam_data, overwrite=True)

        napari.utils.notifications.show_info(
            f"Loaded {self.model_type}: {os.path.basename(self.model.checkpoint)}"
        )

        self.update_instructions()

        if self.is_data_loaded:
            self.update_image()

    ####### SAM
    def add_sam_pts_layer(self):
        """Add a points layer for SAM prompts and associated callbacks"""
        self.sam_pts_layer = self.viewer.add_points(name="SAM Points")
        self.sam_pts_layer.events.data.connect(self._update_sam_mask)
        self.sam_pts_layer.mouse_drag_callbacks.append(self._mouse_button_modifier)
        self.sam_pts_layer.mode = "add"

    def _mouse_button_modifier(self, layer: NapariPointsLayer, event: Event) -> None:
        """Use the mouse button modifier to set the point prompt 
        type (left click = positive, right click = negative)"""
        self.sam_pts_layer.selected_data = []
         # prompt schema: 1: pos, 0: neg, -1: background
        if event.button == 1:
            self.sam_pts_layer.current_face_color = "blue"
            self.sam_pts_layer.current_properties = {"prompt": 1}
        else:
            self.sam_pts_layer.current_face_color = "red"
            self.sam_pts_layer.current_properties = {"prompt": 0}

    def _update_sam_mask(self, event: Optional[Event] = None) -> None:
        """Update the SAM mask using the current SAM prompt points"""

        points = None
        points_labels = None
        # get the prompt points from sam_pts layer
        if len(self.sam_pts_layer.data) > 0:
            points = [np.flip(self.sam_pts_layer.data, axis=-1).tolist()]
            points_labels = [self.sam_pts_layer.properties["prompt"].tolist()]

        # prepare image for sam model (requires rgb)
        if self.image is None:
            idx = int(self.viewer.dims.point[0])
            image = np.asarray(self.img_layer.data[idx]) # req for lazy load
            self.image = Image.fromarray(image).convert("RGB")

        # TODO: dont' recompute image embedding each time
        mask, score, self.logits = self.model.predict(image=self.image, 
                                                points=points, 
                                                labels=points_labels, 
                                                input_masks=self.logits,
                                                multimask_output=False)

        # add layer for mask
        try:
            self.sam_mask_layer.data = mask
        except Exception:
            sam_mask_config = CONFIGURATION["LAYERS"]["SAM_MASK"]
            self.sam_mask_layer = self.viewer.add_labels(
                data=mask,
                name=sam_mask_config["name"],
                opacity=sam_mask_config["opacity"],
                blending= sam_mask_config["blending"],
                colormap=sam_mask_config["colormap"],
            )

        self.set_sam_points_active()

    def set_sam_points_active(self):
        """Enable the SAM points layer for interaction"""
        try:
            # set sam points layer active
            self.viewer.layers.selection.active = self.sam_pts_layer
            self.sam_pts_layer.mode = "add"
        except Exception as e:
            logging.warning(f"Unable to set SAM points layer active: {e}")

    def accept_sam_mask(self, event: Optional[Event] = None) -> None:
        """Accept the current SAM mask, and add it to the mask layer"""
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

        self.clear_sam_data()

    def increment_class_index(self, event: Optional[Event] = None) -> None:
        """Increment the class index by 1"""
        self.change_class_index(1)

    def decrement_class_index(self, event: Optional[Event] = None) -> None:
        """Decrement the class index by 1"""
        self.change_class_index(-1)

    # TODO: change this function to just directly set the index to a value, rather than this weird increment/decrement
    def change_class_index(self, val: int = 0) -> None:
        """Increment/Decrement the the labelling class index, and change the mask layer colour
        Args:
            val: class index change. Defaults to 0.
        """
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

    def clear_sam_data(self, event = None) -> None:
        """Clear SAM points and mask data layers"""
        # clear points
        # TODO: find the proper way to do this, not doing this way causes issues with face colour
        self.viewer.layers.remove(self.sam_pts_layer)
        self.add_sam_pts_layer()

        # clear mask
        if self.sam_mask_layer is not None:
            self.sam_mask_layer.data = np.zeros_like(self.sam_mask_layer.data)

    def closeEvent(self, event) -> None:
        """Attempt to save the current image before closing the window"""
        try:
            self.save_image()
        except Exception:
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
