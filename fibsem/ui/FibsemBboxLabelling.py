import logging
import os
import sys
import glob

import yaml
import fibsem
import napari
import napari.utils.notifications
import numpy as np
from PIL import Image
from fibsem.ui.qtdesigner_files import FibsemBBoxLabellingWidget
from PyQt5 import QtWidgets
from fibsem.segmentation.config import CLASS_COLORS, CLASS_LABELS, convert_color_names_to_rgb, CLASS_CONFIG_PATH

from fibsem.ui import _stylesheets
from fibsem.ui.utils import _get_directory_ui,_get_file_ui, _get_save_file_ui

from napari.layers import Points, Shapes
from typing import Any, Generator, Optional

# setup a basic logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import pandas as pd

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


INSTRUCTIONS = {
    "START": START_INSTRUCTIONS,
    "READY": READY_INSTRUCTIONS,

}

CONFIGURATION = {
    "IMAGES": {
        "FILE_EXT": ".tif",
        "SUPPORTED_FILE_EXT": [".tif", ".png", ".jpg", ".jpeg"]
    },
    "LABELS": {
        "SHAPE_LAYER_NAME": "Bounding Boxes",
        "COLOR_MAP": CLASS_COLORS,
        "LABEL_MAP": CLASS_LABELS,
    },
    "SAVE": {
        "SAVE_RGB": True,
    },
    "UI": {
        "INSTRUCTIONS": INSTRUCTIONS,
        "LOAD_DATA_BUTTON_COLOR": _stylesheets._BLUE_PUSHBUTTON_STYLE,
        "CONFIRM_BUTTON_COLOR": _stylesheets._GREEN_PUSHBUTTON_STYLE,
        "CLEAR_BUTTON_COLOR": _stylesheets._RED_PUSHBUTTON_STYLE,
        "IMAGE_OPACITY": 0.7,
        "BBOX_OPACITY": 0.3,
        "BBOX_EDGE_WIDTH": 5,
    },
    "TOOLTIPS": {
        "AUTOSAVE": "Automatically save the image when moving to the next image",
        "SAVE_RGB": "Save the RGB mask when saving the image. This is in additional to the class mask, and is only for visualisation purposes.",
    }

}

class FibsemBBoxLabellingUI(FibsemBBoxLabellingWidget.Ui_Form, QtWidgets.QDialog):
    def __init__(
        self,
        viewer: napari.Viewer,
        parent=None,
    ):
        super(FibsemBBoxLabellingUI, self).__init__(parent=parent)
        self.setupUi(self)
        self.viewer = viewer
        self.last_idx = 0
        self.DATA_LOADED = False

        self.img_layer = None
        self.shapes_layer = None

        self.bboxes = []

        self.setup_connections()

        self.lineEdit_data_path.setText("/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/train")
        self.lineEdit_labels_path.setText("/home/patrick/github/fibsem/fibsem/log/test-bbox-1")

    def setup_connections(self):
        self.pushButton_load_data.clicked.connect(self.load_data)
        
        # save path buttons

        self.pushButton_data_path.clicked.connect(self.select_filepath)
        self.pushButton_labels_path.clicked.connect(self.select_filepath)
        self.pushButton_data_config.clicked.connect(self.select_filepath)
        self.lineEdit_data_config.setText(CLASS_CONFIG_PATH)
        self.comboBox_data_file_ext.addItems(CONFIGURATION["IMAGES"]["SUPPORTED_FILE_EXT"])

        self.viewer.bind_key("R", self.next_image)
        self.viewer.bind_key("E", self.previous_image)
        self.viewer.bind_key("S", self.save_image)
        self.viewer.bind_key("D", self.decrement_class_index)
        self.viewer.bind_key("F", self.increment_class_index)

        self.comboBox_class_map.currentIndexChanged.connect(self._change_class_index)
        self.pushButton_add_mode.clicked.connect(self._add_shape_mode)
        self.pushButton_edit_mode.clicked.connect(self._edit_shape_mode)

        # style
        self.pushButton_load_data.setStyleSheet(CONFIGURATION["UI"]["LOAD_DATA_BUTTON_COLOR"])

        # tooltips
        self.checkBox_autosave.setToolTip(CONFIGURATION["TOOLTIPS"]["AUTOSAVE"])
        # self.checkBox_save_rgb.setToolTip(CONFIGURATION["TOOLTIPS"]["SAVE_RGB"])

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

        elif self.sender() == self.pushButton_data_config:
            path = _get_file_ui(msg="Select Configuration File", path=CLASS_CONFIG_PATH, _filter="*.yaml")
            if path is not None and path != "":
                self.lineEdit_data_config.setText(path)

                with open(path) as f:
                    CLASS_CONFIG = yaml.load(f, Loader=yaml.FullLoader) 

                CONFIGURATION["LABELS"]["COLOR_MAP"] = CLASS_CONFIG["CLASS_COLORS"]
                CONFIGURATION["LABELS"]["LABEL_MAP"] = CLASS_CONFIG["CLASS_LABELS"]


    def increment_class_index(self, _: Optional[Any] = None) -> None:
        self.change_class_index(1)

    def decrement_class_index(self, _: Optional[Any] = None) -> None:
        self.change_class_index(-1)

    def _change_class_index(self) -> None:
        # disable signals to prevent circular update the current index
        self.comboBox_class_map.blockSignals(True)
        self.change_class_index()
        self.comboBox_class_map.blockSignals(False)

    def change_class_index(self, val: int = 0) -> None:
        # get the current class index
        cidx = self.comboBox_class_map.currentIndex()
        cidx += val

        # clip to 0,number of classes
        n_labels = len(CONFIGURATION["LABELS"]["LABEL_MAP"])
        cidx = np.clip(cidx, 0, n_labels - 1)

        if cidx != self.comboBox_class_map.currentIndex():
            self.comboBox_class_map.setCurrentIndex(cidx)
        self.viewer.status = f"Current Class: {cidx}"
        
        # set the label index for the mask layer
        if self.shapes_layer is not None:
            self.shapes_layer.current_edge_color = CONFIGURATION["LABELS"]["COLOR_MAP"][cidx]
            self.shapes_layer.current_face_color = "white"

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
        CONFIGURATION["IMAGES"]["FILE_EXT"] = self.comboBox_data_file_ext.currentText()
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
            self.viewer.layers.remove(self.shapes_layer)
        except:
            pass

        # load filenames as single layer
        self.img_layer = self.viewer.open(filenames, name="Image", stack=True)[0]
        self.img_layer.opacity = CONFIGURATION["UI"]["IMAGE_OPACITY"]
        self.img_layer.blending="additive"

        self.shapes_layer = self.viewer.add_shapes(
            name=CONFIGURATION["LABELS"]["SHAPE_LAYER_NAME"],
            opacity=CONFIGURATION["UI"]["BBOX_OPACITY"],
            edge_width=CONFIGURATION["UI"]["BBOX_EDGE_WIDTH"],
            face_color="white",
            edge_color="white",
            blending="additive",
        )

        self.shapes_layer.events.data.connect(self._shapes_updated)


        self.viewer.dims.events.current_step.connect(self.update_image)
        self.update_viewer_to_image(0)

        self.update_ui_elements()

        self.update_instructions()

        # TODO: fix this event, it just doesn't work?
        # self.shapes_layer.events.data.connect(self.save_image) 

    def update_ui_elements(self):
        # update ui elements on data loaded
        n_labels = len(CONFIGURATION["LABELS"]["LABEL_MAP"])
        ldict = CONFIGURATION['LABELS']['LABEL_MAP']
        cdict = CONFIGURATION['LABELS']['COLOR_MAP']
        label_map = [f"{i:02d} - {ldict.get(i, 'Unspecified')} ({cdict.get(i, 'Unspecified')})" for i in range(n_labels)]
        self.comboBox_class_map.clear()
        self.comboBox_class_map.addItems(label_map)

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
        # self.shapes_layer.data = self.get_label_image()

        # set active layers
        self.set_active_layers()

    def set_active_layers(self):

        self.viewer.layers.selection.active = self.shapes_layer
        self.shapes_layer.mode = "add_rectangle"


    def update_instructions(self):
        # display instructions
        msg=""

        if not self.DATA_LOADED:
            msg = INSTRUCTIONS["START"]
        else: 
            msg = INSTRUCTIONS["READY"]

        self.label_instructions.setText(msg)

    def get_label_image(self) -> np.ndarray:
        pass


    def closeEvent(self, event):
        # try to save the current image on close
        try:
            self.save_image()
        except:
            pass
        event.accept()



    def _add_shape_mode(self):
        """Add a feature to the currently selected image"""
        self.viewer.layers.selection.active = self.shapes_layer
        self.shapes_layer.mode = "add_rectangle"

    def _edit_shape_mode(self):
        """Edit the currently selected feature"""
        self.viewer.layers.selection.active = self.shapes_layer
        self.shapes_layer.mode = "select"

    def _shapes_updated(self, event: Optional[Any] = None) -> None:
        # if self.feature is None:
        #     self._get_feature()

        # if self.feature is None:
        #     napari.utils.notifications.show_warning(f"No feature selected. Exiting.")
        #     return

        # get latest shapes
        shapes = self.shapes_layer.data

        # get which point was moved
        index: list[int] = list(self.shapes_layer.selected_data)

        # def napari_pt_to_point(pt: np.ndarray, dtype=int) -> Point:
        #     return Point.from_list(np.flip(pt, axis=-1).astype(dtype))


        for shape in shapes:
            print("--------")
            print("shape: ", shape)
            reduced_area = convert_shape_to_image_area(shape, self.img_layer.data.shape)
            print(reduced_area, is_valid_reduced_area(reduced_area))

        print("-"*80)

        # TODO: map -> classes

        # add new feature
        # if len(points) > len(self.features):
        #     # get latest feature
        #     feature = copy.deepcopy(self.feature)
        #     feature.update(napari_pt_to_point(points[-1]).to_dict())
        #     feature["color"] = get_feature_color(feature["name"])
        #     self.features.append(copy.deepcopy(feature))

        # # feature was deleted
        # elif len(points) < len(self.features):
        #     # deleted

        #     for idx in index[::-1]:
        #         logging.info(f"point deleted: {self.features[idx]['name']}")
        #         self.features.pop(idx)

        # # feature was moved
        # else:
        #     # moved
        #     for idx in index:
        #         self.features[idx].update(
        #             copy.deepcopy(napari_pt_to_point(points[idx])).to_dict()
        #         )

        #
        # text = {
        #     "string": [f["name"] for f in self.features],
        #     "color": CONFIGURATION["UI"]["TEXT_COLOR"],
        #     "translation": CONFIGURATION["UI"]["TEXT_TRANSLATION"],
        # }

        # self.pts_layer.text = text
        # self.pts_layer.face_color = [f["color"] for f in self.features]

        # update dataframe, write to csv
        # self.update_dataframe()


    # def _features_updated(self, event: Optional[Any] = None) -> None:
    #     if self.feature is None:
    #         self._get_feature()

    #     if self.feature is None:
    #         napari.utils.notifications.show_warning(f"No feature selected. Exiting.")
    #         return

    #     # get latest points
    #     points = self.pts_layer.data

    #     # get which point was moved
    #     index: list[int] = list(self.pts_layer.selected_data)

    #     def napari_pt_to_point(pt: np.ndarray, dtype=int) -> Point:
    #         return Point.from_list(np.flip(pt, axis=-1).astype(dtype))

    #     # add new feature
    #     if len(points) > len(self.features):
    #         # get latest feature
    #         feature = copy.deepcopy(self.feature)
    #         feature.update(napari_pt_to_point(points[-1]).to_dict())
    #         feature["color"] = get_feature_color(feature["name"])
    #         self.features.append(copy.deepcopy(feature))

    #     # feature was deleted
    #     elif len(points) < len(self.features):
    #         # deleted

    #         for idx in index[::-1]:
    #             logging.info(f"point deleted: {self.features[idx]['name']}")
    #             self.features.pop(idx)

    #     # feature was moved
    #     else:
    #         # moved
    #         for idx in index:
    #             self.features[idx].update(
    #                 copy.deepcopy(napari_pt_to_point(points[idx])).to_dict()
    #             )

    #     #
    #     text = {
    #         "string": [f["name"] for f in self.features],
    #         "color": CONFIGURATION["UI"]["TEXT_COLOR"],
    #         "translation": CONFIGURATION["UI"]["TEXT_TRANSLATION"],
    #     }

    #     self.pts_layer.text = text
    #     self.pts_layer.face_color = [f["color"] for f in self.features]

    #     # update dataframe, write to csv
    #     self.update_dataframe()

    def update_dataframe(self):
        """Update the dataframe with the latest features"""
        idx = int(self.viewer.dims.point[0])
        fname = self.filenames[idx]

        df: pd.DataFrame = self.df
        # get features
        features = self.features

        # if features is empty, add empty row to maintain image in dataframe
        if len(features) == 0:
            features = [{"name": np.nan, "x": np.nan, "y": np.nan}]

        # get dataframe for current image
        df_filt = df[df["filename"] == os.path.basename(fname)]

        # create dataframe
        df_new = pd.DataFrame(
            columns=[
                "filename",
                "feature",
                "px.x",
                "px.y",
                "pixelsize",
                "corrected",
                "method",
                "experiment",
            ]
        )
        df_new["filename"] = [os.path.basename(fname)] * len(features)
        df_new["feature"] = [f["name"] for f in features]
        df_new["px.x"] = [f["x"] for f in features]
        df_new["px.y"] = [f["y"] for f in features]
        df_new["pixelsize"] = [None] * len(features)  # pixel size is not known
        df_new["corrected"] = [True] * len(features)
        df_new["method"] = [df_filt["method"].values[0]] * len(features)
        df_new["experiment"] = [df_filt["experiment"].values[0]] * len(features)

        # update dataframe
        df.drop(df_filt.index, inplace=True)
        df = pd.concat([df, df_new], axis=0)
        df.reset_index(drop=True, inplace=True)  # mind your index

        # save dataframe
        df.to_csv(self.csv_path, index=False)

        # update dataframe
        self.df = df

    def update_points_from_dataframe(self, idx: int):
        """Update the points layer from the dataframe"""
        # get dataframe for current image
        fname = self.filenames[idx]
        df_filt = self.df[self.df["filename"] == os.path.basename(fname)]

        # initial empty df, then no features were added
        if len(df_filt) == 1 and pd.isna(df_filt["feature"].values[0]):
            napari.utils.notifications.show_info(
                f"No features yet for {os.path.basename(fname)}"
            )
            return

        # load features from dataframe
        for _, row in df_filt.iterrows():
            d = {
                "name": row["feature"],
                "x": row["px.x"],
                "y": row["px.y"],
                "color": get_feature_color(row["feature"]),
            }
            self.features.append(copy.deepcopy(d))

        # update points layer
        self.pts_layer.data = np.array([[f["y"], f["x"]] for f in self.features])

        # update text
        text = {
            "string": [f["name"] for f in self.features],
            "color": "white",
            "translation": np.array([-30, 0]),
        }

        self.pts_layer.text = text
        self.pts_layer.face_color = [f["color"] for f in self.features]
        napari.utils.notifications.show_info(
            f"Loaded {len(self.features)} features from {os.path.basename(fname)}"
        )


from fibsem.structures import FibsemRectangle
def convert_reduced_area_to_napari_shape(reduced_area: FibsemRectangle, image_shape: tuple, offset_shape: tuple = None) -> np.ndarray:
    """Convert a reduced area to a napari shape."""
    x0 = reduced_area.left * image_shape[1]
    y0 = reduced_area.top * image_shape[0]
    if offset_shape:
        x0 += offset_shape[1]
    x1 = x0 + reduced_area.width * image_shape[1]
    y1 = y0 + reduced_area.height * image_shape[0]
    data = [[y0, x0], [y0, x1], [y1, x1], [y1, x0]]
    return data
            #     data = convert_reduced_area_to_napari_shape(FibsemRectangle(0.375, 0.375, 0.25, 0.25), self.image_widget.ib_image.data.shape, self.image_widget.eb_image.data.shape)
            #     # TODO: create as FibsemRectangle -> Convert to shape
            #     # data = create_default_alignment_area(self.image_widget.ib_image.data.shape, self.image_widget.eb_image.data.shape)

            #     self.alignment_layer = self.viewer.add_shapes(data=data, name="alignment_area", 
            #                 shape_type="rectangle", edge_color="red", 
            #                 face_color="red", opacity=0.5)
            #     self.alignment_layer.metadata = {"type": "alignment"}
            #     self.alignment_layer.events.data.connect(self.update_alignment)



def convert_shape_to_image_area(shape: list[list[int]], image_shape: tuple, offset_shape: tuple = None) -> FibsemRectangle:
    """Convert a napari shape (rectangle) to  a FibsemRectangle expressed as a percentage of the image (reduced area)
    shape: the coordinates of the shape
    image_shape: the shape of the image (usually the ion beam image)
    offset_shape: the shape of the offset image (usually the electron beam image, as it translates to the ion beam image)
    
    """
    # get limits of rectangle
    y0, x0 = shape[0]
    y1, x1 = shape[2]

    # subtract shape of eb image
    if offset_shape:
        x0 -= offset_shape[1]
        x1 -= offset_shape[1]

    # convert to percentage of image
    x0 = x0 / image_shape[1]
    x1 = x1 / image_shape[1]
    y0 = y0 / image_shape[0]
    y1 = y1 / image_shape[0]
    w = x1 - x0
    h = y1 - y0

    reduced_area = FibsemRectangle(left=x0, top=y0, width=w, height=h)
    print("Reduced Area: ", reduced_area)

    return reduced_area

def is_valid_reduced_area(reduced_area: FibsemRectangle) -> bool:
    """Check whether the reduced area is valid. 
    Left and top must be between 0 and 1, and width and height must be between 0 and 1.
    Must not exceed the boundaries of the image 0 - 1
    """
    # if left or top is less than 0, or width or height is greater than 1, return False
    if reduced_area.left < 0 or reduced_area.top < 0 or reduced_area.width > 1 or reduced_area.height > 1:
        return False
    if reduced_area.left + reduced_area.width > 1 or reduced_area.top + reduced_area.height > 1:
        return False
    # no negative values
    if reduced_area.left < 0 or reduced_area.top < 0 or reduced_area.width < 0 or reduced_area.height < 0:
        return False
    return True



def main():
    viewer = napari.Viewer(ndisplay=2)
    fibsem_labelling_ui = FibsemBBoxLabellingUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_labelling_ui,
        area="right",
        add_vertical_stretch=True,
        name=f"OpenFIBSEM v{fibsem.__version__} - BBox Labelling",

    )
    napari.run()


if __name__ == "__main__":
    main()
