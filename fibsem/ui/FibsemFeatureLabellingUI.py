import napari

import logging

import fibsem
from fibsem.ui.qtdesigner_files import FibsemFeatureDetectionUI as FibsemFeatureLabellingUI
from fibsem.ui.utils import (
    _get_directory_ui,
    _get_file_ui,
    _get_text_ui,
    _get_save_file_ui,
)

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import glob
import os

import napari.utils.notifications
from fibsem.detection.detection import __FEATURES__, get_feature

from napari.layers import Points
import numpy as np
import os
import glob
import pandas as pd

from typing import Optional, Any
import copy
from fibsem.structures import Point
from fibsem.ui import _stylesheets

logging.basicConfig(level=logging.INFO)


def create_dataframe(filenames, method="null"):
    df = pd.DataFrame(
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
    df["filename"] = filenames
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(x))
    df["path"] = df["filename"].apply(lambda x: os.path.dirname(x))
    df["method"] = method
    return df


# TODO: remove this debug path
TEST_PATH_DEBUG = "/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/test"

# new line char in html
NL = "<br>"

DATA_LOADED_INSTRUCTIONS = f"""
<h3>Instructions</h3>
Use the Slider to Change Images{NL}{NL}

<strong>Add Mode</strong>{NL}
Select Feature from Dropdown{NL}
Left Click to Add Feature{NL}{NL}

<strong>Edit Mode</strong>{NL}
Press Edit to enter Edit Mode{NL}
Select and Drag the Feature to move it.{NL}
Select a Feature and pPress Delete Key{NL}{NL}

<strong>HotKeys</strong>{NL}
Press 2 to enter Add Mode{NL}
Press 3 to enter Edit Mode{NL}
Press 4 to enter Pan/Zoom Mode{NL}
Press Delete in Edit Mode to Delete Feature{NL}{NL}

<strong>Save</strong>{NL}
Data is automatically saved to csv file when a feature is added, edited or deleted.
"""
START_INSTRUCTIONS = f"""
<h3>Instructions</h3>
Load Data from File Menu{NL}{NL}

<strong>Image Directory</strong>{NL}
If starting from scratch, load a directory of images and a csv file will be created for you.
Features will be saved to the csv file.{NL}{NL}

<strong>CSV File</strong>{NL}
If you are resuming work, load the csv file and the images will be loaded for you.
Features will be reloaded from the csv file.{NL}{NL}
"""

INSTRUCTIONS = {"START": START_INSTRUCTIONS, "DATA_LOADED": DATA_LOADED_INSTRUCTIONS}

FEATURES_TO_IGNORE = ["ImageCentre", "CoreFeature"]
USABLE_FEATURES = sorted([f.name for f in __FEATURES__ if f.name not in FEATURES_TO_IGNORE])

CONFIGURATION = {
    
    "FEATURES": USABLE_FEATURES,

    "UI": {
        "IMAGE_LAYER_NAME": "Image",
        "POINT_LAYER_NAME": "Features",
        "SHAPE_LAYER_NAME": "Bounding Boxes",
        "POINT_SIZE": 5,
        "TEXT_COLOR": "white",
        "TEXT_TRANSLATION": np.array([-30, 0]),
        "FACE_COLOR": "white",
        "ADD_BUTTON_COLOR": _stylesheets._GREEN_PUSHBUTTON_STYLE,
        "EDIT_BUTTON_COLOR": _stylesheets._BLUE_PUSHBUTTON_STYLE,
    },

}


# TODO:
# add support for custom features
# import features from txt file
# import features from yaml file (with color map)


def get_feature_color(feature: str) -> str:
    """try to get the feature color 
    from the feature name, with fallback 
    to base colour"""

    try:
        return get_feature(feature).color # only for fibsem features
    except:
        return CONFIGURATION["UI"]["FACE_COLOR"]


class FibsemFeatureLabellingUI(FibsemFeatureLabellingUI.Ui_MainWindow, QtWidgets.QMainWindow
):
    _minimap_signal = pyqtSignal(object)

    def __init__(self, viewer: napari.Viewer):
        super(FibsemFeatureLabellingUI, self).__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.img_layer = None
        self.pts_layer = None

        self.feature = None
        self.features: list[dict] = []

        self.setup_connections()

    def setup_connections(self):
        self.actionLoad_Image_Directory.triggered.connect(self.load_image_directory)
        self.actionLoad_CSV.triggered.connect(self.load_csv)

        # controls
        self.spinBox_data_progress.valueChanged.connect(self._change_image)
        self.spinBox_data_progress.setKeyboardTracking(False)

        self.comboBox_features.addItems(CONFIGURATION["FEATURES"])
        self.comboBox_features.currentIndexChanged.connect(self._get_feature)
        self.pushButton_add_feature.clicked.connect(self._add_feature_mode)
        self.pushButton_edit_feature.clicked.connect(self._edit_feature_mode)

        self.pushButton_add_feature.setStyleSheet(CONFIGURATION["UI"]["ADD_BUTTON_COLOR"])
        self.pushButton_edit_feature.setStyleSheet(CONFIGURATION["UI"]["EDIT_BUTTON_COLOR"])

        self.update_ui_elements()

    def update_ui_elements(self):
        _data_loaded = self.img_layer is not None

        self.label_features_header.setVisible(_data_loaded)
        self.pushButton_add_feature.setVisible(_data_loaded)
        self.pushButton_edit_feature.setVisible(_data_loaded)
        self.comboBox_features.setVisible(_data_loaded)
        self.label_data_progress.setVisible(_data_loaded)
        self.spinBox_data_progress.setVisible(_data_loaded)

        self.spinBox_data_progress.setEnabled(_data_loaded)

        if _data_loaded:
            self.label_data_progress.setText(
                f"Current Image (Total: {len(self.filenames)})"
            )
            self.spinBox_data_progress.setMaximum(len(self.filenames) - 1)
            self.spinBox_data_progress.setValue(0)

            instructions = INSTRUCTIONS["DATA_LOADED"]
        else:
            instructions = INSTRUCTIONS["START"]
        self.label_instructions.setText(instructions)

    def load_image_directory(self):
        """Load a directory of images into napari"""
        print(f"Loading image directory")

        path = _get_directory_ui(
            msg="Select image directory (*.tif)", parent=self, path=TEST_PATH_DEBUG
        )

        if path is None:
            return

        filenames = sorted(glob.glob(os.path.join(path, "*.tif*")))

        if len(filenames) == 0:
            napari.utils.notifications.show_warning(
                f"No images found in {path}. Exiting"
            )
            return

        method, ret = _get_text_ui(
            msg="Enter method name",
            title="Dataset Method",
            default="autolamella",
            parent=self,
        )

        if ret == False:
            napari.utils.notifications.show_warning(f"No method name entered. Exiting.")
            return

        # create dataframe
        df = create_dataframe(filenames, method=method)

        # save dataframe
        csv_path = _get_save_file_ui(
            msg="Save CSV file",
            path=os.path.join(path, "data.csv"),
            _filter="CSV (*.csv)",
            parent=self,
        )
        if csv_path is None:
            napari.utils.notifications.show_warning(f"No CSV file selected. Exiting.")
            return

        df.to_csv(csv_path, index=False)

        # load data
        self.load_data(df, filenames, path, csv_path)

    def load_csv(self):
        """Load a CSV file into napari"""

        # get csv path from user
        csv_path = _get_file_ui(
            msg="Select CSV file",
            _filter="CSV (*.csv)",
            path=TEST_PATH_DEBUG,
            parent=self,
        )

        if csv_path is None:
            napari.utils.notifications.show_warning(f"No CSV file selected. Exiting.")
            return

        df = pd.read_csv(csv_path)

        # sort on filenames
        df.sort_values(by=["filename"], inplace=True)

        # get image path from user
        path = _get_directory_ui(
            msg="Select image directory (*.tif)",
            parent=self,
            path=os.path.dirname(csv_path),
        )

        if path is None:
            napari.utils.notifications.show_warning(
                f"No image directory selected. Exiting."
            )
            return

        filenames = []
        for fname in pd.unique(df["filename"].values):
            filenames += glob.glob(os.path.join(path, f"*{fname}*"))

        if len(filenames) == 0:
            napari.utils.notifications.show_warning(
                f"No images found in {path}. Exiting"
            )
            return

        self.load_data(df, filenames, path, csv_path)

    def load_data(
        self, df: pd.DataFrame, filenames: list[str], path: str, csv_path: str
    ):
        # set attributes
        self.filenames = filenames
        self.path = path
        self.csv_path = csv_path
        self.df = df

        # load filenames as single layer
        self.img_layer = self.viewer.open(filenames, stack=True)[0]

        # subscribe to dims events
        self.viewer.dims.events.current_step.connect(self._change_image)

        # load empty points layer
        self.pts_layer = self.viewer.add_points(
            name=CONFIGURATION["UI"]["POINT_LAYER_NAME"], 
            face_color=CONFIGURATION["UI"]["FACE_COLOR"], 
            size=CONFIGURATION["UI"]["POINT_SIZE"]
        )

        self.pts_layer.events.data.connect(self._features_updated)

        # update viewer
        self.update_viewer_to_image(0)

        # metadata
        self.img_layer.name = CONFIGURATION["UI"]["IMAGE_LAYER_NAME"]
        self.img_layer.metadata["path"] = path
        self.img_layer.metadata["csv_path"] = csv_path

        logging.info(f"Loaded {len(filenames)} images from {path}")
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
        napari.utils.notifications.show_info(
            f"Loaded {len(filenames)} images from {path}"
        )

        self.update_ui_elements()

    def update_viewer_to_image(self, idx: int):
        self.viewer.dims.set_point(0, idx)

    def _change_image(self):
        """Change the currently selected image"""
        if self.sender() == self.spinBox_data_progress:
            idx = self.spinBox_data_progress.value()
            self.update_viewer_to_image(idx)
        else:
            idx = int(self.viewer.dims.point[0])
            self.spinBox_data_progress.blockSignals(True)
            self.spinBox_data_progress.setValue(idx)
            self.spinBox_data_progress.blockSignals(False)

        # clear features / points
        self.features = []
        self.feature = None
        self.pts_layer.data = []
        self._add_feature_mode()

        # update data from dataframe
        self.update_points_from_dataframe(idx)

    def _get_feature(self):
        """Get the currently selected feature"""
        self.feature = {"name": self.comboBox_features.currentText()}
        return self.feature

    def _add_feature_mode(self):
        """Add a feature to the currently selected image"""
        self.viewer.layers.selection.active = self.pts_layer
        self.pts_layer.mode = "add"

    def _edit_feature_mode(self):
        """Edit the currently selected feature"""
        self.viewer.layers.selection.active = self.pts_layer
        self.pts_layer.mode = "select"

    def _features_updated(self, event: Optional[Any] = None) -> None:
        if self.feature is None:
            self._get_feature()

        if self.feature is None:
            napari.utils.notifications.show_warning(f"No feature selected. Exiting.")
            return

        # get latest points
        points = self.pts_layer.data

        # get which point was moved
        index: list[int] = list(self.pts_layer.selected_data)

        def napari_pt_to_point(pt: np.ndarray, dtype=int) -> Point:
            return Point.from_list(np.flip(pt, axis=-1).astype(dtype))

        # add new feature
        if len(points) > len(self.features):
            # get latest feature
            feature = copy.deepcopy(self.feature)
            feature.update(napari_pt_to_point(points[-1]).to_dict())
            feature["color"] = get_feature_color(feature["name"])
            self.features.append(copy.deepcopy(feature))

        # feature was deleted
        elif len(points) < len(self.features):
            # deleted

            for idx in index[::-1]:
                logging.info(f"point deleted: {self.features[idx]['name']}")
                self.features.pop(idx)

        # feature was moved
        else:
            # moved
            for idx in index:
                self.features[idx].update(
                    copy.deepcopy(napari_pt_to_point(points[idx])).to_dict()
                )

        #
        text = {
            "string": [f["name"] for f in self.features],
            "color": CONFIGURATION["UI"]["TEXT_COLOR"],
            "translation": CONFIGURATION["UI"]["TEXT_TRANSLATION"],
        }

        self.pts_layer.text = text
        self.pts_layer.face_color = [f["color"] for f in self.features]

        # update dataframe, write to csv
        self.update_dataframe()

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


def main():
    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemFeatureLabellingUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_ui,
        area="right",
        add_vertical_stretch=True,
        name=f"OpenFIBSEM v{fibsem.__version__} - Keypoint Labelling",
    )
    napari.run()


if __name__ == "__main__":
    main()
