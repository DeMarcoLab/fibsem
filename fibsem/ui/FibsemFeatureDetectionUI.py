import napari

import logging

import fibsem
from fibsem.ui.qtdesigner_files import FibsemFeatureDetectionUI
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


class FibsemFeatureDetectionUI(
    FibsemFeatureDetectionUI.Ui_MainWindow, QtWidgets.QMainWindow
):
    _minimap_signal = pyqtSignal(object)

    def __init__(self, viewer: napari.Viewer):
        super(FibsemFeatureDetectionUI, self).__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.img_layer = None
        self.pts_layer = None

        self.feature = None
        self.features = []

        self.setup_connections()

    def setup_connections(self):
        self.actionLoad_Image_Directory.triggered.connect(self.load_image_directory)
        self.actionLoad_CSV.triggered.connect(self.load_csv)

        # controls
        self.spinBox_data_progress.valueChanged.connect(self._change_image)
        self.spinBox_data_progress.setKeyboardTracking(False)

        self.comboBox_features.addItems(sorted([f.name for f in __FEATURES__]))
        self.comboBox_features.currentIndexChanged.connect(self._get_feature)
        self.pushButton_add_feature.clicked.connect(self._add_feature_mode)
        self.pushButton_edit_feature.clicked.connect(self._edit_feature_mode)

        self.pushButton_add_feature.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_edit_feature.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)

    def update_ui_elements(self):
        _data_loaded = self.img_layer is not None
        self.spinBox_data_progress.setEnabled(_data_loaded)

        if _data_loaded:
            self.label_data_progress.setText(
                f"Current Image (Total: {len(self.filenames)})"
            )
            self.spinBox_data_progress.setMaximum(len(self.filenames) - 1)
            self.spinBox_data_progress.setValue(0)

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

        df.to_csv(csv_path, index=True)

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
        for fname in df["filename"].values:
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
            name="Features", face_color="white", size=20
        )

        self.pts_layer.events.data.connect(self._features_updated)

        # update viewer
        self.update_viewer_to_image(0)

        # metadata
        self.img_layer.name = "Image"
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
        self.feature = get_feature(self.comboBox_features.currentText())
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
            return Point.__from_list__(np.flip(pt, axis=-1).astype(dtype))

        # add new feature
        if len(points) > len(self.features):
            # get latest feature
            feature = copy.deepcopy(self.feature)
            feature.px = napari_pt_to_point(points[-1])

            self.features.append(feature)

        # feature was deleted
        elif len(points) < len(self.features):
            # deleted

            for idx in index[::-1]:
                logging.info(f"point deleted: {self.features[idx].name}")
                self.features.pop(idx)

        # feature was moved
        else:
            # moved
            for idx in index:
                self.features[idx].px = napari_pt_to_point(points[idx])

        #
        text = {
            "string": [f.name for f in self.features],
            "color": "white",
            "translation": np.array([-30, 0]),
        }

        self.pts_layer.text = text
        self.pts_layer.face_color = [f.color for f in self.features]

        # update dataframe, write to csv
        self.update_dataframe()

    def update_dataframe(self):
        """Update the dataframe with the latest features"""
        idx = int(self.viewer.dims.point[0])
        fname = self.filenames[idx]

        df: pd.DataFrame = self.df
        # get features
        features = self.features

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
        df_new["feature"] = [f.name for f in features]
        df_new["px.x"] = [f.px.x for f in features]
        df_new["px.y"] = [f.px.y for f in features]
        df_new["pixelsize"] = [None] * len(features)  # pixel size is not known
        df_new["corrected"] = [True] * len(features)
        df_new["method"] = [df_filt["method"].values[0]] * len(features)
        df_new["experiment"] = [df_filt["experiment"].values[0]] * len(features)

        # update dataframe
        df.drop(df_filt.index, inplace=True)
        df = pd.concat([df, df_new], axis=0)
        df.reset_index(drop=True, inplace=True) # mind your index

        # save dataframe
        df.to_csv(self.csv_path.replace("data.csv", "data-new.csv"), index=True)

        # update dataframe
        self.df = df

    def update_points_from_dataframe(self, idx: int):
        
        print(f"Updating points from dataframe: {idx}")

        # get dataframe for current image
        fname = self.filenames[idx]
        df_filt = self.df[self.df["filename"] == os.path.basename(fname)]

        # initial empty df, then no features were added
        if len(df_filt) == 1 and pd.isna(df_filt["feature"].values[0]):
            napari.utils.notifications.show_info(f"No features yet for {os.path.basename(fname)}")
            return

        # load features from dataframe
        for _, row in df_filt.iterrows():
            x, y = row["px.x"], row["px.y"]
            name = row["feature"]

            feature = get_feature(name)
            feature.px = Point(x, y)
            self.features.append(feature)

        # update points layer
        self.pts_layer.data = np.array([[f.px.y, f.px.x] for f in self.features])

        # update text
        text = {
            "string": [f.name for f in self.features],
            "color": "white",
            "translation": np.array([-30, 0]),
        }

        self.pts_layer.text = text
        self.pts_layer.face_color = [f.color for f in self.features]


def main():
    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemFeatureDetectionUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_ui,
        area="right",
        add_vertical_stretch=True,
        name=f"OpenFIBSEM v{fibsem.__version__} - Feature Detection",
    )
    napari.run()


if __name__ == "__main__":
    main()
