
import napari

import logging

import fibsem
from fibsem import config as cfg
from fibsem.microscope import (DemoMicroscope, FibsemMicroscope,
                               MicroscopeSettings, TescanMicroscope,
                               ThermoMicroscope)
from fibsem.structures import BeamType

from fibsem.ui.qtdesigner_files import FibsemFeatureDetectionUI
from fibsem.ui.utils import message_box_ui, _get_directory_ui, _get_file_ui, _get_text_ui, _get_save_file_ui

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal
import glob
import os

import napari.utils.notifications

logging.basicConfig(level=logging.INFO)




import os
import glob
import pandas as pd


def create_dataframe(filenames, method="null"):
    df = pd.DataFrame(columns=["filename", "feature", "px.x", "px.y", "pixelsize", "corrected", "method", "experiment"])
    df["filename"] = filenames
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(x))
    df["path"] = df["filename"].apply(lambda x: os.path.dirname(x))
    df["method"] = method
    return df

# TODO: remove this debug path
TEST_PATH_DEBUG =  "/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/test"


class FibsemFeatureDetectionUI(FibsemFeatureDetectionUI.Ui_MainWindow, QtWidgets.QMainWindow):
    _minimap_signal = pyqtSignal(object)

    def __init__(self, viewer: napari.Viewer):
        super(FibsemFeatureDetectionUI, self).__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.img_layer = None

        self.setup_connections()

    def setup_connections(self):

        self.actionLoad_Image_Directory.triggered.connect(self.load_image_directory)
        self.actionLoad_CSV.triggered.connect(self.load_csv)


        # controls
        self.spinBox_data_progress.valueChanged.connect(self._change_image)
        self.spinBox_data_progress.setKeyboardTracking(False)

        from fibsem.detection.detection import __FEATURES__

        self.comboBox_features.addItems(f.name for f in __FEATURES__)

    def update_ui_elements(self):

        _data_loaded = self.img_layer is not None
        self.spinBox_data_progress.setEnabled(_data_loaded)
        
        if _data_loaded:
            self.label_data_progress.setText(f"Current Image (Total: {len(self.filenames)})")
            self.spinBox_data_progress.setMaximum(len(self.filenames)-1)
            self.spinBox_data_progress.setValue(0)

    def load_image_directory(self):
        """Load a directory of images into napari"""
        print(f"Loading image directory")


        path = _get_directory_ui(msg="Select image directory (*.tif)", parent=self, 
                                 path=TEST_PATH_DEBUG)

        if path is None:
            return
        
        filenames = sorted(glob.glob(os.path.join(path, "*.tif*")))

        if len(filenames) == 0:
            napari.utils.notifications.show_warning(f"No images found in {path}. Exiting")
            return

        method, ret = _get_text_ui(msg="Enter method name", 
                            title="Dataset Method", 
                            default="autolamella", parent=self)

        if ret == False:
            napari.utils.notifications.show_warning(f"No method name entered. Exiting.")
            return
        
        # create dataframe
        df = create_dataframe(filenames, method=method)

        # save dataframe
        csv_path = _get_save_file_ui(msg="Save CSV file", 
                                     path=os.path.join(path, "data.csv"),
                                     _filter="CSV (*.csv)", 
                                     parent=self)
        if csv_path is None:
            napari.utils.notifications.show_warning(f"No CSV file selected. Exiting.")
            return
        
        df.to_csv(csv_path, index=False)

        # load data
        self.load_data(df, filenames, path, csv_path)

    def load_csv(self):
        """Load a CSV file into napari"""

        # get csv path from user   
        csv_path = _get_file_ui(msg="Select CSV file", 
                                _filter="CSV (*.csv)", 
                                path=TEST_PATH_DEBUG,
                                parent=self)
        
        if csv_path is None:
            napari.utils.notifications.show_warning(f"No CSV file selected. Exiting.")
            return
        
        df = pd.read_csv(csv_path)

        # get image path from user
        path = _get_directory_ui(msg="Select image directory (*.tif)", parent=self, 
                                 path=os.path.dirname(csv_path))

        if path is None:
            napari.utils.notifications.show_warning(f"No image directory selected. Exiting.")
            return

        filenames = []
        for fname in df["filename"].values:
            filenames += glob.glob(os.path.join(path, f"*{fname}*"))
        
        if len(filenames) == 0:
            napari.utils.notifications.show_warning(f"No images found in {path}. Exiting")
            return

        self.load_data(df, filenames, path, csv_path)


    def load_data(self, df: pd.DataFrame, filenames: list[str], path: str, csv_path: str):
        
        # load filenames as single layer
        self.img_layer = self.viewer.open(filenames, stack=True)[0]
        self.update_viewer_to_image(0)

        # metadata
        self.img_layer.name = "Image"
        self.img_layer.metadata["path"] = path
        self.img_layer.metadata["csv_path"] = csv_path
        self.img_layer.metadata["df"] = df

        # set attributes
        self.filenames = filenames
        self.path = path
        self.csv_path = csv_path
        self.df = df

        logging.info(f"Loaded {len(filenames)} images from {path}")
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
        napari.utils.notifications.show_info(f"Loaded {len(filenames)} images from {path}")

        self.update_ui_elements()

        
    def update_viewer_to_image(self, idx: int):
        self.viewer.dims.set_point(0, idx)

    def _change_image(self):
        """Change the currently selected image"""
        idx = self.spinBox_data_progress.value()
        self.update_viewer_to_image(idx)
        

    def _get_feature(self):
        """Get the currently selected feature"""
        return self.comboBox_features.currentText()
    


def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemFeatureDetectionUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, 
                                  area="right", 
                                  add_vertical_stretch=True, 
                                  name=f"OpenFIBSEM v{fibsem.__version__} - Feature Detection")
    napari.run()


if __name__ == "__main__":
    main()
