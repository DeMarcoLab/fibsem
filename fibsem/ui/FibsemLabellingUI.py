import logging
import os
import sys

import fibsem
import napari
import napari.utils.notifications
import yaml
import glob
import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from PIL import Image
from fibsem import calibration, constants, utils
from fibsem.structures import (
    BeamSystemSettings,
    BeamType,
    MicroscopeSettings,
    StageSettings,
    SystemSettings,
)
from fibsem.ui.qtdesigner_files import FibsemLabellingUI
from PyQt5 import QtWidgets

BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")


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


        self.setup_connections()

    def setup_connections(self):

        print("setup connections")

        self.pushButton_load_data.clicked.connect(self.load_data)
        self.pushButton_next.clicked.connect(self.next_image)
        self.pushButton_previous.clicked.connect(self.previous_image)

    def load_data(self):

        print("load_data")

        # read raw data
        self.raw_path = self.lineEdit_raw_data.text()
        self.save_path = self.lineEdit_save_path.text()

        print(self.raw_path)
        print(self.save_path)

        # create required directories
        self.filenames = sorted(glob.glob(os.path.join(self.raw_path, "*.tif*")))

        # initialise viewer layers
        img = Image.open(self.filenames[0])
        self.viewer.layers.clear()
        self.viewer.add_image(img, name="img")
        self.viewer.add_labels(np.zeros_like(img, dtype=np.uint8), name="Labels")

    def next_image(self):

        print("next image")

        # save current image
        fname = str(self.idx).zfill(9)
        
        self.viewer.layers["img"].save(os.path.join(self.save_path, "images", f"{fname}.tif"))
        label = self.viewer.layers["Labels"].data.astype(np.uint8)

        im = Image.fromarray(label) 
        im.save(os.path.join(self.save_path, "labels", f"{fname}.tif"))  # or 'test.tif'

        # advance index
        self.idx += 1

        # update / clear viewer
        img = Image.open(self.filenames[self.idx])
        self.viewer.layers.clear()
        self.viewer.add_image(img, name="img")
        self.viewer.add_labels(np.zeros_like(img, dtype=np.uint8), name="Labels")


    def previous_image(self):

        print("previous_image")

        # save current image
        fname = str(self.idx).zfill(9)
        
        self.viewer.layers["img"].save(os.path.join(self.save_path, "images", f"{fname}.tif"))
        label = self.viewer.layers["Labels"].data.astype(np.uint8)

        im = Image.fromarray(label) 
        im.save(os.path.join(self.save_path, "labels", f"{fname}.tif"))  # or 'test.tif'

        # decrement index
        self.idx -= 1

        # update / clear viewer
        img = Image.open(self.filenames[self.idx])
        self.viewer.layers.clear()
        self.viewer.add_image(img, name="img")
        self.viewer.add_labels(np.zeros_like(img, dtype=np.uint8), name="Labels")




    #TODO: port functionality

def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_labelling_ui = FibsemLabellingUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_labelling_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
