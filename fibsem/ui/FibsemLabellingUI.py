import logging
import os
import sys
import glob
import fibsem
import napari
import napari.utils.notifications
import numpy as np
import tifffile as tff
import zarr
from PIL import Image
""" import yaml
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import calibration, constants, utils
from fibsem.structures import (
    BeamSystemSettings,
    BeamType,
    MicroscopeSettings,
    StageSettings,
    SystemSettings,
) """
from qtdesigner_files import FibsemLabellingUI
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
        raw_path = self.lineEdit_raw_data.text()
        self.save_path = self.lineEdit_save_path.text()

        print(raw_path)
        print(self.save_path)

        vol = tff.imread(os.path.join(raw_path, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
        self.zarr_set = zarr.open(vol)
        self.filenames = sorted(glob.glob(os.path.join(raw_path, "*.tif*")))
        # create required directories        
        os.makedirs(os.path.join(self.save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "labels"), exist_ok=True)
        # initialise viewer layers
        self.img = self.zarr_set[self.idx]
        self.fname = self.filenames[self.idx]
        self.viewer.add_image(self.img, name="img")
        self.viewer.add_labels(np.zeros_like(self.img), name="Labels")

    def next_image(self):

        print("next image")

        # save current image
        bname = os.path.basename(self.fname).split(".")[0]
        
        self.viewer.layers["img"].save(os.path.join(self.save_path, "images", f"{bname}.tif"))
        label = self.viewer.layers["Labels"].data.astype(np.uint8)

        im = Image.fromarray(label) 
        im.save(os.path.join(self.save_path, "labels", f"{bname}.tif"))  # or 'test.tif'
        # advance index
        self.idx = self.idx+1
        # update / clear viewer
        self.viewer.layers.clear()
        self.img = self.zarr_set[self.idx]
        self.fname = self.filenames[self.idx]
        self.viewer.add_image(self.img, name="img")
        self.viewer.add_labels(np.zeros_like(self.img), name="Labels")


    def previous_image(self):

        print("previous_image")

        # save current image
        bname = os.path.basename(self.fname).split(".")[0]
        
        self.viewer.layers["img"].save(os.path.join(self.save_path, "images", f"{bname}.tif"))
        label = self.viewer.layers["Labels"].data.astype(np.uint8)

        im = Image.fromarray(label) 
        im.save(os.path.join(self.save_path, "labels", f"{bname}.tif"))  # or 'test.tif'
        # advance index
        self.idx = self.idx-1
        # update / clear viewer
        self.viewer.layers.clear()
        self.img = self.zarr_set[self.idx]
        self.fname = self.filenames[self.idx]
        self.viewer.add_image(self.img, name="img")
        self.viewer.add_labels(np.zeros_like(self.img), name="Labels")





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
