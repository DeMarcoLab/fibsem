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
        
        self.update_image()


    def next_image(self):

        print("next image")

        self.save_image()

        # advance index
        self.idx +=1
        self.idx = np.clip(self.idx, 0, len(self.filenames)-1)

        self.update_image()

    def previous_image(self):

        print("previous_image")

        self.save_image()

        # decrement index
        self.idx -= 1
        self.idx = np.clip(self.idx, 0, len(self.filenames)-1)
        
        self.update_image()

    def save_image(self):

        # save current image
        bname = os.path.basename(self.fname).split(".")[0]
        
        self.viewer.layers["img"].save(os.path.join(self.save_path, "images", f"{bname}.tif"))
        label = self.viewer.layers["Labels"].data.astype(np.uint8)

        im = Image.fromarray(label) 
        im.save(os.path.join(self.save_path, "labels", f"{bname}.tif"))  # or 'test.tif'
        
    def update_image(self):
        
        # update progress text
        self.label_progress.setText(f"{self.idx + 1}/{len(self.filenames)}")        

        # update / clear viewer
        self.viewer.layers.clear()
        self.img = self.zarr_set[self.idx]
        self.fname = self.filenames[self.idx]
        self.viewer.add_image(self.img, name="img")

        label_image = self.get_label_image()
        self.viewer.add_labels(label_image, name="Labels")

        # disable buttons
        IS_NOT_FIRST_INDEX = bool(self.idx != 0)
        IS_NOT_LAST_INDEX = bool(self.idx != int(len(self.filenames)-1))
        self.pushButton_previous.setEnabled(IS_NOT_FIRST_INDEX)
        self.pushButton_next.setEnabled(IS_NOT_LAST_INDEX)

    def get_label_image(self) -> np.ndarray:

        if os.path.basename(self.fname).split(".")[0] in os.listdir(os.path.join(self.save_path, "images")): 
            label_image = tff.imread(os.path.join(self.save_path, "labels", self.fname))
        else:
            label_image = np.zeros_like(self.img)

        return label_image

    # TODO: port functionality
    # TODO: show existing labels if exist
    # TODO: remove use of PIl, use tf to save

def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_labelling_ui = FibsemLabellingUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_labelling_ui, area="right", add_vertical_stretch=False
    )
    napari.run()

if __name__ == "__main__":
    main()
