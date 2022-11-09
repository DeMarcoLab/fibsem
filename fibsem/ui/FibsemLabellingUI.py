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

# BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")


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

        # get filenames
        self.filenames = sorted(glob.glob(os.path.join(raw_path, "*.tif*")))
        
        # create required directories        
        os.makedirs(os.path.join(self.save_path, "labels"), exist_ok=True)
        
        self.update_image()


    def next_image(self):

        self.save_image()

        # advance index
        self.idx +=1
        self.idx = np.clip(self.idx, 0, len(self.filenames)-1)

        self.update_image()

    def previous_image(self):

        self.save_image()

        # decrement index
        self.idx -= 1
        self.idx = np.clip(self.idx, 0, len(self.filenames)-1)
        
        self.update_image()

    def save_image(self):

        # save current image
        bname = os.path.basename(self.fname).split(".")[0]
        
        # only resave the labels...
        label = self.viewer.layers["Labels"].data.astype(np.uint8)

        im = Image.fromarray(label) 
        im.save(os.path.join(self.save_path, "labels", f"{bname}.tif"))  # or 'test.tif'
        im.save(os.path.join(self.save_path, "labels", f"{bname}.png")) # TODO: convert to RBG
        
    def update_image(self):
        
        # update progress text
        self.label_progress.setText(f"{self.idx + 1}/{len(self.filenames)}")        

        # update / clear viewer
        self.viewer.layers.clear()
        self.fname = self.filenames[self.idx]
        self.img = tff.imread(self.fname)
        self.viewer.add_image(self.img, name="img")

        label_image = self.get_label_image()
        self.viewer.add_labels(label_image, name="Labels")

        # disable buttons
        IS_NOT_FIRST_INDEX = bool(self.idx != 0)
        IS_NOT_LAST_INDEX = bool(self.idx != int(len(self.filenames)-1))
        self.pushButton_previous.setEnabled(IS_NOT_FIRST_INDEX)
        self.pushButton_next.setEnabled(IS_NOT_LAST_INDEX)

    def get_label_image(self) -> np.ndarray:

        if os.path.basename(self.fname) in os.listdir(os.path.join(self.save_path, "labels")): 
            
            label_fname = os.path.join(self.save_path, "labels", os.path.basename(self.fname))
            label_image = tff.imread(label_fname)
            label_image = np.array(label_image, dtype=np.uint8)
        else:
            label_image = np.zeros_like(self.img)

        return label_image

    def closeEvent(self, event):
        # try to save the current image on close
        try:
            self.save_image()
        except: 
            pass
        event.accept()

    # TODO: remove use of PIl, use tf to save
    # BUG: no way to save the last image in the dataset? except go back?

    # TODO: go to index
    # TODO: hotkeys
    # 

def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_labelling_ui = FibsemLabellingUI(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_labelling_ui, area="right", add_vertical_stretch=False
    )
    napari.run()

if __name__ == "__main__":
    main()
