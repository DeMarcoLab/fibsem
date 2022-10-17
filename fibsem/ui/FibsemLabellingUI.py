import logging
import os
import sys

import fibsem
import napari
import napari.utils.notifications
import yaml
from autoscript_sdb_microscope_client import SdbMicroscopeClient
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
        save_path = self.lineEdit_save_path.text()

        print(raw_path)
        print(save_path)

        # create required directories

        # initialise viewer layers


    def next_image(self):

        print("next image")

        # save current image

        # advance index

        # update / clear viewer
        import numpy as np
        img = np.random.random(size=(500, 1000))
        self.viewer.layers.clear()
        self.viewer.add_image(img, name="img")
        self.viewer.add_labels(np.zeros_like(img, dtype=np.uint8), name="Labels")


    def previous_image(self):

        print("previous_image")

        # save current image

        # decrement index

        # update / clear viewer





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
