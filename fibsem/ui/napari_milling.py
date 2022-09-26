import logging
import sys
from enum import Enum

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment
from fibsem.structures import BeamType, MicroscopeSettings
from fibsem.ui.qtdesigner_files import NapariMilling
from PyQt5 import QtCore, QtWidgets

import napari

# TODO: maybe have to change to dialog?


class NapariMillingUI(NapariMilling.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self, viewer: napari.Viewer, parent = None
    ):
        super(NapariMillingUI, self).__init__(parent = parent)
        self.setupUi(self)

        self.setup_connections()

        self.viewer = viewer
    
    def setup_connections(self):
        print("setup connections")


        self.pushButton_run_milling.clicked.connect(self.run_milling)


    def run_milling(self):

        print("run milling")

        print(f"Shapes: {self.viewer.layers['Shapes'].shape_type}")
        print(f"Data: {self.viewer.layers['Shapes'].data}")


# TODO: override enter


def main():
    # from liftout import utils
    # from fibsem.ui import windows as fibsem_ui_windows
    # microscope, settings= utils.quick_setup()


    app = QtWidgets.QApplication([])
    viewer = napari.Viewer(ndisplay=2)
    napari_milling_ui = NapariMillingUI(viewer=viewer)
    viewer.window.add_dock_widget(napari_milling_ui, area='right')  

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
