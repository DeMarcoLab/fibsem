#v2

import logging
import sys
from enum import Enum
import traceback

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import utils
from fibsem.ui.qtdesigner_files import FibsemUI2
from PyQt5 import QtCore, QtWidgets


import napari.utils.notifications
import napari

from pprint import pprint


from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemToolsWidget import FibsemToolsWidget
from fibsem.ui.FibsemSystemWidget import FibsemSystemWidget

class FibsemUI(FibsemUI2.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self, viewer: napari.Viewer, parent = None
    ):
        super(FibsemUI, self).__init__(parent = parent)
        self.setupUi(self)

        self.viewer = viewer

        self.viewer.grid.enabled = True
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
        self.viewer.axes.dashed = True

        # manually connect to microscope...
        self.microscope, self.settings = utils.setup_session()

        self.setup_connections()


    def setup_connections(self):

        # setup layout

        # reusable components
        self.image_widget = FibsemImageSettingsWidget(microscope = self.microscope, image_settings=self.settings.image, viewer=self.viewer)
        self.movement_widget = FibsemMovementWidget(microscope = self.microscope, settings=self.settings, image_widget=self.image_widget)
        self.tools_widget = FibsemToolsWidget()
        self.system_widget = FibsemSystemWidget()
        
        self.gridLayout_imaging.addWidget(self.image_widget, 0, 0)
        self.gridLayout_movement.addWidget(self.movement_widget, 0, 0)
        self.gridLayout_tools.addWidget(self.tools_widget, 0, 0)
        self.gridLayout_system.addWidget(self.system_widget, 0, 0)




def main():
    
    viewer = napari.Viewer()
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, area='right', add_vertical_stretch=False)  

    napari.run()


if __name__ == "__main__":
    main()
