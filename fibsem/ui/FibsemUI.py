#v2



import logging
import sys
from enum import Enum
import traceback

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment, utils, milling, calibration
from fibsem.structures import BeamType, MicroscopeSettings, MillingSettings, Point
from fibsem.ui.qtdesigner_files import FibsemUI2
from PyQt5 import QtCore, QtWidgets


import napari.utils.notifications
import napari

from pprint import pprint

from fibsem.ui import utils as ui_utils
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

        self.setup_connections()


    def setup_connections(self):

        print("setup_connections")

        # setup layout

        # reusable components
        self.image_widget = FibsemImageSettingsWidget(viewer=self.viewer)
        self.movement_widget = FibsemMovementWidget()
        self.tools_widget = FibsemToolsWidget()
        self.system_widget = FibsemSystemWidget()
        
        self.gridLayout_imaging.addWidget(self.image_widget, 0, 0)
        self.gridLayout_movement.addWidget(self.movement_widget, 0, 0)
        self.gridLayout_tools.addWidget(self.tools_widget, 0, 0)
        self.gridLayout_system.addWidget(self.system_widget, 0, 0)

        self.eb_image = np.zeros(shape=(1024, 1536))
        self.ib_image = np.ones(shape=(1024, 1536))

        self.ib_layer = self.viewer.add_image(self.ib_image, name=BeamType.ION.name)
        self.eb_layer = self.viewer.add_image(self.eb_image, name=BeamType.ELECTRON.name)

        self.eb_layer.mouse_double_click_callbacks.append(self._double_click)
        self.ib_layer.mouse_double_click_callbacks.append(self._double_click)


    def _double_click(self, layer, event):

        coords = layer.world_to_data(event.position)

        # get the relative coords in the beam image
        coords, beam_type = ui_utils.get_beam_coords_from_click(coords, self.eb_image)

        if beam_type is None:
            napari.utils.notifications.show_info(f"Please click inside image to move.")
            return 

        if beam_type is BeamType.ELECTRON:
            adorned_image = self.eb_image
        if beam_type is BeamType.ION:
            adorned_image = self.ib_image

        print(f"beam_type: {beam_type}, coords: {coords}")

        # move

        # take new images

def main():
    
    viewer = napari.Viewer()
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, area='right')  

    napari.run()


if __name__ == "__main__":
    main()
