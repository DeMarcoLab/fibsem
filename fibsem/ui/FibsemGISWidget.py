import logging

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets

from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (MicroscopeSettings)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemGISWidget


class FibsemGISWidget(FibsemGISWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        image_widget: FibsemImageSettingsWidget = None, 
        parent=None,
    ):
        super(FibsemGISWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget

        self.setup_connections()

        self.update_ui()

    def setup_connections(self):
        pass
    
    def update_ui(self):
        pass

def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemGISWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()