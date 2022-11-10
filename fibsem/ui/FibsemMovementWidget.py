import os


import fibsem
import napari
import napari.utils.notifications
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import calibration, constants, utils
from fibsem.structures import (
    BeamType,
    GammaSettings,
    ImageSettings,
    MicroscopeSettings
)
from fibsem.ui.qtdesigner_files import FibsemMovementWidget
from PyQt5 import QtWidgets

BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")

from enum import Enum

class MovementMode(Enum):
    Stable = 1
    Eucentric = 2

class FibsemMovementWidget(FibsemMovementWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemMovementWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer

        self.setup_connections()

    def setup_connections(self):

        # set ui elements
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.comboBox_movement_stage_coordinate_system.addItems(["Specimen", "Raw"])

        # buttons
        self.pushButton_move.clicked.connect(self.move_to_position)
        self.pushButton_save_position.clicked.connect(self.save_position)
        self.pushButton_continue.clicked.connect(self.continue_pressed)
        self.pushButton_auto_eucentric.clicked.connect(self.auto_eucentric_correction)

    def auto_eucentric_correction(self):

        print("auto eucentric")

    def continue_pressed(self):
        print("continue pressed")

    def save_position(self):
        print("save position pressed")

    def move_to_position(self):
        print("move_to position pressed")


def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMovementWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
