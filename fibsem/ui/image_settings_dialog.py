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
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.qtdesigner_files import ImageSettingsDialog
from PyQt5 import QtCore, QtWidgets



class GUIImageSettings(ImageSettingsDialog.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
    ):
        super(GUIImageSettings, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.setup_connections()


    def setup_connections(self):

        print("setup connections")

        self.comboBox_beam_type.addItems([beam.name for beam in BeamType])
        self.pushButton_accept.clicked.connect(self.accept_button_pressed)
        self.pushButton_cancel.clicked.connect(self.cancel_button_pressed)
        self.buttonBox.clicked.connect(self.accept_button_pressed)
    
    def accept_button_pressed(self):
        print(f"accept button pressed: {self.sender()}")

    def cancel_button_pressed(self):
        print("cancel button pressed")    

def main():
    app = QtWidgets.QApplication([])
    image_settings_ui = GUIImageSettings(
        parent=None
    )
    image_settings_ui.show()
    image_settings_ui.exec_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
