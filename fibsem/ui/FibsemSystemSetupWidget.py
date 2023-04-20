import logging
import traceback

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMessageBox


from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamType, FibsemStagePosition,
                               MicroscopeSettings, MovementMode, Point)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemSystemSetupWidget
from fibsem import utils
from fibsem.ui.utils import message_box_ui


class FibsemSystemSetupWidget(FibsemSystemSetupWidget.Ui_Form, QtWidgets.QWidget):
    set_stage_signal = pyqtSignal()
    connected_signal = pyqtSignal()
    disconnected_signal = pyqtSignal()
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
        config_path: str = None,
    ):
        super(FibsemSystemSetupWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.config_path = config_path

        self.setup_connections()

    def setup_connections(self):
        self.setStage_button.clicked.connect(self.set_stage_parameters)
        self.microscope_button.clicked.connect(self.connect)

    def set_stage_parameters(self):
        self.settings.system.stage.needle_stage_height_limit = self.needleStageHeightLimitnMmDoubleSpinBox.value()*constants.MILLIMETRE_TO_METRE
        self.settings.system.stage.tilt_flat_to_electron = self.tiltFlatToElectronSpinBox.value()
        self.settings.system.stage.tilt_flat_to_ion = self.tiltFlatToIonSpinBox.value()
        self.settings.system.stage.rotation_flat_to_electron = self.rotationFlatToElectronSpinBox.value()
        self.settings.system.stage.rotation_flat_to_ion = self.rotationFlatToIonSpinBox.value()
        self.set_stage_signal.emit()

    def connect(self):
        if self.lineEdit_ipadress.text() == "":
            _ = message_box_ui(
                title="IP adress not set.",
                text="Please enter an IP adress before connecting to microscope.",
                buttons=QMessageBox.Ok,
            )
            return
        try:
            ip_address = self.lineEdit_ipadress.text()
            manufacturer = self.comboBox_manufacturer.currentText()
            self.microscope, self.settings = utils.setup_session(ip_address=ip_address, manufacturer=manufacturer, config_path=self.config_path)
            logging.info("Microscope Connected")
            self.microscope_status.setText("Microscope Connected")
            self.microscope_status.setStyleSheet("background-color: green")
            self.microscope_button.clicked.disconnect()
            self.microscope_button.clicked.connect(self.disconnect_from_microscope)
            self.microscope_button.setText("Disconnect")
            self.needleStageHeightLimitnMmDoubleSpinBox.setValue(self.settings.system.stage.needle_stage_height_limit*constants.METRE_TO_MILLIMETRE)
            self.tiltFlatToElectronSpinBox.setValue(self.settings.system.stage.tilt_flat_to_electron)
            self.tiltFlatToIonSpinBox.setValue(self.settings.system.stage.tilt_flat_to_ion)
            self.rotationFlatToElectronSpinBox.setValue(self.settings.system.stage.rotation_flat_to_electron)
            self.rotationFlatToIonSpinBox.setValue(self.settings.system.stage.rotation_flat_to_ion)
            self.connected_signal.emit()
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {traceback.format_exc()}")
            self.microscope_status.setText("Microscope Disconnected")
            self.microscope_status.setStyleSheet("background-color: red")


    def disconnect_from_microscope(self):
        self.microscope.disconnect()
        self.microscope = None
        self.microscope_settings = None
        self.microscope_status.setText("Microscope Disconnected")
        self.microscope_status.setStyleSheet("background-color: red")
        self.microscope_button.clicked.disconnect()
        self.microscope_button.clicked.connect(self.connect)
        self.microscope_button.setText("Connect")
        self.disconnected_signal.emit()



def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemSystemSetupWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
