import logging
import traceback

import napari
import napari.utils.notifications
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from fibsem import config as cfg
from fibsem import constants, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, StageSettings
from fibsem.ui.qtdesigner_files import FibsemSystemSetupWidget

def log_status_message(step: str):
    logging.debug(
        f"STATUS | sYSTEM Widget | {step}"
    )


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
        self.config_path = config_path  # TODO: allow user to set this

        self.setup_connections()
        self.update_ui()

    def setup_connections(self):
        #
        self.lineEdit_ipadress.setText("localhost")
        self.comboBox_manufacturer.addItems(cfg.__SUPPORTED_MANUFACTURERS__)

        # buttons
        self.microscope_button.clicked.connect(self.connect_to_microscope)
        self.setStage_button.clicked.connect(self.get_stage_settings_from_ui)

    def get_stage_settings_from_ui(self):
        if self.microscope is None:
            return
        self.settings.system.stage.needle_stage_height_limit = (
            self.needleStageHeightLimitnMmDoubleSpinBox.value()
            * constants.MILLIMETRE_TO_METRE
        )
        self.settings.system.stage.tilt_flat_to_electron = (
            self.tiltFlatToElectronSpinBox.value()
        )
        self.settings.system.stage.tilt_flat_to_ion = self.tiltFlatToIonSpinBox.value()
        self.settings.system.stage.rotation_flat_to_electron = (
            self.rotationFlatToElectronSpinBox.value()
        )
        self.settings.system.stage.rotation_flat_to_ion = (
            self.rotationFlatToIonSpinBox.value()
        )
        self.set_stage_signal.emit()

    def set_stage_settings_to_ui(self, stage_settings: StageSettings) -> None:
        self.needleStageHeightLimitnMmDoubleSpinBox.setValue(
            stage_settings.needle_stage_height_limit * constants.METRE_TO_MILLIMETRE
        )
        self.tiltFlatToElectronSpinBox.setValue(stage_settings.tilt_flat_to_electron)
        self.tiltFlatToIonSpinBox.setValue(stage_settings.tilt_flat_to_ion)
        self.rotationFlatToElectronSpinBox.setValue(
            stage_settings.rotation_flat_to_electron
        )
        self.rotationFlatToIonSpinBox.setValue(stage_settings.rotation_flat_to_ion)

    def connect(self):
        if self.lineEdit_ipadress.text() == "":
            napari.utils.notifications.show_error(
                f"IP address not set. Please enter an IP address before connecting to microscope."
            )
            return

        try:
            log_status_message("CONNECTING")
            ip_address = self.lineEdit_ipadress.text()
            manufacturer = self.comboBox_manufacturer.currentText()
            # user notification
            msg = f"Connecting to microscope at {ip_address}"
            logging.info(msg)
            napari.utils.notifications.show_info(msg)

            # connect
            self.microscope, self.settings = utils.setup_session(
                ip_address=ip_address,
                manufacturer=manufacturer,
                config_path=self.config_path,
            )

            # user notification
            msg = f"Connected to microscope at {ip_address}"
            log_status_message("CONNECTED AT " + ip_address)
            logging.info(msg)
            napari.utils.notifications.show_info(msg)

        except Exception as e:
            msg = f"Unable to connect to the microscope: {traceback.format_exc()}"
            logging.error(msg)
            log_status_message(F"CONNECTION FAILED {traceback.format_exc()}")
            napari.utils.notifications.show_error(msg)

    def connect_to_microscope(self):

        _microscope_connected = bool(self.microscope)

        if _microscope_connected:
            self.microscope.disconnect()
            self.microscope, self.settings = None, None
        else:
            self.connect()

        self.update_ui()

    def update_ui(self):

        _microscope_connected = bool(self.microscope)

        self.setStage_button.setEnabled(_microscope_connected)

        if _microscope_connected:
            self.microscope_button.setText("Microscope Connected")
            self.microscope_button.setStyleSheet("background-color: green")
            self.set_stage_settings_to_ui(self.settings.system.stage)
            self.connected_signal.emit()

        else:
            self.microscope_button.setText("Connect To Microscope")
            self.microscope_button.setStyleSheet("background-color: gray")
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
