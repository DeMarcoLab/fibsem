import logging
import traceback
import yaml
import os 
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

from fibsem import config as cfg
from fibsem import constants, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, StageSettings, FibsemHardware
from fibsem.ui.qtdesigner_files import FibsemSystemSetupWidget
from fibsem.ui.utils import _get_file_ui


def log_status_message(step: str):
    logging.debug(
        f"STATUS | System Widget | {step}"
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
        self.lineEdit_ipadress.setText(cfg.__DEFAULT_IP_ADDRESS__)
        self.comboBox_manufacturer.addItems(cfg.__SUPPORTED_MANUFACTURERS__)

        # buttons
        self.microscope_button.clicked.connect(self.connect_to_microscope)
        self.setStage_button.clicked.connect(self.get_stage_settings_from_ui)
        self.pushButton_save_defaults.connect(self.save_defaults)

        #checkboxes
        self.checkBox_eb.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_ib.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_stage_enabled.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_stage_rotation.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_stage_tilt.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_needle_enabled.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_needle_rotation.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_needle_tilt.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_gis_enabled.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_multichem.stateChanged.connect(self.get_model_from_ui)

    def save_defaults(self):
        system_dict = {}

        system_dict["system"]["ip_address"] = self.lineEdit_ipadress.text()
        system_dict["system"]["manufacturer"] = self.comboBox_manufacturer.currentText()
        system_dict["system"]["ion"]["voltage"] = self.spinBox_ion_voltage.value()
        system_dict["system"]["ion"]["current"] = self.spinBox_ion_current.value()
        system_dict["system"]["ion"]["plasma_gas"] = self.lineEdit_plasma_gas.text().capitalize()
        system_dict["system"]["ion"]["eucentric_height"] = self.doubleSpinBox_height_ion.value()
        system_dict["system"]["ion"]["detector_type"] = self.lineEdit_detector_type_ion.text()
        system_dict["system"]["ion"]["detector_mode"] = self.lineEdit_detector_mode_ion.text()
        system_dict["system"]["electron"]["voltage"] = self.spinBox_voltage_eb.value()
        system_dict["system"]["electron"]["current"] = self.spinBox_current_eb.value()
        system_dict["system"]["electron"]["eucentric_height"] = self.doubleSpinBox_height_eb.value()
        system_dict["system"]["electron"]["detector_type"] = self.lineEdit_detector_type_eb.text()
        system_dict["system"]["electron"]["detector_mode"] = self.lineEdit_detector_mode_eb.text()

        system_dict["stage"]["rotation_flat_to_electron"] = self.spinBox_rotation_eb.value()
        system_dict["stage"]["rotation_flat_to_ion"] = self.spinBox_rotation_ib.value()
        system_dict["stage"]["tilt_flat_to_electron"] = self.spinBox_tilt_eb.value()
        system_dict["stage"]["tilt_flat_to_ion"] = self.spinBox_tilt_ib.value()
        system_dict["stage"]["pre_tilt"] = self.spinBox_pretilt.value()
        system_dict["stage"]["needle_stage_height_limit"] = self.doubleSpinBox_needle_height.value()

        system_dict["user"]["milling"]["milling_current"] = self.doubleSpinBox_milling_current.value()*constants.NANO_TO_SI,
        system_dict["user"]["milling"]["spot_size"] = self.doubleSpinBox_spotsize.value()*constants.NANO_TO_SI,
        system_dict["user"]["milling"]["rate"] = self.doubleSpinBox_rate.value()*constants.NANO_TO_SI,
        system_dict["user"]["milling"]["dwell_time"] = self.doubleSpinBox_dwell_time_milling.value()*constants.MICRO_TO_SI,
    
        system_dict["user"]["imaging"]["imaging_current"] = self.doubleSpinBox_imaging_current.value()*constants.NANO_TO_SI,
        system_dict["user"]["imaging"]["resolution"] = [self.spinBox_res_width.value(), self.spinBox_res_height.value()],
        system_dict["user"]["imaging"]["hfw"] = self.spinBox_hfw.value()*constants.MICRO_TO_SI,
        system_dict["user"]["imaging"]["beam_type"] = self.lineEdit_beam_type.text(),
        system_dict["user"]["imaging"]["autocontrast"] = self.checkBox_autocontrast.isChecked(),
        system_dict["user"]["imaging"]["dwell_time"] = self.doubleSpinBox_dwell_time_imaging.value()*constants.MICRO_TO_SI,
        system_dict["user"]["imaging"]["save"] = self.checkBox_save.isChecked(),
        system_dict["user"]["imaging"]["gamma"] = self.checkBox_gamma.isChecked(),
    
        protocol_path = _get_file_ui(msg="Select protocol file")
        if protocol_path == '':
            return
        with open(os.path.join(protocol_path), "w") as f:
            yaml.safe_dump(system_dict, f, indent=4)

        logging.info("Protocol saved to file")

    def set_defaults_to_ui(self):
        pass

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
        self.settings.system.stage.pre_tilt = self.preTiltSpinBox.value()
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
        self.preTiltSpinBox.setValue(stage_settings.pre_tilt)

    def set_model_to_ui(self, hardware_settings: FibsemHardware) -> None:
        self.checkBox_eb.setChecked(hardware_settings.electron_beam)
        self.checkBox_ib.setChecked(hardware_settings.ion_beam)
        self.checkBox_stage_enabled.setChecked(hardware_settings.stage_rotation)
        self.checkBox_stage_rotation.setChecked(hardware_settings.stage_rotation)
        self.checkBox_stage_tilt.setChecked(hardware_settings.stage_tilt)
        self.checkBox_needle_enabled.setChecked(hardware_settings.manipulator_enabled)
        self.checkBox_needle_rotation.setChecked(hardware_settings.manipulator_rotation)
        self.checkBox_needle_tilt.setChecked(hardware_settings.manipulator_tilt)
        self.checkBox_gis_enabled.setChecked(hardware_settings.gis_enabled)
        self.checkBox_multichem.setChecked(hardware_settings.gis_multichem)

    def get_model_from_ui(self) -> FibsemHardware:
        hardware_settings = FibsemHardware()
        hardware_settings.electron_beam = self.checkBox_eb.isChecked()
        hardware_settings.ion_beam = self.checkBox_ib.isChecked()
        hardware_settings.stage_rotation = self.checkBox_stage_rotation.isChecked()
        hardware_settings.stage_tilt = self.checkBox_stage_tilt.isChecked()
        hardware_settings.manipulator_enabled = self.checkBox_needle_enabled.isChecked()
        hardware_settings.manipulator_rotation = self.checkBox_needle_rotation.isChecked()
        hardware_settings.manipulator_tilt = self.checkBox_needle_tilt.isChecked()
        hardware_settings.gis_enabled = self.checkBox_gis_enabled.isChecked()
        hardware_settings.gis_multichem = self.checkBox_multichem.isChecked()

        self.settings.hardware = hardware_settings
        self.microscope.hardware_settings = hardware_settings
        logging.info(f"Updated hardware settings: {hardware_settings}")


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
            log_status_message("CONNECTED_AT_" + ip_address)
            logging.info(msg)
            napari.utils.notifications.show_info(msg)
            # self.connected_signal.emit()

        except Exception as e:
            msg = f"Unable to connect to the microscope: {traceback.format_exc()}"
            logging.error(msg)
            log_status_message(F"CONNECTION_FAILED_{traceback.format_exc()}")
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
            self.set_model_to_ui(self.settings.hardware)
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
