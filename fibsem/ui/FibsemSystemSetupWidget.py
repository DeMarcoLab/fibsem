import logging
import traceback
import yaml
import os 
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

import fibsem
from fibsem import config as cfg
from fibsem import constants, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, StageSettings, FibsemHardware, BeamSystemSettings, BeamType, ImageSettings, FibsemMillingSettings, SystemSettings
from fibsem.ui.qtdesigner_files import FibsemSystemSetupWidget
from fibsem.ui.utils import _get_file_ui, _get_save_file_ui

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
        image_widget: QtWidgets.QWidget = None,
        milling_widget: QtWidgets.QWidget = None,
        parent=None,
        config_path: str = None,
    ):
        super(FibsemSystemSetupWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.config_path = config_path  # TODO: allow user to set this

        settings_dict = utils.load_yaml(os.path.join(self.config_path))
        
        self.auto_connect = False
        self.apply_settings = False
        if bool(settings_dict["connect_to_microscope_on_startup"]):
            self.auto_connect = True
            self.connect_to_microscope(ip_address=settings_dict["system"]["ip_address"], manufacturer=settings_dict["system"]["manufacturer"])

        self.setup_connections(ip_address=settings_dict["system"]["ip_address"], manufacturer=settings_dict["system"]["manufacturer"])
        self.update_ui()
        

    def setup_connections(self, ip_address: str, manufacturer: str):
        #
        self.lineEdit_ipadress.setText(ip_address)
        self.comboBox_manufacturer.addItems(cfg.__SUPPORTED_MANUFACTURERS__)
        self.comboBox_manufacturer.setCurrentText(manufacturer)

        # buttons
        self.microscope_button.clicked.connect(self.connect_to_microscope)
        self.setStage_button.clicked.connect(self.get_stage_settings_from_ui)
        self.pushButton_save_yaml.clicked.connect(lambda: self.save_defaults(path=None))
        self.pushButton_apply_settings.clicked.connect(self.apply_defaults_settings)
        self.pushButton_import_yaml.clicked.connect(self.import_yaml)

        #checkboxes
        self.checkBox_eb.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_ib.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_select_plasma_gas.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_stage_enabled.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_stage_rotation.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_stage_tilt.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_needle_enabled.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_needle_rotation.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_needle_tilt.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_gis_enabled.stateChanged.connect(self.get_model_from_ui)
        self.checkBox_multichem.stateChanged.connect(self.get_model_from_ui)

    def import_yaml(self):
        path = _get_file_ui(msg="Select system file", path=cfg.CONFIG_PATH)
        if path == "":
            return
        self.settings = utils.load_settings_from_config(path)
        self.set_defaults_to_ui()
        self.set_stage_settings_to_ui(self.settings.system.stage)
        self.set_model_to_ui(self.settings.hardware)

    def get_default_settings_from_ui(self):
        microscope_settings = MicroscopeSettings(
            system=SystemSettings(
                ip_address=self.lineEdit_ipadress.text(),
                manufacturer=self.comboBox_manufacturer.currentText(),
                stage=StageSettings(
                    rotation_flat_to_electron=self.rotationFlatToElectronSpinBox.value(),
                    rotation_flat_to_ion=self.rotationFlatToIonSpinBox.value(),
                    tilt_flat_to_electron=self.tiltFlatToElectronSpinBox.value(),
                    tilt_flat_to_ion=self.tiltFlatToIonSpinBox.value(),
                    pre_tilt=self.preTiltSpinBox.value(),
                    needle_stage_height_limit=self.needleStageHeightLimitnMmDoubleSpinBox.value(),                    
                ),
                ion=BeamSystemSettings(
                    beam_type=BeamType.ION,
                    voltage=self.spinBox_ion_voltage.value(),
                    current=self.doubleSpinBox_ion_current.value() * constants.NANO_TO_SI,
                    plasma_gas=self.lineEdit_plasma_gas.text().capitalize(),
                    eucentric_height=self.doubleSpinBox_height_ion.value(),
                    detector_type=self.lineEdit_detector_type_ion.text(),
                    detector_mode=self.lineEdit_detector_mode_ion.text(),
                ),
                electron=BeamSystemSettings(
                    beam_type=BeamType.ELECTRON,
                    voltage=self.spinBox_voltage_eb.value(),
                    current=self.doubleSpinBox_current_eb.value() * constants.NANO_TO_SI,
                    eucentric_height=self.doubleSpinBox_height_eb.value(),
                    detector_type=self.lineEdit_detector_type_eb.text(),
                    detector_mode=self.lineEdit_detector_mode_eb.text(),
                ),
            ),
            image=ImageSettings(
                resolution=[self.spinBox_res_width.value(), self.spinBox_res_height.value()],
                dwell_time=self.doubleSpinBox_dwell_time_imaging.value()*constants.MICRO_TO_SI,
                hfw=self.spinBox_hfw.value()*constants.MICRO_TO_SI,
                beam_type = BeamType[self.lineEdit_beam_type.text().upper()],
                autocontrast=self.checkBox_autocontrast.isChecked(),
                save=self.checkBox_save.isChecked(),
                gamma_enabled=self.checkBox_gamma.isChecked(),
            ),
            milling=FibsemMillingSettings(
                dwell_time=self.doubleSpinBox_dwell_time_milling.value()*constants.MICRO_TO_SI,
                rate=self.doubleSpinBox_rate.value()*constants.NANO_TO_SI,
                spot_size=self.doubleSpinBox_spotsize.value()*constants.MICRO_TO_SI,
                milling_current=self.doubleSpinBox_milling_current.value()*constants.NANO_TO_SI,
            ),
            hardware=self.get_model_from_ui(),
        )

        return microscope_settings

    def apply_defaults_settings(self):
        microscope_settings = self.get_default_settings_from_ui()

        self.microscope.set_beam_settings(microscope_settings.system.ion)
        self.microscope.set_beam_settings(microscope_settings.system.electron)
        
        self.image_widget.set_ui_from_settings(microscope_settings.image, beam_type=microscope_settings.image.beam_type)

        self.get_stage_settings_from_ui()
        self.get_model_from_ui()
        self.milling_widget.set_milling_settings_ui(microscope_settings.milling)


    def save_defaults(self, path: str = None):
        system_dict = {}
        system_dict["system"] = {}
        system_dict["user"] = {}
        system_dict["system"]["ion"] = {}
        system_dict["system"]["electron"] = {}
        system_dict["system"]["stage"] = {}
        system_dict["user"]["milling"] = {}
        system_dict["user"]["imaging"] = {}

        system_dict["system"]["ip_address"] = self.lineEdit_ipadress.text()
        system_dict["system"]["manufacturer"] = self.comboBox_manufacturer.currentText()
        system_dict["system"]["ion"]["voltage"] = self.spinBox_ion_voltage.value()
        system_dict["system"]["ion"]["current"] = self.doubleSpinBox_ion_current.value() * constants.NANO_TO_SI
        system_dict["system"]["ion"]["plasma_gas"] = self.lineEdit_plasma_gas.text().capitalize()
        system_dict["system"]["ion"]["eucentric_height"] = self.doubleSpinBox_height_ion.value()
        system_dict["system"]["ion"]["detector_type"] = self.lineEdit_detector_type_ion.text()
        system_dict["system"]["ion"]["detector_mode"] = self.lineEdit_detector_mode_ion.text()
        system_dict["system"]["electron"]["voltage"] = self.spinBox_voltage_eb.value()
        system_dict["system"]["electron"]["current"] = self.doubleSpinBox_current_eb.value() * constants.NANO_TO_SI
        system_dict["system"]["electron"]["eucentric_height"] = self.doubleSpinBox_height_eb.value()
        system_dict["system"]["electron"]["detector_type"] = self.lineEdit_detector_type_eb.text()
        system_dict["system"]["electron"]["detector_mode"] = self.lineEdit_detector_mode_eb.text()

        system_dict["system"]["stage"]["rotation_flat_to_electron"] = self.rotationFlatToElectronSpinBox.value()
        system_dict["system"]["stage"]["rotation_flat_to_ion"] = self.rotationFlatToIonSpinBox.value()
        system_dict["system"]["stage"]["tilt_flat_to_electron"] = self.tiltFlatToElectronSpinBox.value()
        system_dict["system"]["stage"]["tilt_flat_to_ion"] = self.tiltFlatToIonSpinBox.value()
        system_dict["system"]["stage"]["pre_tilt"] = self.preTiltSpinBox.value()
        system_dict["system"]["stage"]["needle_stage_height_limit"] = self.needleStageHeightLimitnMmDoubleSpinBox.value()*constants.MILLIMETRE_TO_METRE

        system_dict["user"]["milling"]["milling_current"] = self.doubleSpinBox_milling_current.value()*constants.NANO_TO_SI
        system_dict["user"]["milling"]["spot_size"] = self.doubleSpinBox_spotsize.value()*constants.NANO_TO_SI
        system_dict["user"]["milling"]["rate"] = self.doubleSpinBox_rate.value()*constants.NANO_TO_SI
        system_dict["user"]["milling"]["dwell_time"] = self.doubleSpinBox_dwell_time_milling.value()*constants.MICRO_TO_SI
    
        system_dict["user"]["imaging"]["imaging_current"] = self.doubleSpinBox_imaging_current.value()*constants.NANO_TO_SI
        system_dict["user"]["imaging"]["resolution"] = [self.spinBox_res_width.value(), self.spinBox_res_height.value()]
        system_dict["user"]["imaging"]["hfw"] = self.spinBox_hfw.value()*constants.MICRO_TO_SI
        system_dict["user"]["imaging"]["beam_type"] = self.lineEdit_beam_type.text()
        system_dict["user"]["imaging"]["autocontrast"] = self.checkBox_autocontrast.isChecked()
        system_dict["user"]["imaging"]["dwell_time"] = self.doubleSpinBox_dwell_time_imaging.value()*constants.MICRO_TO_SI
        system_dict["user"]["imaging"]["save"] = self.checkBox_save.isChecked()
        system_dict["user"]["imaging"]["gamma"] = self.checkBox_gamma.isChecked()

        hardware_settings = self.get_model_from_ui()
        system_dict["model"] = hardware_settings.__to_dict__()
 
        system_dict["system"]["name"] = self.microscope.model
        system_dict["system"]["manufacturer"] = self.comboBox_manufacturer.currentText()
        system_dict["system"]["version"] = fibsem.__version__
        from PyQt5.QtWidgets import QInputDialog
        system_dict["system"]["id"] = QInputDialog.getText(self, "Microscope ID", "Please enter ID of microscope")[0]
        system_dict["system"]["description"] = QInputDialog.getText(self, "Description", "Please enter system description")[0]
    
        system_dict["connect_to_microscope_on_startup"] = bool(self.checkBox_connect_automatically.isChecked())
        system_dict["apply_settings_on_startup"] = bool(self.checkBox_apply_settings.isChecked())

        if path is None:
            path = _get_save_file_ui(msg="Save system file", path=cfg.CONFIG_PATH)
        
        if path == '':
            napari.utils.notifications.show_info(f"No file selected. System configuration not saved.")
            return
        
        utils.save_yaml(path=path, data=system_dict)
        
        napari.utils.notifications.show_info(f"System configuration saved to {os.path.basename(path)}")
        logging.info(f"System configuration saved to {os.path.basename(path)}")

    def set_defaults_to_ui(self):
        self.lineEdit_ipadress.setText(self.settings.system.ip_address)
        self.comboBox_manufacturer.setCurrentText( self.settings.system.manufacturer )
        self.spinBox_ion_voltage.setValue( self.settings.system.ion.voltage )
        self.doubleSpinBox_ion_current.setValue( self.settings.system.ion.current * constants.SI_TO_NANO )
        self.lineEdit_plasma_gas.setText( self.settings.system.ion.plasma_gas )
        self.doubleSpinBox_height_ion.setValue( self.settings.system.ion.eucentric_height )
        self.lineEdit_detector_type_ion.setText( self.settings.system.ion.detector_type )
        self.lineEdit_detector_mode_ion.setText( self.settings.system.ion.detector_mode )
        self.spinBox_voltage_eb.setValue( self.settings.system.electron.voltage )
        self.doubleSpinBox_current_eb.setValue( self.settings.system.electron.current * constants.SI_TO_NANO)
        self.doubleSpinBox_height_eb.setValue( self.settings.system.electron.eucentric_height )
        self.lineEdit_detector_type_eb.setText( self.settings.system.electron.detector_type )
        self.lineEdit_detector_mode_eb.setText( self.settings.system.electron.detector_mode )

        self.rotationFlatToElectronSpinBox.setValue( self.settings.system.stage.rotation_flat_to_electron )
        self.rotationFlatToIonSpinBox.setValue( self.settings.system.stage.rotation_flat_to_ion )
        self.tiltFlatToElectronSpinBox.setValue( self.settings.system.stage.tilt_flat_to_electron )
        self.tiltFlatToIonSpinBox.setValue( self.settings.system.stage.tilt_flat_to_ion )
        self.preTiltSpinBox.setValue( self.settings.system.stage.pre_tilt )
        self.needleStageHeightLimitnMmDoubleSpinBox.setValue( self.settings.system.stage.needle_stage_height_limit * constants.METRE_TO_MILLIMETRE )

        self.doubleSpinBox_milling_current.setValue( self.settings.milling.milling_current *constants.SI_TO_NANO )
        self.doubleSpinBox_spotsize.setValue( self.settings.milling.spot_size *constants.SI_TO_NANO )
        self.doubleSpinBox_rate.setValue( self.settings.milling.rate *constants.SI_TO_NANO )
        self.doubleSpinBox_dwell_time_milling.setValue( self.settings.milling.dwell_time *constants.SI_TO_MICRO )
    
        self.doubleSpinBox_imaging_current.setValue( 20.e-12 * constants.SI_TO_NANO )
        self.spinBox_res_width.setValue( self.settings.image.resolution[0] )
        self.spinBox_res_height.setValue( self.settings.image.resolution[1] )
        self.spinBox_hfw.setValue( int(self.settings.image.hfw *constants.SI_TO_MICRO) )
        self.lineEdit_beam_type.setText( self.settings.image.beam_type.name )
        self.checkBox_autocontrast.setCheckState( self.settings.image.autocontrast )
        self.doubleSpinBox_dwell_time_imaging.setValue( self.settings.image.dwell_time *constants.SI_TO_MICRO )
        self.checkBox_save.setCheckState( self.settings.image.save )
        self.checkBox_gamma.setCheckState( self.settings.image.gamma_enabled )

        self.checkBox_connect_automatically.setCheckState( self.auto_connect)
        self.checkBox_apply_settings.setCheckState( self.apply_settings)

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
        self.checkBox_select_plasma_gas.setChecked(hardware_settings.can_select_plasma_gas)
        self.checkBox_stage_enabled.setChecked(hardware_settings.stage_enabled)
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
        hardware_settings.can_select_plasma_gas = self.checkBox_select_plasma_gas.isChecked()
        hardware_settings.stage_rotation = self.checkBox_stage_rotation.isChecked()
        hardware_settings.stage_tilt = self.checkBox_stage_tilt.isChecked()
        hardware_settings.manipulator_enabled = self.checkBox_needle_enabled.isChecked()
        hardware_settings.manipulator_rotation = self.checkBox_needle_rotation.isChecked()
        hardware_settings.manipulator_tilt = self.checkBox_needle_tilt.isChecked()
        hardware_settings.gis_enabled = self.checkBox_gis_enabled.isChecked()
        hardware_settings.gis_multichem = self.checkBox_multichem.isChecked()
        hardware_settings.manipulator_positions = self.settings.hardware.manipulator_positions

        self.settings.hardware = hardware_settings
        self.microscope.hardware_settings = hardware_settings
        logging.debug(f"Updated hardware settings: {hardware_settings}")
        return hardware_settings


    def connect(self, ip_address: str = None, manufacturer: str = None) -> None:

        if not isinstance(ip_address, str):
            if self.lineEdit_ipadress.text() == "":
                napari.utils.notifications.show_error(
                    f"IP address not set. Please enter an IP address before connecting to microscope."
                )
                return
            else:
                ip_address = self.lineEdit_ipadress.text() 

        try:
            log_status_message("CONNECTING")
            
            manufacturer = self.comboBox_manufacturer.currentText() if manufacturer is None else manufacturer
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
            log_status_message(f"CONNECTED_AT_{ip_address}")
            logging.info(msg)
            napari.utils.notifications.show_info(msg)
            # self.connected_signal.emit()
            self.set_defaults_to_ui()

        except Exception as e:
            msg = f"Unable to connect to the microscope: {traceback.format_exc()}"
            logging.error(msg)
            log_status_message(F"CONNECTION_FAILED_{traceback.format_exc()}")
            napari.utils.notifications.show_error(msg)

    def connect_to_microscope(self, ip_address: str = None, manufacturer: str = None):

        _microscope_connected = bool(self.microscope)

        if _microscope_connected:
            self.microscope.disconnect()
            self.microscope, self.settings = None, None
        else:
            self.connect(ip_address, manufacturer )

        self.update_ui()

    def update_ui(self):

        _microscope_connected = bool(self.microscope)

        self.setStage_button.setEnabled(_microscope_connected)

        if _microscope_connected:
            self.microscope_button.setText("Microscope Connected")
            self.microscope_button.setStyleSheet("background-color: green")
            self.set_stage_settings_to_ui(self.microscope.stage_settings)
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
