import logging
import os
import sys

import fibsem
import napari
import napari.utils.notifications
import yaml
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import calibration, constants, utils
from fibsem.structures import (
    BeamSystemSettings,
    BeamType,
    DefaultSettings,
    GammaSettings,
    ImageSettings,
    MicroscopeSettings,
    StageSettings,
    SystemSettings,
)
from fibsem.ui.qtdesigner_files import FibsemSettings
from PyQt5 import QtWidgets

BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")


class FibsemSettings(FibsemSettings.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        viewer: napari.Viewer,
        microscope: SdbMicroscopeClient = None,
        settings: MicroscopeSettings = None,
        parent=None,
    ):
        super(FibsemSettings, self).__init__(parent=parent)
        self.setupUi(self)
        self.viewer = viewer

        # connect to microscope, if required
        if microscope is None:
            self.microscope, self.settings = utils.setup_session()
        else:
            self.microscope = microscope
            self.settings = settings

        self.setup_connections()
        self.update_ui_from_settings(self.settings)

    def setup_connections(self):

        # TODO: set limits based on microscope

        # buttons
        self.pushButton_save_settings.clicked.connect(self.save_settings)
        self.pushButton_load_settings.clicked.connect(self.load_settings)
        self.pushButton_get_electron_settings.clicked.connect(self.get_beam_settings)
        self.pushButton_get_ion_settings.clicked.connect(self.get_beam_settings)

        # system
        app_files = sorted(self.microscope.patterning.list_all_application_files())
        self.comboBox_system_application_file.addItems(app_files)

        # stage

        # electron
        self.microscope.imaging.set_active_view(BeamType.ELECTRON.value)
        self.microscope.imaging.set_active_device(BeamType.ELECTRON.value)
        eb_detector_modes = self.microscope.detector.mode.available_values
        eb_detector_types = self.microscope.detector.type.available_values

        self.comboBox_electron_detector_mode.addItems(eb_detector_modes)
        self.comboBox_electron_detector_type.addItems(eb_detector_types)

        # ion
        self.microscope.imaging.set_active_view(BeamType.ION.value)
        self.microscope.imaging.set_active_device(BeamType.ION.value)
        ib_detector_modes = self.microscope.detector.mode.available_values
        ib_detector_types = self.microscope.detector.type.available_values
        ib_plasma_gases = (
            self.microscope.beams.ion_beam.source.plasma_gas.available_values
        )
        self.comboBox_ion_detector_mode.addItems(ib_detector_modes)
        self.comboBox_ion_detector_type.addItems(ib_detector_types)
        self.comboBox_ion_plasma_gas.addItems(ib_plasma_gases)

        # imaging
        self.comboBox_imaging_beam_type.addItems([beam.name for beam in BeamType])
        self.comboBox_imaging_resolution.addItems([res for res in self.microscope.beams.electron_beam.scanning.resolution.available_values])


    def update_ui_from_settings(self, settings: MicroscopeSettings):

        # system
        self.lineEdit_system_ip_address.setText(settings.system.ip_address)
        self.comboBox_system_application_file.setCurrentText(settings.system.application_file)

        # stage
        self.spinBox_stage_rotation_flat_to_electron.setValue(
            settings.system.stage.rotation_flat_to_electron
        )
        self.spinBox_stage_rotation_flat_to_ion.setValue(
            settings.system.stage.rotation_flat_to_ion
        )
        self.spinBox_stage_pre_tilt.setValue(0)  # TODO: fix when pre-titl implemented
        self.spinBox_stage_tilt_flat_to_electron.setValue(
            settings.system.stage.tilt_flat_to_electron
        )
        self.spinBox_stage_tilt_flat_to_ion.setValue(settings.system.stage.tilt_flat_to_ion)
        self.doubleSpinBox_stage_needle_height_limit.setValue(
            settings.system.stage.needle_stage_height_limit * constants.METRE_TO_MILLIMETRE
        )

        # electron
        self.doubleSpinBox_electron_voltage.setValue(
            settings.system.electron.voltage * constants.SI_TO_KILO
        )
        self.doubleSpinBox_electron_current.setValue(
            settings.system.electron.current * constants.SI_TO_PICO
        )
        self.comboBox_electron_detector_mode.setCurrentText(
            settings.system.electron.detector_mode
        )
        self.comboBox_electron_detector_type.setCurrentText(
            settings.system.electron.detector_type
        )
        self.doubleSpinBox_electron_eucentric_height.setValue(
            settings.system.electron.eucentric_height * constants.METRE_TO_MILLIMETRE
        )

        # ion
        self.doubleSpinBox_ion_voltage.setValue(
            settings.system.ion.voltage * constants.SI_TO_KILO
        )
        self.doubleSpinBox_ion_current.setValue(
            settings.system.ion.current * constants.SI_TO_PICO
        )
        self.comboBox_ion_detector_mode.setCurrentText(settings.system.ion.detector_mode)
        self.comboBox_ion_detector_type.setCurrentText(settings.system.ion.detector_type)
        self.doubleSpinBox_ion_eucentric_height.setValue(
            settings.system.ion.eucentric_height * constants.METRE_TO_MILLIMETRE
        )
        self.comboBox_ion_plasma_gas.setCurrentText(settings.system.ion.plasma_gas)

        # imaging
        self.comboBox_imaging_resolution.setCurrentText(settings.image.resolution)
        self.doubleSpinBox_imaging_dwell_time.setValue(settings.image.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_imaging_hfw.setValue(settings.image.hfw * constants.SI_TO_MICRO)
        self.checkBox_imaging_use_autocontrast.setChecked(settings.image.autocontrast)
        self.checkBox_imaging_use_autogamma.setChecked(settings.image.gamma.enabled)
        self.checkBox_imaging_save_image.setChecked(settings.image.save)


    def get_beam_settings(self):

        if self.sender() == self.pushButton_get_electron_settings:
            beam_type = BeamType.ELECTRON

        if self.sender() == self.pushButton_get_ion_settings:
            beam_type = BeamType.ION

        beam_settings = calibration.get_current_beam_system_state(
            self.microscope, beam_type
        )

        # electron
        if beam_type is BeamType.ELECTRON:
            self.doubleSpinBox_electron_voltage.setValue(
                beam_settings.voltage * constants.SI_TO_KILO
            )
            self.doubleSpinBox_electron_current.setValue(
                beam_settings.current * constants.SI_TO_PICO
            )
            self.comboBox_electron_detector_mode.setCurrentText(
                beam_settings.detector_mode
            )
            self.comboBox_electron_detector_type.setCurrentText(
                beam_settings.detector_type
            )
            self.doubleSpinBox_electron_eucentric_height.setValue(
                beam_settings.eucentric_height * constants.METRE_TO_MILLIMETRE
            )

        # ion
        if beam_type is BeamType.ION:
            self.doubleSpinBox_ion_voltage.setValue(
                beam_settings.voltage * constants.SI_TO_KILO
            )
            self.doubleSpinBox_ion_current.setValue(
                beam_settings.current * constants.SI_TO_PICO
            )
            self.comboBox_ion_detector_mode.setCurrentText(beam_settings.detector_mode)
            self.comboBox_ion_detector_type.setCurrentText(beam_settings.detector_type)
            self.doubleSpinBox_ion_eucentric_height.setValue(
                beam_settings.eucentric_height * constants.METRE_TO_MILLIMETRE
            )
            self.comboBox_ion_plasma_gas.setCurrentText(beam_settings.plasma_gas)

    def get_settings_from_ui(self) -> dict:

        settings = MicroscopeSettings( 
            system = SystemSettings(
                ip_address=self.lineEdit_system_ip_address.text(),
                application_file=self.comboBox_system_application_file.currentText(),
                stage=StageSettings(
                    rotation_flat_to_electron=int(
                        self.spinBox_stage_rotation_flat_to_electron.value()
                    ),
                    rotation_flat_to_ion=int(
                        self.spinBox_stage_rotation_flat_to_ion.value()
                    ),
                    tilt_flat_to_electron=int(
                        self.spinBox_stage_tilt_flat_to_electron.value()
                    ),
                    tilt_flat_to_ion=int(self.spinBox_stage_tilt_flat_to_ion.value()),
                    needle_stage_height_limit=float(
                        self.doubleSpinBox_stage_needle_height_limit.value()
                    ),
                ),
                electron=BeamSystemSettings(
                    beam_type=BeamType.ELECTRON,
                    voltage=float(
                        self.doubleSpinBox_electron_voltage.value() * constants.KILO_TO_SI
                    ),
                    current=float(
                        self.doubleSpinBox_electron_current.value() * constants.PICO_TO_SI
                    ),
                    detector_type=str(self.comboBox_electron_detector_type.currentText()),
                    detector_mode=str(self.comboBox_electron_detector_mode.currentText()),
                    eucentric_height=float(
                        self.doubleSpinBox_electron_eucentric_height.value()
                        * constants.MILLIMETRE_TO_METRE
                    ),
                    plasma_gas=None,
                ),
                ion=BeamSystemSettings(
                    beam_type=BeamType.ION,
                    voltage=float(
                        self.doubleSpinBox_ion_voltage.value() * constants.KILO_TO_SI
                    ),
                    current=float(
                        self.doubleSpinBox_ion_current.value() * constants.PICO_TO_SI
                    ),
                    detector_type=str(self.comboBox_ion_detector_type.currentText()),
                    detector_mode=str(self.comboBox_ion_detector_mode.currentText()),
                    eucentric_height=float(
                        self.doubleSpinBox_ion_eucentric_height.value()
                        * constants.MILLIMETRE_TO_METRE
                    ),
                    plasma_gas=str(self.comboBox_ion_plasma_gas.currentText()),
                ),
            ),
            image = ImageSettings(
                beam_type = BeamType.ELECTRON,
                resolution=self.comboBox_imaging_resolution.currentText(),
                dwell_time=self.doubleSpinBox_imaging_dwell_time.value() * constants.MICRO_TO_SI,
                hfw = self.doubleSpinBox_imaging_hfw.value() * constants.MICRO_TO_SI,
                autocontrast=self.checkBox_imaging_use_autocontrast.isChecked(),
                save=self.checkBox_imaging_save_image.isChecked(),
                gamma = GammaSettings(
                    self.checkBox_imaging_use_autogamma.isChecked()
                ),
                label = None
            ),
            default=DefaultSettings(
                imaging_current= None,
                milling_current=None

            )
        )
        return settings.__to_dict__()

    def save_settings(self):

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            caption="Save Settings File",
            directory=BASE_PATH,
            filter="Yaml Files (*.yaml)",
        )

        if filename == "":
            return

        # get and save settings
        settings_dict = self.get_settings_from_ui()

        with open(filename, "w") as f:
            yaml.dump(settings_dict, f, indent=4)
        logging.info(f"settings saved to: {filename}")

    def load_settings(self):

        logging.info(f"save settings")

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption="Load Settings File",
            directory=BASE_PATH,
            filter="Yaml Files (*.yaml)",
        )

        if filename == "":
            return

        # load settings
        with open(filename, "r") as f:
            system_settings = SystemSettings.__from_dict__(yaml.safe_load(f))

        self.settings.system = system_settings
        self.update_ui_from_settings(self.settings.system)
        logging.info(f"loaded settings from: {filename}")

    def cancel(self):

        self.close()
        self.viewer.close()

def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_settings = FibsemSettings(viewer=viewer)
    viewer.window.add_dock_widget(
        fibsem_settings, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
