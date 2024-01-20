import argparse
import os
import logging
import napari
import napari.utils.notifications
import fibsem
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.ui.qtdesigner_files import FibsemMicroscopeConfigurationWidget
from fibsem.ui import _stylesheets
from fibsem.structures import BeamType
from fibsem.ui.utils import _get_file_ui, _get_save_file_ui, message_box_ui
from pprint import pprint


CONFIGURATION = {
    "tooltips": {"electron-column-tilt": "The tilt of the electron column. Typically 0 degrees.",
                 "ion-column-tilt": "The tilt of the ion column (Thermo = 52 deg, Tescan = 55 deg).", 
                 "rotation-reference": "The rotation of the stage when loading the sample (under the SEM)",
                 "rotation-180": "The rotation of the stage under the FIB (180 degrees from rotation-reference)",},
}


class FibsemMicroscopeConfigurationWidget(FibsemMicroscopeConfigurationWidget.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self,
        path: str = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemMicroscopeConfigurationWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle("Microscope Configuration")

        self.viewer = viewer

        self.setup_connections()

        if path is not None:
            self.path = path
            self.configuration = self.load_configuration(path)
            self.load_configuration_to_ui()

    def setup_connections(self):

        # combo boxes
        self.comboBox_configuration_manufacturer.addItems(cfg.__SUPPORTED_MANUFACTURERS__)
        self.comboBox_imaging_beam_type.addItems([b.name for b in BeamType])
        self.comboBox_configuration_manufacturer.currentTextChanged.connect(self.update_configuration_ui)

        # actions
        self.actionSave_Configuration.triggered.connect(self.save_configuration)
        self.actionLoad_Configuration.triggered.connect(self.load_configuration_from_file)
        
        # manufactuer specific widgets
        self.tescan_widgets = [
            self.label_milling_dwell_time,  self.doubleSpinBox_milling_dwell_time,
            self.label_milling_rate,        self.doubleSpinBox_milling_rate,
            self.label_milling_spot_size,   self.doubleSpinBox_milling_spot_size,
            self.label_milling_preset,      self.lineEdit_milling_preset,
        ]

        self.thermo_widgets = [
                self.label_milling_voltage, self.doubleSpinBox_milling_voltage,
                self.label_milling_current, self.doubleSpinBox_milling_current,
        ]

        # tooltips
        self.spinBox_electron_column_tilt.setToolTip(CONFIGURATION["tooltips"]["electron-column-tilt"])
        self.spinBox_ion_column_tilt.setToolTip(CONFIGURATION["tooltips"]["ion-column-tilt"])
        self.spinBox_stage_rotation_reference.setToolTip(CONFIGURATION["tooltips"]["rotation-reference"])
        self.spinBox_stage_rotation_180.setToolTip(CONFIGURATION["tooltips"]["rotation-180"])

        # dynamic changing configuration
        # change configuration values based on derived values, e.g. manufactuer -> column-tilt

    def update_configuration_ui(self):

        manufactuer = self.comboBox_configuration_manufacturer.currentText()

        TESCAN_MANUFACTUER = manufactuer == "Tescan"
        THERMO_MANUFACTUER = manufactuer == "Thermo"
        DEMO_MANUFACTURER = manufactuer == "Demo"

        if DEMO_MANUFACTURER:
            TESCAN_MANUFACTUER, THERMO_MANUFACTUER = True, True
        
        # toggle widgets based on manufactuer
        for widget in self.tescan_widgets:
            widget.setVisible(TESCAN_MANUFACTUER)
        for widget in self.thermo_widgets:
            widget.setVisible(THERMO_MANUFACTUER)


        # set ion column tilts
        if TESCAN_MANUFACTUER:
            self.spinBox_ion_column_tilt.setValue(cfg.DEFAULT_CONFIGURATION_VALUES["Tescan"]["ion-column-tilt"])
        elif THERMO_MANUFACTUER:
            self.spinBox_ion_column_tilt.setValue(cfg.DEFAULT_CONFIGURATION_VALUES["Thermo"]["ion-column-tilt"])

    def save_configuration(self):

        configuration = self.read_configuration_from_ui()
        configuration_name = configuration["info"]["name"]
        path = os.path.join(os.path.dirname(self.path), f"{configuration_name}.yaml")

        # select save file 
        path = _get_save_file_ui(msg="Select microscope configuration file", 
                                path=path, 
                                _filter="YAML (*.yaml)",
                                parent=self)
        if path == "":
            napari.utils.notifications.show_error(f"Please select a file to save the configuration to. Configuration not saved.")
            return 
        
        # save configuration
        utils.save_yaml(path, configuration)
        napari.utils.notifications.show_info(f"Configuration {configuration_name} saved to {path}")

        # add to user configurations
        msg = f"Would you like to add this configuration ({configuration_name}) to the user configurations?"
        ret = message_box_ui(text=msg, title="Add to user configurations?")

        # add to user configurations
        if ret:
            cfg.add_configuration(configuration_name=configuration_name, path=path)
            
            # set default configuration
            msg = f"Would you like to make this the default configuration?"
            ret = message_box_ui(text=msg, title="Set default configuration?")
            
            if ret:
                cfg.set_default_configuration(configuration_name=configuration_name)

    def load_configuration_from_file(self):

        path = _get_file_ui(msg="Select microscope configuration file", 
                            path=cfg.DEFAULT_CONFIGURATION_PATH, 
                            _filter="YAML (*.yaml)", parent=self)

        if path == "":
            napari.utils.notifications.show_error(f"Please select a file to load the configuration from. Configuration not loaded.")
            return
        
        self.path = path
        self.configuration = self.load_configuration(path)
        self.load_configuration_to_ui()

    def load_configuration(self, path: str):
        
        self.configuration = utils.load_yaml(path)

        return self.configuration
    
    def load_configuration_to_ui(self):
        
        # core
        info_configuration = self.configuration["info"]
        self.lineEdit_configuration_name.setText(info_configuration["name"])
        self.lineEdit_configuration_path.setText(self.path)
        self.lineEdit_configuration_ip_address.setText(info_configuration["ip_address"])
        self.comboBox_configuration_manufacturer.setCurrentText(info_configuration["manufacturer"])

        # stage
        stage_configuration = self.configuration["stage"]
        self.spinBox_stage_rotation_reference.setValue(stage_configuration["rotation_reference"])
        self.spinBox_stage_rotation_180.setValue(stage_configuration["rotation_180"])
        self.spinBox_stage_shuttle_pre_tilt.setValue(int(stage_configuration["shuttle_pre_tilt"]))
        self.doubleSpinBox_stage_manipulator_height_limit.setValue(stage_configuration["manipulator_height_limit"] * constants.SI_TO_MILLI) # mm

        # electron
        electron_configuration = self.configuration["electron"]
        self.spinBox_electron_column_tilt.setValue(electron_configuration["column_tilt"])                               # deg
        self.doubleSpinBox_electron_eucentric_height.setValue(electron_configuration["eucentric_height"] * constants.SI_TO_MILLI) # mm
        self.doubleSpinBox_electron_voltage.setValue(electron_configuration["voltage"])                                 # V
        self.doubleSpinBox_electron_current.setValue(electron_configuration["current"] * constants.SI_TO_NANO)         # nA
        self.lineEdit_electron_detector_mode.setText(electron_configuration["detector_mode"])
        self.lineEdit_electron_detector_type.setText(electron_configuration["detector_type"])

        # ion
        ion_configuration = self.configuration["ion"]
        self.spinBox_ion_column_tilt.setValue(ion_configuration["column_tilt"])                                         # deg
        self.doubleSpinBox_ion_eucentric_height.setValue(ion_configuration["eucentric_height"] * constants.SI_TO_MILLI) # mm
        self.doubleSpinBox_ion_voltage.setValue(ion_configuration["voltage"])                                           # V
        self.doubleSpinBox_ion_current.setValue(ion_configuration["current"] * constants.SI_TO_NANO)               # nA
        self.lineEdit_ion_detector_mode.setText(ion_configuration["detector_mode"])
        self.lineEdit_ion_detector_type.setText(ion_configuration["detector_type"])
        
        # imaging
        imaging_configuration =  self.configuration["imaging"]
        self.comboBox_imaging_beam_type.setCurrentText(imaging_configuration["beam_type"])
        self.spinBox_imaging_resolution_x.setValue(imaging_configuration["resolution"][0])
        self.spinBox_imaging_resolution_y.setValue(imaging_configuration["resolution"][1])
        self.doubleSpinBox_imaging_hfw.setValue(imaging_configuration["hfw"] * constants.SI_TO_MICRO)                   # um
        self.doubleSpinBox_imaging_dwell_time.setValue(imaging_configuration["dwell_time"] * constants.SI_TO_MICRO)     # us
        self.doubleSpinBox_imaging_current.setValue(imaging_configuration["imaging_current"] * constants.SI_TO_NANO)    # nA
        self.checkBox_imaging_autocontrast.setChecked(imaging_configuration["autocontrast"])
        self.checkBox_imaging_autogamma.setChecked(imaging_configuration["autogamma"])
        self.checkBox_imaging_autosave.setChecked(imaging_configuration["save"])

        # milling
        milling_configuration = self.configuration["milling"]
        self.doubleSpinBox_milling_voltage.setValue(milling_configuration["milling_voltage"])                                   # V
        self.doubleSpinBox_milling_current.setValue(milling_configuration["milling_current"] * constants.SI_TO_NANO)            # nA
        self.doubleSpinBox_milling_dwell_time.setValue(milling_configuration["dwell_time"] * constants.SI_TO_MICRO)     # us
        self.doubleSpinBox_milling_rate.setValue(milling_configuration["rate"])                                         # um3/s
        self.doubleSpinBox_milling_spot_size.setValue(milling_configuration["spot_size"] * constants.SI_TO_NANO)        # nm
        self.lineEdit_milling_preset.setText(milling_configuration["preset"])

        # subsystems
        subsystem_configuration = self.configuration
        # electron
        self.checkBox_subsystems_electron_enabled.setChecked(subsystem_configuration["electron"]["enabled"])
        # ion
        self.checkBox_subsystems_ion_enabled.setChecked(subsystem_configuration["ion"]["enabled"])
        self.checkBox_subsystems_ion_plasma.setChecked(subsystem_configuration["ion"]["plasma"])
        # stage
        self.checkBox_subsystems_stage_enabled.setChecked(subsystem_configuration["stage"]["enabled"])
        self.checkBox_subsystems_stage_rotation.setChecked(subsystem_configuration["stage"]["rotation"])
        self.checkBox_subsystems_stage_tilt.setChecked(subsystem_configuration["stage"]["tilt"])
        # manipulator
        self.checkBox_subsystems_manipulator_enabled.setChecked(subsystem_configuration["manipulator"]["enabled"])
        self.checkBox_subsystems_manipulator_rotation.setChecked(subsystem_configuration["manipulator"]["rotation"])
        self.checkBox_subsystems_manipulator_tilt.setChecked(subsystem_configuration["manipulator"]["tilt"])
        # gis
        self.checkBox_subsystems_gis_enabled.setChecked(subsystem_configuration["gis"]["enabled"])
        self.checkBox_subsystems_gis_multichem_enabled.setChecked(subsystem_configuration["gis"]["multichem"])
        self.checkBox_subsystems_gis_sputter_coater_enabled.setChecked(subsystem_configuration["gis"]["sputter_coater"])


    def read_configuration_from_ui(self) -> dict:

        # core
        info_configuration = self.configuration["info"]
        info_configuration["name"] = self.lineEdit_configuration_name.text()
        info_configuration["ip_address"] = self.lineEdit_configuration_ip_address.text()
        info_configuration["manufacturer"] = self.comboBox_configuration_manufacturer.currentText()

        # stage
        stage_configuration = self.configuration["stage"]
        stage_configuration["enabled"] = self.checkBox_subsystems_stage_enabled.isChecked()
        stage_configuration["rotation"] = self.checkBox_subsystems_stage_rotation.isChecked()
        stage_configuration["tilt"] = self.checkBox_subsystems_stage_tilt.isChecked()
        stage_configuration["rotation_reference"] = self.spinBox_stage_rotation_reference.value()
        stage_configuration["rotation_180"] = self.spinBox_stage_rotation_180.value()
        stage_configuration["shuttle_pre_tilt"] = self.spinBox_stage_shuttle_pre_tilt.value()
        stage_configuration["manipulator_height_limit"] = self.doubleSpinBox_stage_manipulator_height_limit.value() * constants.MILLI_TO_SI

        # electron
        electron_configuration = self.configuration["electron"]
        electron_configuration["enabled"] = self.checkBox_subsystems_electron_enabled.isChecked()
        electron_configuration["column_tilt"] = self.spinBox_electron_column_tilt.value()
        electron_configuration["eucentric_height"] = self.doubleSpinBox_electron_eucentric_height.value() * constants.MILLI_TO_SI
        electron_configuration["voltage"] = self.doubleSpinBox_electron_voltage.value()
        electron_configuration["current"] = self.doubleSpinBox_electron_current.value() * constants.NANO_TO_SI
        electron_configuration["detector_mode"] = self.lineEdit_electron_detector_mode.text()
        electron_configuration["detector_type"] = self.lineEdit_electron_detector_type.text()

        # ion
        ion_configuration = self.configuration["ion"]
        ion_configuration["enabled"] = self.checkBox_subsystems_ion_enabled.isChecked()
        ion_configuration["plasma"] = self.checkBox_subsystems_ion_plasma.isChecked()
        ion_configuration["column_tilt"] = self.spinBox_ion_column_tilt.value()
        ion_configuration["eucentric_height"] = self.doubleSpinBox_ion_eucentric_height.value() * constants.MILLI_TO_SI
        ion_configuration["voltage"] = self.doubleSpinBox_ion_voltage.value()
        ion_configuration["current"] = self.doubleSpinBox_ion_current.value() * constants.NANO_TO_SI
        ion_configuration["detector_mode"] = self.lineEdit_ion_detector_mode.text()
        ion_configuration["detector_type"] = self.lineEdit_ion_detector_type.text()

        # manipulator
        manipulator_configuration = self.configuration["manipulator"]
        manipulator_configuration["enabled"] = self.checkBox_subsystems_manipulator_enabled.isChecked()
        manipulator_configuration["rotation"] = self.checkBox_subsystems_manipulator_rotation.isChecked()
        manipulator_configuration["tilt"] = self.checkBox_subsystems_manipulator_tilt.isChecked()
        
        # gis
        gis_configuration = self.configuration["gis"]
        gis_configuration["enabled"] = self.checkBox_subsystems_gis_enabled.isChecked()
        gis_configuration["multichem"] = self.checkBox_subsystems_gis_multichem_enabled.isChecked()
        gis_configuration["sputter_coater"] = self.checkBox_subsystems_gis_sputter_coater_enabled.isChecked()

        # imaging
        imaging_configuration =  self.configuration["imaging"]
        imaging_configuration["beam_type"] = self.comboBox_imaging_beam_type.currentText()
        imaging_configuration["resolution"] = [self.spinBox_imaging_resolution_x.value(), self.spinBox_imaging_resolution_y.value()]
        imaging_configuration["hfw"] = self.doubleSpinBox_imaging_hfw.value() * constants.MICRO_TO_SI
        imaging_configuration["dwell_time"] = self.doubleSpinBox_imaging_dwell_time.value() * constants.MICRO_TO_SI
        imaging_configuration["imaging_current"] = self.doubleSpinBox_imaging_current.value() * constants.NANO_TO_SI
        imaging_configuration["autocontrast"] = self.checkBox_imaging_autocontrast.isChecked()
        imaging_configuration["autogamma"] = self.checkBox_imaging_autogamma.isChecked()
        imaging_configuration["save"] = self.checkBox_imaging_autosave.isChecked()

        # milling
        milling_configuration = self.configuration["milling"]
        milling_configuration["milling_voltage"] = self.doubleSpinBox_milling_voltage.value()
        milling_configuration["milling_current"] = self.doubleSpinBox_milling_current.value() * constants.NANO_TO_SI
        milling_configuration["dwell_time"] = self.doubleSpinBox_milling_dwell_time.value() * constants.MICRO_TO_SI
        milling_configuration["rate"] = self.doubleSpinBox_milling_rate.value()
        milling_configuration["spot_size"] = self.doubleSpinBox_milling_spot_size.value() * constants.NANO_TO_SI
        milling_configuration["preset"] = self.lineEdit_milling_preset.text()

        # configuration
        configuration = {
            "info": info_configuration,
            "stage": stage_configuration,
            "electron": electron_configuration,
            "ion": ion_configuration,
            "manipulator": manipulator_configuration,
            "gis": gis_configuration,         
            "imaging": imaging_configuration,
            "milling": milling_configuration,
        }

        return configuration
    



def main():

    # parse arguments
    parser = argparse.ArgumentParser(f"Microscope Configuration UI")
    parser.add_argument("--config", type=str, default=cfg.DEFAULT_CONFIGURATION_PATH, help="Path to microscope configuration file")
    args = parser.parse_args()

    # widget viewer
    viewer = napari.Viewer(ndisplay=2)
    microscope_configuration = FibsemMicroscopeConfigurationWidget(path=args.config, viewer=viewer)
    viewer.window.add_dock_widget(
        microscope_configuration, 
        area="right", 
        add_vertical_stretch=False,
        name=f"OpenFIBSEM v{fibsem.__version__} Microscope Configuration",
    )
    napari.run()


if __name__ == "__main__":
    main()