import logging
import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.microscope import FibsemMicroscope, TescanMicroscope, ThermoMicroscope, DemoMicroscope
from fibsem.structures import (MicroscopeSettings)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemGISWidget
from fibsem.gis import deposit_platinum


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

        self.gas_protocol = {}

        self.gis_and_mc_setup()

        self.initial_gas_setup()

        self.setup_connections()

        self.update_ui()


    def gis_and_mc_setup(self):

        if isinstance(self.microscope, ThermoMicroscope):
            self.gis_lines = self.microscope.multichem_available_lines()
        else:
            self.gis_lines = self.microscope.GIS_available_lines()
       

        self.gas_combobox.addItems(self.gis_lines)
        self.gis_current_line = self.gis_lines[0]

        self.temp_label.setText(f"Temp: Need Warm Up")
        self.warm_button.setEnabled(True)

        self.gas_combobox.setCurrentText(self.gis_current_line)

        if isinstance(self.microscope, ThermoMicroscope):
            self.gis_available_positions = self.microscope.multichem_available_positions()
        else:
            self.gis_available_positions = self.microscope.GIS_available_positions()
        
        self.position_combobox.addItems(self.gis_available_positions)

        protocol = utils.load_yaml(cfg.PROTOCOL_PATH)
        self.protocol = protocol

        if isinstance(self.microscope, TescanMicroscope):
            self.tescan_setup()
        if isinstance(self.microscope, (ThermoMicroscope, DemoMicroscope)):
            self.thermo_setup()


    def initial_gas_setup(self):

        self.gas_protocol = {
            "application_file": self.protocol["milling"]["application_file"],
            "gas": self.gis_current_line,
            "position": "Retract",
            "hfw":self.image_widget.image_settings.hfw,
            "length": 7.0e-6,
            "spot_size": 5.0e-8,
            "beam_current": 5.0e-10,
            "dwell_time": 1.0e-6,
            "beam_type": "ELECTRON",
            "time": 1.5,
            "blank_beam": False,
        }

        self.timeDuration_spinbox.setValue(self.gas_protocol["time"])

    def tescan_setup(self):
        
        self.app_file_combobox.hide()
        self.app_file_label.hide()
        self.blankBeamcheckbox.hide()
        self.timeDuration_spinbox.setValue(1)
        self.update_ui()

    def thermo_setup(self):

        self.gas_combobox.clear()
        self.gas_combobox.addItems(self.gis_lines)
        self.gas_combobox.setCurrentText(self.gis_current_line)

        self.blankBeamcheckbox.setChecked(False)


        app_files =  self.microscope.get_available_values(key="application_file")

        self.app_file_combobox.addItems(app_files)

        if self.protocol['milling']['application_file'] in app_files:
            self.app_file_combobox.setCurrentText(self.protocol['milling']['application_file'])
        else:
            self.app_file_combobox.setCurrentText(app_files[0])
            self.gas_protocol['application_file'] = app_files[0]
            logging.info(f"Application file {self.protocol['milling']['application_file']} not found in microscope")
            logging.info(f'Application file set to {app_files[0]}')

        self.update_ui()


    def update_gas_protocol(self):

        self.gas_protocol["application_file"] = self.app_file_combobox.currentText()
        self.gas_protocol["beam_type"] = self.beamtype_combobox.currentText()
        self.gas_protocol["hfw"] = self.hfw_spinbox.value()*constants.MICRON_TO_METRE
        self.gas_protocol["time"] = self.timeDuration_spinbox.value()
        self.gas_protocol["blank_beam"] = self.blankBeamcheckbox.isChecked()
        self.gas_protocol["position"] = self.position_combobox.currentText()
        self.gas_protocol["gas"] = self.gas_combobox.currentText()

    def setup_connections(self):
        self.gas_combobox.currentIndexChanged.connect(self.update_ui)
        self.move_GIS_button.clicked.connect(self.move_gis)
        self.warm_button.clicked.connect(self.warm_up_gis)
        self.run_button.clicked.connect(self.run_gis)
        self.beamtype_combobox.setCurrentText("ION")
        self.pushButton_settings_update.clicked.connect(self.update_gas_protocol)

    

    def warm_up_gis(self):
        
        line_name = self.gis_current_line

        self.microscope.multichem_heat_up(line_name) if isinstance(self.microscope, ThermoMicroscope) else self.microscope.GIS_heat_up(line_name)
        self.temp_label.setText("Temp: Ready")


    def move_gis(self):

        position = self.position_combobox.currentText()
        if isinstance(self.microscope, ThermoMicroscope):
            self.microscope.multichem_move_to(position)
        else:
            self.microscope.GIS_move_to(self.gis_current_line,position)
        
        self.gas_protocol["position"] = position
        self.update_ui()

    def update_ui(self):

        line_name = self.gas_combobox.currentText()
        if line_name == "":
            return

        self.gis_current_line = line_name
       
            
        if isinstance(self.microscope, ThermoMicroscope):
            current_position =self.microscope.multichem_position()
            temp_ready = "Ready" if self.microscope.multichem_temp_ready(self.gis_current_line) else "Need Warm Up"
        else:
            temp_ready = "Ready" if self.microscope.GIS_temp_ready(self.gis_current_line) else "Need Warm Up"
            current_position = self.microscope.GIS_position(self.gis_current_line)

        self.temp_label.setText(f"Temp: {temp_ready}")

        self.position_combobox.setCurrentText(current_position)
        self.current_position_label.setText(f"Current Position: {current_position}")
        self.hfw_spinbox.setValue(self.image_widget.image_settings.hfw*constants.METRE_TO_MICRON)



    def run_gis(self):

        self.gas_protocol["gas"] = self.gis_current_line
        deposit_platinum(self.microscope, self.gas_protocol)
        
        


def main(): 

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemGISWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()