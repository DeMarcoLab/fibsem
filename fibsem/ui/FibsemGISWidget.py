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
        self.GIS_inserted = False
        self.GIS_insert_status_label.setText(f"GIS Status: inserted" if self.GIS_inserted else "GIS Status: retracted")

        self.gis_lines = self.microscope.GIS_available_lines()
       

        self.gas_combobox.addItems(self.gis_lines)
        self.gis_current_line = self.gis_lines[0]

        self.temp_label.setText(f"Temp: Need Warm Up")
        self.warm_button.setEnabled(True)

        self.gas_combobox.setCurrentText(self.gis_current_line)

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
            "position": "cryo",
            "hfw":self.image_widget.image_settings.hfw,
            "length": 7.0e-6,
            "spot_size": 5.0e-8,
            "beam_current": 5.0e-10,
            "dwell_time": 1.0e-6,
            "beam_type": "ELECTRON",
            "sputter_time": 1.0e-6,
        }

        self.timeDuration_spinbox.setValue(self.gas_protocol["sputter_time"]*constants.SI_TO_MICRO)

    def tescan_setup(self):
        
        self.GIS_insert_status_label.hide()
        self.insertGIS_button.setEnabled(False)
        self.insertGIS_button.hide()
        self.GIS_radioButton.hide()
        self.multichem_radioButton.hide()
        self.app_file_combobox.hide()
        self.app_file_label.hide()
        self.blankBeamcheckbox.hide()
        self.timeDuration_spinbox.setValue(1)
        self.GIS = True
        

        self.update_ui()

    def thermo_setup(self):

        self.mc_lines = self.microscope.multichem_available_lines()
        self.mc_current_line = self.mc_lines[0]
        self.mc_available_positions = self.microscope.multichem_available_positions()
        

        self.GIS_inserted = True
        self.insert_retract_gis()
        self.GIS = True
        self.GIS_radioButton.setChecked(True)
        self.GIS = True
        self.position_combobox.hide()
        self.position_combobox.setEnabled(False)

        self.move_GIS_button.setEnabled(False)
        self.move_GIS_button.hide()
        
        self.gas_combobox.clear()
        self.gas_combobox.addItems(self.gis_lines)
        self.gas_combobox.setCurrentText(self.gis_current_line)

        self.label_position.hide()
        self.label_position.setEnabled(False)

        self.insertGIS_button.setEnabled(True)
        self.insertGIS_button.show()

        self.current_position_label.setEnabled(False)
        self.current_position_label.hide()

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

    def change_gis_multichem(self):
        
        checked_button = self.sender()

        if not checked_button.isChecked():
            return
    
        if checked_button == self.GIS_radioButton:
            self.GIS = True
            # self.position_combobox.setCurrentText(self.gis_current_line)
            self.position_combobox.hide()
            self.position_combobox.setEnabled(False)

            self.move_GIS_button.setEnabled(False)
            self.move_GIS_button.hide()
            
            self.gas_combobox.clear()
            self.gas_combobox.addItems(self.gis_lines)
            self.gas_combobox.setCurrentText(self.gis_current_line)

            self.label_position.hide()
            self.label_position.setEnabled(False)

            self.insertGIS_button.setEnabled(True)
            self.insertGIS_button.show()

            self.current_position_label.setEnabled(False)
            self.current_position_label.hide()

            self.GIS_insert_status_label.setEnabled(True)
            self.GIS_insert_status_label.show()
        
        if checked_button == self.multichem_radioButton:
            self.GIS = False
            self.position_combobox.clear()
            self.position_combobox.addItems(self.mc_available_positions)
            self.position_combobox.setEnabled(True)
            self.position_combobox.show()

            self.move_GIS_button.setEnabled(True)
            self.move_GIS_button.show()
            
            self.insertGIS_button.setEnabled(False)
            self.insertGIS_button.hide()

            self.current_position_label.setEnabled(True)
            self.current_position_label.show()

            self.GIS_insert_status_label.setEnabled(False)
            self.GIS_insert_status_label.hide()

            self.gas_combobox.clear()
            self.gas_combobox.addItems(self.mc_lines)
            self.gas_combobox.setCurrentText(self.mc_current_line)
        
        self.update_ui()
            

    def update_gas_protocol(self):

        self.gas_protocol["application_file"] = self.app_file_combobox.currentText()
        self.gas_protocol["beam_type"] = self.beamtype_combobox.currentText()
        self.gas_protocol["hfw"] = self.hfw_spinbox.value()*constants.MICRON_TO_METRE
        self.gas_protocol["sputter_time"] = self.timeDuration_spinbox.value()*constants.MICRO_TO_SI

    def setup_connections(self):
        self.insertGIS_button.clicked.connect(self.insert_retract_gis)
        self.gas_combobox.currentIndexChanged.connect(self.update_ui)
        self.move_GIS_button.clicked.connect(self.move_gis)
        self.GIS_radioButton.toggled.connect(self.change_gis_multichem)
        self.multichem_radioButton.toggled.connect(self.change_gis_multichem)
        self.warm_button.clicked.connect(self.warm_up_gis)
        self.run_button.clicked.connect(self.run_gis)
        self.beamtype_combobox.setCurrentText("ION")
        self.beamtype_combobox.currentIndexChanged.connect(self.update_gas_protocol)
        self.app_file_combobox.currentIndexChanged.connect(self.update_gas_protocol)
    

    def warm_up_gis(self):
        
        if self.GIS:
            line_name = self.gis_current_line
            self.microscope.GIS_heat_up(line_name)
            self.temp_label.setText("Temp: Ready")
        else:
            line_name = self.mc_current_line
            self.microscope.multichem_heat_up(line_name)
            self.temp_label.setText("Temp: Ready")

    def move_gis(self):

        if self.GIS:
            position = self.position_combobox.currentText()
            self.microscope.GIS_move_to(self.gis_current_line, position)
        else:
            position = self.position_combobox.currentText()
            self.microscope.multichem_move_to(position)
        self.update_ui()

    def update_ui(self):

        line_name = self.gas_combobox.currentText()
        if line_name == "":
            return
        if self.GIS:
            self.gis_current_line = line_name
            temp_ready = "Ready" if self.microscope.GIS_temp_ready(self.gis_current_line) else "Need Warm Up"
            self.temp_label.setText(f"Temp: {temp_ready}")
        else:
            self.mc_current_line = line_name
            temp_ready = "Ready" if self.microscope.multichem_temp_ready(self.mc_current_line) else "Need Warm Up"
            self.temp_label.setText(f"Temp: {temp_ready}")
            
        current_position_gis = self.microscope.GIS_position(self.gis_current_line)
        current_position_multichem = self.microscope.multichem_position() if isinstance(self.microscope,(ThermoMicroscope,DemoMicroscope)) else None

        current_position = current_position_gis if self.GIS else current_position_multichem

        self.position_combobox.setCurrentText(current_position)
        self.current_position_label.setText(f"Current Position: {current_position}")
        self.hfw_spinbox.setValue(self.image_widget.image_settings.hfw*constants.METRE_TO_MICRON)

    def insert_retract_gis(self):
        if self.GIS_inserted:
            self.microscope.GIS_move_to(self.gis_current_line, "Retract")
            self.GIS_inserted = False
            self.insertGIS_button.setText("Insert GIS")
            self.GIS_insert_status_label.setText("GIS Status: Retracted")
        else:
            self.microscope.GIS_move_to(self.gis_current_line, "Insert")
            self.GIS_inserted = True
            self.insertGIS_button.setText("Retract GIS")
            self.GIS_insert_status_label.setText("GIS Status: Inserted")

    def run_gis(self):

        if self.GIS:
            self.gas_protocol["gas"] = self.gis_current_line
            self.microscope.setup_GIS(self.gas_protocol)
            self.microscope.setup_GIS_pattern(self.gas_protocol)
            self.microscope.run_GIS(self.gas_protocol)
        else:
            self.gas_protocol["gas"] = self.mc_current_line
            self.microscope.run_Multichem(self.gas_protocol)
        


def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemGISWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()