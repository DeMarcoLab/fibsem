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
        

        self.gis_and_mc_setup()

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


        if isinstance(self.microscope, TescanMicroscope):
            self.tescan_setup()
        if isinstance(self.microscope, (ThermoMicroscope, DemoMicroscope)):
            
            protocol = utils.load_yaml(cfg.PROTOCOL_PATH)
            self.protocol = protocol
            self.thermo_setup()


    def tescan_setup(self):
        
        self.GIS_insert_status_label.hide()
        self.insertGIS_button.setEnabled(False)
        self.insertGIS_button.hide()
        self.GIS_radioButton.hide()
        self.multichem_radioButton.hide()
        self.app_file_lineEdit.hide()
        self.app_file_label.hide()
        self.GIS = True
        

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
            

    def thermo_setup(self):

        self.mc_lines = self.microscope.multichem_available_lines()
        self.mc_current_line = self.mc_lines[0]
        self.mc_available_positions = self.microscope.multichem_available_positions()
        self.app_file_lineEdit.setText(self.protocol['milling']['application_file'])

        self.GIS_inserted = True
        self.insert_retract_gis()
        self.GIS = True
        self.GIS_radioButton.setChecked(True)
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


        self.update_ui()

    # def change_gas(self):
    #     line_name = self.gas_combobox.currentText()
    #     if self.GIS:
    #         self.gis_current_line = line_name
    #     else:
    #         self.mc_current_line = line_name

    #     self.update_ui()

    def setup_connections(self):
        self.insertGIS_button.clicked.connect(self.insert_retract_gis)
        self.gas_combobox.currentIndexChanged.connect(self.update_ui)
        self.move_GIS_button.clicked.connect(self.move_gis)
        self.GIS_radioButton.toggled.connect(self.change_gis_multichem)
        self.multichem_radioButton.toggled.connect(self.change_gis_multichem)
        self.warm_button.clicked.connect(self.warm_up_gis)
    

    def warm_up_gis(self):
        
        line_name = self.gis_current_line
        self.microscope.GIS_heat_up(line_name)
        self.temp_label.setText("Temp: Ready")
        self.temp_ready = True
        self.warm_button.setEnabled(False)

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
            self.warm_button.setEnabled(not self.microscope.GIS_temp_ready(self.gis_current_line))
        else:
            self.mc_current_line = line_name

        current_position_gis = self.microscope.GIS_position(self.gis_current_line)
        current_position_multichem = self.microscope.multichem_position() if isinstance(self.microscope,(ThermoMicroscope,DemoMicroscope)) else None

        current_position = current_position_gis if self.GIS else current_position_multichem

        self.position_combobox.setCurrentText(current_position)
        self.current_position_label.setText(f"Current Position: {current_position}")


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

def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemGISWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()