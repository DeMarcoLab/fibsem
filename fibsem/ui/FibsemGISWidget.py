import logging

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets

from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope, TescanMicroscope
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
        self.GIS_inserted = False
        self.GIS_insert_status_label.setText(f"GIS Status: inserted" if self.GIS_inserted else "GIS Status: retracted")

        if isinstance(self.microscope, TescanMicroscope):
            self.tescan_setup()


        self.setup_connections()

        self.update_ui()

    def tescan_setup(self):
        self.lines = self.microscope.GIS_available_lines()

        self.gas_combobox.addItems(self.lines.keys())
        self.current_line = self.lines[list(self.lines.keys())[0]]
        self.gas_combobox.setCurrentText(self.current_line.name)

        self.available_positions = self.microscope.GIS_available_positions()
        self.position_combobox.addItems(self.available_positions.__members__.keys())
        
        self.update_ui()

    def change_gas(self):
        line_name = self.gas_combobox.currentText()
        self.current_line = self.lines[line_name]

        self.update_ui()

    def setup_connections(self):
        self.insertGIS_button.clicked.connect(self.insert_retract_gis)
        self.gas_combobox.currentIndexChanged.connect(self.change_gas)
        self.move_GIS_button.clicked.connect(self.move_gis)
    
    def move_gis(self):

        position = self.position_combobox.currentText()
        self.microscope.GIS_move_to(self.current_line, position)
        self.update_ui()

    def update_ui(self):
        current_position = self.microscope.GIS_position(self.current_line)
        self.position_combobox.setCurrentText(current_position.name)
        self.current_position_label.setText(f"Current Position: {current_position.name}")


    def insert_retract_gis(self):
        if self.GIS_inserted:
            # self.microscope.retract_gis()
            self.GIS_inserted = False
            self.insertGIS_button.setText("Insert GIS")
            self.GIS_insert_status_label.setText("GIS Status: retracted")
        else:
            # self.microscope.insert_gis()
            self.GIS_inserted = True
            self.insertGIS_button.setText("Retract GIS")
            self.GIS_insert_status_label.setText("GIS Status: inserted")

def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemGISWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()