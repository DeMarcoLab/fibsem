import logging

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope, ThermoMicroscope, TescanMicroscope, DemoMicroscope
from fibsem.structures import (MicroscopeSettings, FibsemManipulatorPosition,BeamType)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemManipulatorWidget
from fibsem.ui.utils import message_box_ui


class FibsemManipulatorWidget(FibsemManipulatorWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        image_widget: FibsemImageSettingsWidget = None, 
        parent=None,
    ):
        super(FibsemManipulatorWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget
        self.saved_positions = {}
        self.manipulator_inserted = False # TODO need to read from microscope if manipulator is inserted or not??
        


        self.setup_connections()

        self.update_ui()

        _THERMO = isinstance(self.microscope, (ThermoMicroscope,DemoMicroscope))
        _TESCAN = isinstance(self.microscope, (TescanMicroscope))


        if _THERMO:
            
            try:
                park_position = self.microscope._get_saved_manipulator_position("PARL")
                eucentric_position = self.microscope._get_saved_manipulator_position("EUCENTRIC")
                self.saved_positions["PARK"] = park_position
                self.saved_positions["EUCENTRIC"] = eucentric_position
                self.savedPosition_combobox.addItem("PARK")
                self.savedPosition_combobox.addItem("EUCENTRIC")
                
            except :
                
                message_box_ui(title="Error loading positions",
                    text="Error loading PARK and EUCENTRIC positions, calibration of manipulator is possibly needed.",
                    buttons=QMessageBox.Ok)
                
            




            self.move_type_comboBox.currentIndexChanged.connect(self.change_move_type)
            self.move_type_comboBox.setCurrentIndex(0)
            self.change_move_type()
            self.dR_spinbox.setEnabled(False)
            self.dR_spinbox.hide()
            self.dr_label.hide()

        if _TESCAN:
            
            self.move_type_comboBox.hide()
            self.beam_type_label.hide()
            self.beam_type_combobox.hide()
            self.insertManipulator_button.hide()
            self.manipulatorStatus_label.hide()



    def change_move_type(self):

        if self.move_type_comboBox.currentText() == "Relative Move":
            self.dZ_spinbox.setEnabled(True)
            self.dZ_spinbox.show()
            self.dz_label.show()
            self.beam_type_label.hide()
            self.beam_type_combobox.hide()
        else:
            self.dZ_spinbox.setEnabled(False)
            self.dZ_spinbox.hide()
            self.dz_label.hide()
            self.beam_type_label.show()
            self.beam_type_combobox.show()

       

    def update_ui(self):

        self.dX_spinbox.setValue(0)
        self.dY_spinbox.setValue(0)
        self.dZ_spinbox.setValue(0)
        self.dR_spinbox.setValue(0)


        
    def setup_connections(self):

        self.insertManipulator_button.clicked.connect(self.insert_retract_manipulator)
        self.addSavedPosition_button.clicked.connect(self.add_saved_position)
        self.goToPosition_button.clicked.connect(self.move_to_saved_position)
        self.moveRelative_button.clicked.connect(self.move_relative)
        self.insertManipulator_button.setText("Insert")
        self.manipulatorStatus_label.setText("Manipulator Status: Inserted" if self.manipulator_inserted else "Manipulator Status: Retracted")



    def move_relative(self):

        if self.move_type_comboBox.currentText() == "Relative Move" or isinstance(self.microscope, (TescanMicroscope)):
            x = self.dX_spinbox.value() * constants.MICRO_TO_SI
            y = self.dY_spinbox.value() * constants.MICRO_TO_SI
            z = self.dZ_spinbox.value() * constants.MICRO_TO_SI
            r = self.dR_spinbox.value() * constants.DEGREES_TO_RADIANS


            position = FibsemManipulatorPosition(x=x,y=y,z=z,r=r, coordinate_system="STAGE")

            e = self.microscope.move_manipulator_relative(position=position)
            if e is not None:
                error_message = f"Error moving manipulator (Relative): {str(e)}"
                napari.utils.notifications.show_error(error_message)
                self.update_ui()
            
            self.update_ui()
        else:
            beam_type = getattr(BeamType,self.beam_type_combobox.currentText())
            dx = self.dX_spinbox.value() * constants.MICRO_TO_SI
            dy = self.dY_spinbox.value() * constants.MICRO_TO_SI
            e = self.microscope.move_manipulator_corrected(dx=dx,dy=dy,beam_type=beam_type)
            if e is not None:
                error_message = f"Error moving manipulator (Corrected): {str(e)}"
                napari.utils.notifications.show_error(error_message)
                self.update_ui()
            self.update_ui()      
  
        

    def insert_retract_manipulator(self):
        
        if self.manipulator_inserted:

            self.microscope.retract_manipulator()
            self.insertManipulator_button.setText("Insert")
            self.manipulatorStatus_label.setText("Manipulator Status: Retracted")
            self.update_ui()
            self.manipulator_inserted = False
        
        else:

            self.microscope.insert_manipulator()
            self.insertManipulator_button.setText("Retract")
            self.manipulatorStatus_label.setText("Manipulator Status: Inserted")
            self.update_ui()
            self.manipulator_inserted = True



    def add_saved_position(self):

        if self.savedPositionName_lineEdit.text() == "":
            _ = message_box_ui(
                title="No name.",
                text="Please enter a position name.",
                buttons=QMessageBox.Ok,
            )
            return
        name = self.savedPositionName_lineEdit.text()
        position = self.microscope.get_manipulator_position()
        self.saved_positions[name] = position
        logging.info(f"Saved position {name} at {position}")
        self.savedPosition_combobox.addItem(name)
        self.savedPositionName_lineEdit.clear()

    def move_to_saved_position(self):
        name = self.savedPosition_combobox.currentText()
        position = self.saved_positions[name]
        logging.info(f"Moving to saved position {name} at {position}")
        self.microscope.move_manipulator_absolute(position=position)
        self.update_ui()



def main():

    viewer = napari.Viewer(ndisplay=2)
    manipulator_widget = FibsemManipulatorWidget()
    viewer.window.add_dock_widget(
        manipulator_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()