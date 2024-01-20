import logging
import os

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from fibsem import constants, conversions
from fibsem.microscope import (DemoMicroscope, FibsemMicroscope,
                               TescanMicroscope, ThermoMicroscope)
from fibsem.structures import (BeamType, FibsemManipulatorPosition,
                               MicroscopeSettings)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemManipulatorWidget
from fibsem.ui.utils import message_box_ui
from fibsem.ui import _stylesheets

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
        self.manipulator_inserted = False
        


        self.setup_connections()

        self.update_ui()

        _THERMO = isinstance(self.microscope, (ThermoMicroscope,DemoMicroscope))
        _TESCAN = isinstance(self.microscope, (TescanMicroscope))


        if _THERMO:
            
            try:
                park_position = self.microscope._get_saved_manipulator_position("PARK")
                eucentric_position = self.microscope._get_saved_manipulator_position("EUCENTRIC")
                self.saved_positions["PARK"] = park_position
                self.saved_positions["EUCENTRIC"] = eucentric_position
                self.savedPosition_combobox.addItem("PARK")
                self.savedPosition_combobox.addItem("EUCENTRIC")
                
                
            except :
                napari.utils.notifications.show_warning("Error loading PARK and EUCENTRIC positions, calibration of manipulator is possibly needed.")
            
            self.manipulator_inserted = self.microscope.get_manipulator_state()
                
            self.move_type_comboBox.currentIndexChanged.connect(self.change_move_type)
            self.move_type_comboBox.setCurrentIndex(0)
            self.change_move_type()
            self.dR_spinbox.setEnabled(False)
            self.dR_spinbox.hide()
            self.dr_label.hide()
            self.calibrated_status_label.hide()

        if _TESCAN:
            
            self._initialise_calibration()
            self.move_type_comboBox.hide()
            self.beam_type_label.hide()
            self.beam_type_combobox.hide()
            # self.insertManipulator_button.hide()
            # self.manipulatorStatus_label.hide()
            self.savedPosition_combobox.addItem("Standby")
            self.savedPosition_combobox.addItem("Working")
            self.manipulator_inserted = self.microscope.get_manipulator_state(settings=self.settings)


        
        self._hide_show_buttons(self.manipulator_inserted)
        self.insertManipulator_button.setText("Retract" if self.manipulator_inserted else "Insert")
        self.manipulatorStatus_label.setText("Manipulator Status: Inserted" if self.manipulator_inserted else "Manipulator Status: Retracted")


    def _initialise_calibration(self):

        is_calibrated = self.settings.hardware.manipulator_positions["calibrated"]

        if not is_calibrated:

            self.calibrated_status_label.setText("Not Calibrated, Please run the calibration tool from the tool menu")
            self.insertManipulator_button.setEnabled(False)
            self._hide_show_buttons(show=False)

    def _check_manipulator_positions_setup(self):

        positions = self.settings.hardware.manipulator_positions

        is_calibrated = positions["calibrated"]

        if is_calibrated:
            response = message_box_ui(title="Manipulator Positions Already Calibrated", text="Manipulator Positions are already calibrated, would you like to recalibrate?")
            response = not response
        else:
            response = False

        return response
        

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
        self.insertManipulator_button.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.addSavedPosition_button.clicked.connect(self.add_saved_position)
        self.addSavedPosition_button.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.goToPosition_button.clicked.connect(self.move_to_saved_position)
        self.goToPosition_button.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
        self.moveRelative_button.clicked.connect(self.move_relative)
        self.moveRelative_button.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)

        self.pushButton_refresh_data.clicked.connect(self.refresh_data)
        self.pushButton_refresh_data.setStyleSheet(_stylesheets._GRAY_PUSHBUTTON_STYLE)


    def refresh_data(self):
        self.manipulator_inserted = self.microscope.get_manipulator_state(settings=self.settings)
        
        self._hide_show_buttons(self.manipulator_inserted)
        self.insertManipulator_button.setText("Retract" if self.manipulator_inserted else "Insert")
        self.insertManipulator_button.setStyleSheet(_stylesheets._RED_PUSHBUTTON_STYLE if self.manipulator_inserted else _stylesheets._GREEN_PUSHBUTTON_STYLE)
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
  
    def _hide_show_buttons(self,show:bool = True):

        # show = False

        self.move_type_comboBox.setVisible(show)
        self.dX_spinbox.setVisible(show)
        self.dY_spinbox.setVisible(show)
        self.dZ_spinbox.setVisible(show)
        self.dR_spinbox.setVisible(show)
        self.dz_label.setVisible(show)
        self.dx_label.setVisible(show)
        self.dy_label.setVisible(show)
        self.dr_label.setVisible(show)
        self.beam_type_combobox.setVisible(show)
        self.beam_type_label.setVisible(show)
        self.moveRelative_button.setVisible(show)
        self.addSavedPosition_button.setVisible(show)
        self.goToPosition_button.setVisible(show)
        self.savedPositionName_lineEdit.setVisible(show)
        self.savedPosition_combobox.setVisible(show)




    def insert_retract_manipulator(self):
        
        if self.manipulator_inserted:

            self.microscope.retract_manipulator()
            self.insertManipulator_button.setText("Insert")
            self.manipulatorStatus_label.setText("Manipulator Status: Retracted")
            self.insertManipulator_button.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.update_ui()
            self.manipulator_inserted = False
            self._hide_show_buttons(show=False)
        
        else:

            self.microscope.insert_manipulator()
            self.insertManipulator_button.setText("Retract")
            self.manipulatorStatus_label.setText("Manipulator Status: Inserted")
            self.insertManipulator_button.setStyleSheet(_stylesheets._RED_PUSHBUTTON_STYLE)
            self.update_ui()
            self.manipulator_inserted = True
            self._hide_show_buttons(show=True)



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

        if name in ["Parking","Standby","Working"] and isinstance(self.microscope, (TescanMicroscope)):
            self.microscope.insert_manipulator(name=name)
            position = self.microscope.get_manipulator_position()
            logging.info(f"Moved to saved position {name} at {position}")
            self.update_ui()
            return

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