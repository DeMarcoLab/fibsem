import logging

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope, ThermoMicroscope, TescanMicroscope, DemoMicroscope
from fibsem.structures import (MicroscopeSettings, FibsemManipulatorPosition)
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
        _TESCAN = isinstance(self.microscope, (TescanMicroscope,DemoMicroscope))


        if _THERMO:

            park_position = self.microscope._get_saved_manipulator_position("PARK")
            eucentric_position = self.microscope._get_saved_manipulator_position("EUCENTRIC")
            
            self.saved_positions["PARK"] = park_position
            self.saved_positions["EUCENTRIC"] = eucentric_position

            self.savedPosition_combobox.addItem("PARK")
            self.savedPosition_combobox.addItem("EUCENTRIC")
            

    def update_ui(self):

        manipulator_position = self.microscope.get_manipulator_position()

        self.xCoordinate_spinbox.setValue(manipulator_position.x * constants.SI_TO_MILLI)
        self.yCoordinate_spinbox.setValue(manipulator_position.y * constants.SI_TO_MILLI)
        self.zCoordinate_spinbox.setValue(manipulator_position.z * constants.SI_TO_MILLI)
        self.rotationCoordinate_spinbox.setValue(manipulator_position.r * constants.RADIANS_TO_DEGREES)
        self.tiltCoordinate_spinbox.setValue(manipulator_position.t * constants.RADIANS_TO_DEGREES)

        self.dX_spinbox.setValue(0)
        self.dY_spinbox.setValue(0)
        self.dZ_spinbox.setValue(0)
        self.dR_spinbox.setValue(0)
        self.dT_spinbox.setValue(0)

        
    def setup_connections(self):

        self.movetoposition_button.clicked.connect(self.move_to_position)
        self.insertManipulator_button.clicked.connect(self.insert_retract_manipulator)
        self.addSavedPosition_button.clicked.connect(self.add_saved_position)
        self.goToPosition_button.clicked.connect(self.move_to_saved_position)
        self.moveRelative_button.clicked.connect(self.move_relative)
        self.insertManipulator_button.setText("Insert")
        self.manipulatorStatus_label.setText("Manipulator Status: Inserted" if self.manipulator_inserted else "Manipulator Status: Retracted")


    def move_to_position(self):

        x = self.xCoordinate_spinbox.value() * constants.MILLI_TO_SI
        y = self.yCoordinate_spinbox.value() * constants.MILLI_TO_SI
        z = self.zCoordinate_spinbox.value() * constants.MILLI_TO_SI
        r = self.rotationCoordinate_spinbox.value() * constants.DEGREES_TO_RADIANS
        t = self.tiltCoordinate_spinbox.value() * constants.DEGREES_TO_RADIANS

        position = FibsemManipulatorPosition(x=x,y=y,z=z,r=r,t=t)

         
        e = self.microscope.move_manipulator_absolute(position=position)
        if e is not None:
            error_message = f"Error moving manipulator: {str(e)}"
            napari.utils.notifications.show_error(error_message)
            self.update_ui()

        self.update_ui()

    def move_relative(self):

        x = self.dX_spinbox.value() * constants.MILLI_TO_SI
        y = self.dY_spinbox.value() * constants.MILLI_TO_SI
        z = self.dZ_spinbox.value() * constants.MILLI_TO_SI
        r = self.dR_spinbox.value() * constants.DEGREES_TO_RADIANS
        t = self.dT_spinbox.value() * constants.DEGREES_TO_RADIANS

        position = FibsemManipulatorPosition(x=x,y=y,z=z,r=r,t=t)

        e = self.microscope.move_manipulator_relative(position=position)
        if e is not None:
            error_message = f"Error moving manipulator: {str(e)}"
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
        position = FibsemManipulatorPosition(
            x=self.xCoordinate_spinbox.value() * constants.MILLI_TO_SI,
            y=self.yCoordinate_spinbox.value() * constants.MILLI_TO_SI,
            z=self.zCoordinate_spinbox.value() * constants.MILLI_TO_SI,
            r=self.rotationCoordinate_spinbox.value() * constants.DEGREES_TO_RADIANS,
            t=self.tiltCoordinate_spinbox.value() * constants.DEGREES_TO_RADIANS,
            coordinate_system='STAGE'
        )
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