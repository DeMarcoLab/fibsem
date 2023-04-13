import logging

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
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
        self.manipulator_inserted = False

        self.setup_connections()

        self.update_ui()

    def update_ui(self):

        manipulator_position = self.microscope.get_manipulator_position()

        self.xCoordinate_spinbox.setValue(manipulator_position.x * constants.SI_TO_MILLI)
        self.yCoordinate_spinbox.setValue(manipulator_position.y * constants.SI_TO_MILLI)
        self.zCoordinate_spinbox.setValue(manipulator_position.z * constants.SI_TO_MILLI)
        self.rotationCoordinate_spinbox.setValue(manipulator_position.r * constants.RADIANS_TO_DEGREES)
        self.tiltCoordinate_spinbox.setValue(manipulator_position.t * constants.RADIANS_TO_DEGREES)

    def setup_connections(self):

        self.movetoposition_button.clicked.connect(self.move_to_position)
        self.insertManipulator_button.clicked.connect(self.insert_retract_manipulator)
        self.addSavedPosition_button.clicked.connect(self.add_saved_position)
        self.goToPosition_button.clicked.connect(self.move_to_saved_position)
        self.insertManipulator_button.setText("Insert")


    def move_to_position(self):

        x = self.xCoordinate_spinbox.value() * constants.MILLI_TO_SI
        y = self.yCoordinate_spinbox.value() * constants.MILLI_TO_SI
        z = self.zCoordinate_spinbox.value() * constants.MILLI_TO_SI
        r = self.rotationCoordinate_spinbox.value() * constants.DEGREES_TO_RADIANS
        t = self.tiltCoordinate_spinbox.value() * constants.DEGREES_TO_RADIANS

        position = FibsemManipulatorPosition(x=x,y=y,z=z,r=r,t=t)

         
        e = self.microscope.move_manipulator_absolute(position=position)
        if e is not None:
            ## how to show napari error message?
            error_message = f"Error moving manipulator: {str(e)}"

            napari.utils.notifications.show_error(error_message)
            self.update_ui()

    def insert_retract_manipulator(self):
        
        if self.manipulator_inserted:

            self.microscope.retract_manipulator()
            self.insertManipulator_button.setText("Insert")
            self.update_ui()
            self.manipulator_inserted = False
        
        else:

            self.microscope.insert_manipulator()
            self.insertManipulator_button.setText("Retract")
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