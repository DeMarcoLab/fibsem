import logging
import traceback

import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer


from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamType, FibsemStagePosition,
                               MicroscopeSettings, MovementMode, Point)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMovementWidget

def log_status_message(step: str):
    logging.debug(
        f"STATUS | Movement Widget | {step}"
    )


class FibsemMovementWidget(FibsemMovementWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        image_widget: FibsemImageSettingsWidget = None, 
        parent=None,
    ):
        super(FibsemMovementWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget

        self.setup_connections()
        self.image_widget.picture_signal.connect(self.update_ui)

   
        self.update_ui()

    def setup_connections(self):

        # set ui elements
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.comboBox_movement_stage_coordinate_system.addItems(["SPECIMEN", "RAW"])

        # buttons
        self.pushButton_move.clicked.connect(self.move_to_position)
        self.pushButton_save_position.clicked.connect(self.save_position)
        self.pushButton_continue.clicked.connect(self.continue_pressed)
        self.pushButton_auto_eucentric.clicked.connect(self.auto_eucentric_correction)

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_double_click_callbacks.append(self._double_click)
        self.image_widget.ib_layer.mouse_double_click_callbacks.append(self._double_click)

        # disable ui elements
        self.label_movement_instructions.setText("Double click to move.")
        self.pushButton_continue.setVisible(False)
        self.pushButton_save_position.setVisible(False)
        self.pushButton_auto_eucentric.setVisible(False)

    def auto_eucentric_correction(self):

        print("auto eucentric")

    def continue_pressed(self):
        print("continue pressed")

    def save_position(self):
        print("save position pressed")

    def move_to_position(self):
        stage_position = self.get_position_from_ui()
        self.microscope.move_stage_absolute(stage_position)
        log_status_message(f"MOVED_TO_{stage_position}")
        self.update_ui_after_movement()
    
    def update_ui(self):

        stage_position: FibsemStagePosition = self.microscope.get_stage_position()

        self.doubleSpinBox_movement_stage_x.setValue(stage_position.x * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_y.setValue(stage_position.y * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_z.setValue(stage_position.z * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_rotation.setValue(np.rad2deg(stage_position.r))
        self.doubleSpinBox_movement_stage_tilt.setValue(np.rad2deg(stage_position.t))
    
    def get_position_from_ui(self):

        stage_position = FibsemStagePosition(
            x=self.doubleSpinBox_movement_stage_x.value() * constants.MILLI_TO_SI,
            y=self.doubleSpinBox_movement_stage_y.value() * constants.MILLI_TO_SI,
            z=self.doubleSpinBox_movement_stage_z.value() * constants.MILLI_TO_SI,
            r=np.deg2rad(self.doubleSpinBox_movement_stage_rotation.value()),
            t=np.deg2rad(self.doubleSpinBox_movement_stage_tilt.value()),
            coordinate_system=self.comboBox_movement_stage_coordinate_system.currentText(),

        )

        return stage_position

    def _double_click(self, layer, event):
        
        if event.button != 1:
            return

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)

        if beam_type is None:
            napari.utils.notifications.show_info(
                f"Clicked outside image dimensions. Please click inside the image to move."
            )
            return

        point = conversions.image_to_microscope_image_coordinates(
            Point(x=coords[1], y=coords[0]), image.data, image.metadata.pixel_size.x,
        )

        # move
        self.movement_mode = MovementMode[self.comboBox_movement_mode.currentText()]

        logging.debug(
            f"Movement: {self.movement_mode.name} | COORD {coords} | SHIFT {point.x:.2e}, {point.y:.2e} | {beam_type}"
        )
        log_status_message(f"MOVING_{self.movement_mode.name}_BY_{point.x:.2e}, {point.y:.2e} | {beam_type}")
        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            self.microscope.eucentric_move(
                settings=self.settings, dy=-point.y
            )

        else:
            # corrected stage movement
            self.microscope.stable_move(
                settings=self.settings,
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )
        
        self.update_ui_after_movement()

    def update_ui_after_movement(self):
        # disable taking images after movement here
        if self.checkBox_movement_acquire_electron.isChecked():
            self.image_widget.take_image(BeamType.ELECTRON)
        if self.checkBox_movement_acquire_ion.isChecked():
            self.image_widget.take_image(BeamType.ION)
        self.update_ui()

def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMovementWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
