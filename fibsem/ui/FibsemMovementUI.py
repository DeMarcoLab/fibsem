import logging
from enum import Enum

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment
from fibsem.structures import BeamType, MicroscopeSettings, Point
from fibsem.ui.qtdesigner_files import movement_dialog 
from PyQt5 import QtCore, QtWidgets
import scipy.ndimage as ndi
import traceback

class MovementMode(Enum):
    Stable = 1
    Eucentric = 2
    # Needle = 3

class MovementType(Enum):
    StableEnabled = 0 
    EucentricEnabled = 1
    TiltEnabled = 2

# TODO: save state...?
# TODO: focus and link?

import napari

class FibsemMovementUI(movement_dialog.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope: SdbMicroscopeClient,
        settings: MicroscopeSettings,
        msg_type: str = None,
        msg: str = None,
        parent=None,
        viewer: napari.Viewer = None
    ):
        super(FibsemMovementUI, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.destination_points = (Point(), Point())

        # TODO: add user defined initial movement mode

        # msg
        self.msg_type = msg_type
        self.msg = msg
        self.movement_mode = MovementMode.Stable

        # enable / disable movement
        self.tilt_movement_enabled = False

        if msg_type in ["alignment"]:
            self.tilt_movement_enabled = True

        self.setup_connections()
        self.set_message(self.msg_type, self.msg)
        self.update_displays()

    def update_displays(self):
        """Update the displays for both Electron and Ion Beam views"""

        try:
            # update settings, take image
            self.settings.image.hfw = self.doubleSpinBox_hfw.value() * constants.MICRON_TO_METRE
            self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.settings.image)
            self.image = np.concatenate([self.eb_image.data, self.ib_image.data], axis=1) # stack both images together

            # median filter
            self.image = ndi.median_filter(self.image, size=3)

            # TODO: convert this to use grid layout instead of concat images (see salami)

            # crosshair
            cy, cx_eb = self.image.shape[0] // 2, self.image.shape[1] // 4 
            cx_ib = cx_eb + self.image.shape[1] // 2 
            
            # # refresh viewer
            self.viewer.layers.clear()
            self.image_layer = self.viewer.add_image(self.image, name="Images", opacity=0.9, blending="additive")
            self.points_layer = self.viewer.add_points(
                data=[[cy, cx_eb], [cy, cx_ib]], 
                symbol="cross", size=50,
                edge_color="yellow", face_color="yellow",
            )
            self.points_layer.editable = False

            self.image_layer.mouse_double_click_callbacks.append(self._double_click) # append callback
            # self.image_layer.mouse_drag_callbacks.append(self._single_click) # append callback
            self.viewer.layers.selection.active = self.image_layer

        except:
            napari.utils.notifications.show_info(f"Unable to update movement image: {traceback.format_exc()}")

    def get_data_from_coord(self, coords: tuple) -> tuple:

        # check inside image dimensions, (y, x)
        eb_shape = self.image.data.shape[0], self.image.data.shape[1] // 2
        ib_shape = self.image.data.shape[0], self.image.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (coords[1] > 0 and coords[1] < eb_shape[1]):
            adorned_image = self.eb_image
            beam_type = BeamType.ELECTRON

        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (coords[1] > eb_shape[0] and coords[1] < ib_shape[1]):
            adorned_image = self.ib_image
            coords = (coords[0], coords[1] - ib_shape[1] // 2)
            beam_type = BeamType.ION
        else:
            beam_type, adorned_image = None, None
        
        return coords, beam_type, adorned_image

    # def _single_click(self, layer, event):
        
    #     if event.type == "mouse_press" and self.movement_mode is MovementMode.Needle:
    #         pass # TODO:
            # get coords
            # coords = layer.world_to_data(event.position)

            # beam_type, adorned_image = self.get_data_from_coord(coords)

            # coord_type = self.comboBox_needle_coordinate.currentText()

            # logging.info(f"Layer: {layer}, Event: {event}")
            # logging.info(f"Coords: {coords}, Type: {coord_type}")
            # logging.info(f"BeamType: {beam_type}, ")

    def _double_click(self, layer, event):

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        
        coords, beam_type, adorned_image = self.get_data_from_coord(coords)

        if beam_type is None:
            napari.utils.notifications.show_info(f"Clicked outside image dimensions. Please click inside the image to move.")
            return

        point = conversions.image_to_microscope_image_coordinates(Point(x=coords[1], y=coords[0]), 
                adorned_image.data, adorned_image.metadata.binary_result.pixel_size.x)  

        logging.debug(f"Movement: {self.movement_mode.name} | COORD {coords} | SHIFT {point.x:.2e}, {point.y:.2e} | {beam_type}")

        # move
        self.movement_mode = MovementMode[self.comboBox_movement_mode.currentText()]

        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            movement.move_stage_eucentric_correction(
                microscope=self.microscope, 
                dy=-point.y
            )

        else:
            # corrected stage movement
            movement.move_stage_relative_with_corrected_movement(
                microscope=self.microscope,
                settings=self.settings,
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )

        self.update_displays()

    def setup_connections(self):

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)
        self.pushButton_take_image.clicked.connect(self.take_image_button_pressed)

        self.doubleSpinBox_hfw.setMinimum(30e-6 * constants.METRE_TO_MICRON)
        self.doubleSpinBox_hfw.setMaximum(900e-6 * constants.METRE_TO_MICRON)
        self.doubleSpinBox_hfw.setValue(self.settings.image.hfw * constants.METRE_TO_MICRON)

        # movement modes
        self.comboBox_movement_mode.clear()
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.comboBox_movement_mode.currentTextChanged.connect(
            self.movement_mode_changed
        )
        self.pushButton_auto_eucentric.clicked.connect(self.auto_eucentric_button_pressed)
        self.comboBox_needle_coordinate.addItems(["Source", "Destination"])

        # DESTINATION_MODE = self.movement_mode is MovementMode.Needle
        DESTINATION_MODE = False
        self.label_needle_coordinate.setVisible(DESTINATION_MODE)
        self.comboBox_needle_coordinate.setVisible(DESTINATION_MODE)


        # tilt functionality
        self.doubleSpinBox_tilt_degrees.setMinimum(0)
        self.doubleSpinBox_tilt_degrees.setMaximum(25.0)
        self.pushButton_tilt_stage.setVisible(self.tilt_movement_enabled)
        self.doubleSpinBox_tilt_degrees.setVisible(self.tilt_movement_enabled)
        self.pushButton_tilt_stage.setEnabled(self.tilt_movement_enabled)
        self.doubleSpinBox_tilt_degrees.setEnabled(self.tilt_movement_enabled)
        self.label_tilt.setVisible(self.tilt_movement_enabled)
        self.label_header_tilt.setVisible(self.tilt_movement_enabled)
        if self.tilt_movement_enabled:
            self.pushButton_tilt_stage.clicked.connect(self.stage_tilt)

    def movement_mode_changed(self):

        mode_name = self.comboBox_movement_mode.currentText()
        self.movement_mode = MovementMode[mode_name]

        logging.info(f"changed mode to: {self.movement_mode}")

        # set instruction message
        self.set_message(self.msg_type, self.msg)

        # DESTINATION_MODE = self.movement_mode is MovementMode.Needle
        # self.label_needle_coordinate.setVisible(DESTINATION_MODE)
        # self.comboBox_needle_coordinate.setVisible(DESTINATION_MODE)

    def set_message(self, msg_type: str, msg: str = None):
            
        # set message
        msg_dict = {
            "eucentric": "Please centre a feature in both Beam views (Double click to move). ",
            "alignment": "Please centre the lamella in the Ion Beam, and tilt so the lamella face is perpendicular to the Ion Beam.",
        }
        if msg is None:
            msg = msg_dict[msg_type]

        if self.movement_mode is MovementMode.Eucentric:
            self.label_message.setText("Centre a feature in the Electron Beam, then double click the same feature in the Ion Beam.") 

        if self.movement_mode is MovementMode.Stable:
            self.label_message.setText(msg)

    def take_image_button_pressed(self):
        """Take a new image with the current image settings."""

        self.update_displays()

    def continue_button_pressed(self):
        self.close()

    def closeEvent(self, event):
        event.accept()

    def stage_tilt(self):
        """Tilt the stage to the desired angle

        Args:
            stage_tilt (float): desired stage tilt angle (degrees)
        """

        stage_tilt_rad: float = np.deg2rad(self.doubleSpinBox_tilt_degrees.value())
        stage = self.microscope.specimen.stage

        move_settings = MoveSettings(rotate_compucentric=True, tilt_compucentric=True)
        stage_position = StagePosition(t=stage_tilt_rad)
        stage.absolute_move(stage_position, move_settings)

        # update displays
        self.update_displays()

    def auto_eucentric_button_pressed(self):

        alignment.auto_eucentric_correction(self.microscope, self.settings.image)

        self.update_displays()

def main():
    from fibsem import utils
    from fibsem.ui import windows as fibsem_ui_windows
    microscope, settings= utils.setup_session()


    fibsem_ui_windows.ask_user_movement(
        microscope,
        settings,
        msg_type="eucentric",
        msg="Move around",

    )

if __name__ == "__main__":
    main()
