import logging
import sys
from enum import Enum

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment
from fibsem.structures import BeamType, MicroscopeSettings
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.qtdesigner_files import movement_dialog as movement_gui
from PyQt5 import QtCore, QtWidgets


class MovementMode(Enum):
    Stable = 1
    Eucentric = 2


class MovementType(Enum):
    StableEnabled = 0 
    EucentricEnabled = 1
    TiltEnabled = 2

# TODO: save state...?
# TODO: focus and link?


class GUIMMovementWindow(movement_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope: SdbMicroscopeClient,
        settings: MicroscopeSettings,
        msg_type: str = None,
        msg: str = None,
        parent=None,
    ):
        super(GUIMMovementWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope
        self.settings = settings

        self.wp_ib = None
        self.wp_eb = None

        # msg
        self.msg_type = msg_type
        self.msg = msg
        self.movement_mode = MovementMode.Stable

        # enable / disable movement
        self.tilt_movement_enabled = False
        self.eb_movement_enabled = True
        self.ib_movement_enabled = True

        if msg_type in ["alignment"]:
            self.tilt_movement_enabled = True

        self.setup_connections()
        self.set_message(self.msg_type, self.msg)
        self.update_displays()

    def update_displays(self):
        """Update the displays for both Electron and Ion Beam views"""

        logging.info("updating displays for Electron and Ion beam views")
        self.eb_image, self.ib_image = acquire.take_reference_images(
            self.microscope, self.settings.image
        )

        # median filter image for better display
        eb_image_smooth = ndi.median_filter(self.eb_image.data, size=3)
        ib_image_smooth = ndi.median_filter(self.ib_image.data, size=3)

        # update eb view
        if self.wp_eb is not None:
            self.label_image_eb.layout().removeWidget(self.wp_eb)
            # TODO: remove layouts properly
            self.wp_eb.deleteLater()

        self.wp_eb = fibsem_ui._WidgetPlot(self, display_image=eb_image_smooth)
        self.label_image_eb.setLayout(QtWidgets.QVBoxLayout())
        self.label_image_eb.layout().addWidget(self.wp_eb)

        # update ib view
        if self.wp_ib is not None:
            self.label_image_ib.layout().removeWidget(self.wp_ib)
            # TODO: remove layouts properly
            self.wp_ib.deleteLater()

        self.wp_ib = fibsem_ui._WidgetPlot(self, display_image=ib_image_smooth)
        self.label_image_ib.setLayout(QtWidgets.QVBoxLayout())
        self.label_image_ib.layout().addWidget(self.wp_ib)

        # draw crosshairs on both images
        fibsem_ui.draw_crosshair(self.eb_image, self.wp_eb.canvas)
        fibsem_ui.draw_crosshair(self.ib_image, self.wp_ib.canvas)

        self.wp_eb.canvas.ax11.set_title("Electron Beam")
        self.wp_ib.canvas.ax11.set_title("Ion Beam")

        self.wp_eb.canvas.draw()
        self.wp_ib.canvas.draw()

        # reconnect buttons
        if self.eb_movement_enabled:
            self.wp_eb.canvas.mpl_connect("button_press_event", self.on_click)

        if self.ib_movement_enabled:
            self.wp_ib.canvas.mpl_connect("button_press_event", self.on_click)

    def setup_connections(self):

        logging.info("setup connections")

        self.pushButton_continue.clicked.connect(self.continue_button_pressed)
        self.pushButton_take_image.clicked.connect(self.take_image_button_pressed)

        self.doubleSpinBox_hfw.setMinimum(30e-6 * constants.METRE_TO_MICRON) # TODO: dynamic limits
        self.doubleSpinBox_hfw.setMaximum(900e-6 * constants.METRE_TO_MICRON)
        self.doubleSpinBox_hfw.setValue(self.settings.image.hfw * constants.METRE_TO_MICRON)
        self.doubleSpinBox_hfw.valueChanged.connect(self.update_image_settings)

        # movement modes
        self.comboBox_movement_mode.clear()
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.comboBox_movement_mode.currentTextChanged.connect(
            self.movement_mode_changed
        )
        self.pushButton_auto_eucentric.clicked.connect(self.auto_eucentric_button_pressed)

        # tilt functionality
        self.doubleSpinBox_tilt_degrees.setMinimum(0)
        self.doubleSpinBox_tilt_degrees.setMaximum(25.0)
        if self.tilt_movement_enabled:
            self.pushButton_tilt_stage.setVisible(True)
            self.doubleSpinBox_tilt_degrees.setVisible(True)
            self.pushButton_tilt_stage.setEnabled(True)
            self.doubleSpinBox_tilt_degrees.setEnabled(True)
            self.label_tilt.setVisible(True)
            self.label_header_tilt.setVisible(True)
            self.pushButton_tilt_stage.clicked.connect(self.stage_tilt)
        else:
            self.label_tilt.setVisible(False)
            self.label_header_tilt.setVisible(False)
            self.pushButton_tilt_stage.setVisible(False)
            self.doubleSpinBox_tilt_degrees.setVisible(False)
            self.pushButton_tilt_stage.setVisible(False)
            self.doubleSpinBox_tilt_degrees.setVisible(False)

    def movement_mode_changed(self):

        mode_name = self.comboBox_movement_mode.currentText()
        self.movement_mode = MovementMode[mode_name]

        logging.info(f"changed mode to: {self.movement_mode}")

        # set instruction message
        self.set_message(self.msg_type, self.msg)

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

    def update_image_settings(self):
        """Update the image settings when ui elements change"""

        self.settings.image.hfw = self.doubleSpinBox_hfw.value() * constants.MICRON_TO_METRE

    def continue_button_pressed(self):
        logging.info("continue button pressed")
        self.close()

    def closeEvent(self, event):
        logging.info("closing movement window")
        event.accept()

    def on_click(self, event):
        """Move to the selected position on user double click"""

        if event.inaxes is self.wp_ib.canvas.ax11:
            beam_type = BeamType.ION
            adorned_image = self.ib_image

        if event.inaxes is self.wp_eb.canvas.ax11:
            beam_type = BeamType.ELECTRON
            adorned_image = self.eb_image

        if event.button == 1 and event.inaxes is not None:
            self.xclick = event.xdata
            self.yclick = event.ydata
            self.center_x, self.center_y = conversions.pixel_to_realspace_coordinate(
                (self.xclick, self.yclick), adorned_image
            )

            # draw crosshair?
            if event.dblclick:
                logging.info(f"Movement, {beam_type}, p=({self.xclick}, {self.yclick})  c=({self.center_x:.2e}, {self.center_y:.2e}) ")

                self.stage_movement(beam_type=beam_type)

    def stage_movement(self, beam_type: BeamType):

        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            logging.info(f"moving eucentricly in {beam_type}")

            movement.move_stage_eucentric_correction(
                microscope=self.microscope, 
                dy=-self.center_y
            )

        else:
            # corrected stage movement
            movement.move_stage_relative_with_corrected_movement(
                microscope=self.microscope,
                settings=self.settings,
                dx=self.center_x,
                dy=self.center_y,
                beam_type=beam_type,
            )

        # update displays
        self.update_displays()

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


# TODO: override enter


def main():
    from fibsem import utils
    from fibsem.ui import windows as fibsem_ui_windows
    microscope, settings= utils.setup_session()


    app = QtWidgets.QApplication([])
    fibsem_ui_windows.ask_user_movement(
        microscope,
        settings,
        msg_type="eucentric",
        msg="Move around",
    )

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
