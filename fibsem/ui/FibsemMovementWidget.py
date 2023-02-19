import os
from enum import Enum

import napari
import napari.utils.notifications
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from PyQt5 import QtWidgets

from fibsem.ui import utils as ui_utils
from fibsem.structures import MicroscopeSettings, BeamType, MovementMode, Point
from fibsem.ui.qtdesigner_files import FibsemMovementWidget
from fibsem import conversions, movement
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
import logging

class FibsemMovementWidget(FibsemMovementWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
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

    def setup_connections(self):

        # set ui elements
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.comboBox_movement_stage_coordinate_system.addItems(["Specimen", "Raw"])

        # buttons
        self.pushButton_move.clicked.connect(self.move_to_position)
        self.pushButton_save_position.clicked.connect(self.save_position)
        self.pushButton_continue.clicked.connect(self.continue_pressed)
        self.pushButton_auto_eucentric.clicked.connect(self.auto_eucentric_correction)

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_double_click_callbacks.append(self._double_click)
        self.image_widget.ib_layer.mouse_double_click_callbacks.append(self._double_click)

    def auto_eucentric_correction(self):

        print("auto eucentric")

    def continue_pressed(self):
        print("continue pressed")

    def save_position(self):
        print("save position pressed")

    def move_to_position(self):
        print("move_to position pressed")

    def _double_click(self, layer, event):

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

        # eucentric is only supported for ION beam
        # if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
        #     self.microscope.eucentric_move(
        #         settings=self.settings, dy=-point.y
        #     )

        # else:
        #     # corrected stage movement
        #     self.microscope.stable_move(
        #         settings=self.settings,
        #         dx=point.x,
        #         dy=point.y,
        #         beam_type=beam_type,
        #     )

        self.image_widget.take_reference_images()












def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMovementWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
