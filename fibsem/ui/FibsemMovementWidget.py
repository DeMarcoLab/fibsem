import io
import logging
import os
import traceback
from copy import deepcopy
from pathlib import Path

import napari
import napari.utils.notifications
import numpy as np
import yaml
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap

from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamType, FibsemStagePosition,
                               MicroscopeSettings, MovementMode, Point)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMovementWidget
from fibsem.ui.utils import _get_file_ui, _get_save_file_ui


def log_status_message(step: str):
    logging.debug(
        f"STATUS | Movement Widget | {step}"
    )


class FibsemMovementWidget(FibsemMovementWidget.Ui_Form, QtWidgets.QWidget):
    move_signal = QtCore.pyqtSignal()
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
        self.positions = []
   
        self.update_ui()

    def setup_connections(self):

        # set ui elements
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.comboBox_movement_stage_coordinate_system.addItems(["SPECIMEN", "RAW"])

        # buttons
        self.pushButton_move.clicked.connect(self.move_to_position)
        self.pushButton_continue.clicked.connect(self.continue_pressed)
        self.pushButton_move_flat_ion.clicked.connect(self.move_flat_to_beam)
        self.pushButton_move_flat_electron.clicked.connect(self.move_flat_to_beam)

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_double_click_callbacks.append(self._double_click)
        self.image_widget.ib_layer.mouse_double_click_callbacks.append(self._double_click)

        # disable ui elements
        self.label_movement_instructions.setText("Double click to move.")
        self.pushButton_continue.setVisible(False)
        self.comboBox_movement_stage_coordinate_system.setVisible(False)
        self.label_movement_stage_coordinate_system.setVisible(False)

        # positions
        self.comboBox_positions.currentIndexChanged.connect(self.select_position)
        self.pushButton_save_position.clicked.connect(self.add_position)
        self.pushButton_remove_position.clicked.connect(self.delete_position)
        self.pushButton_go_to.clicked.connect(self.go_to_saved_position)
        self.pushButton_export.clicked.connect(self.export_positions)
        self.pushButton_import.clicked.connect(self.import_positions)
        self.pushButton_update_position.clicked.connect(self.update_saved_position)

    def auto_eucentric_correction(self):

        print("auto eucentric")

    def continue_pressed(self):
        print("continue pressed")

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
        
        if event.button != 1 or "Shift" in event.modifiers:
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
        if "Alt" in event.modifiers:
            self.movement_mode = MovementMode.Eucentric
        else:
            self.movement_mode = MovementMode[self.comboBox_movement_mode.currentText()]

        logging.debug(
            f"Movement: {self.movement_mode.name} | COORD {coords} | SHIFT {point.x:.2e}, {point.y:.2e} | {beam_type}"
        )
        log_status_message(f"MOVING_{self.movement_mode.name}_BY_{point.x:.2e}, {point.y:.2e} | {beam_type}")
        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            self.microscope.eucentric_move(
                settings=self.settings, dx=point.x, dy=-point.y
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

    def select_position(self):
        if self.comboBox_positions.currentIndex() != -1:
            position = self.positions[self.comboBox_positions.currentIndex()]
            self.label_current_position.setText(f"x={position.x*constants.METRE_TO_MILLIMETRE:.3f}, y={position.y*constants.METRE_TO_MILLIMETRE:.3f}, z={position.z*constants.METRE_TO_MILLIMETRE:.3f}, r={position.r*constants.RADIANS_TO_DEGREES:.1f}, t={position.t*constants.RADIANS_TO_DEGREES:.1f}")

    def add_position(self):
        position = self.microscope.get_stage_position()
        name = self.lineEdit_position_name.text()
        if name == "":
            napari.utils.notifications.show_warning("Please enter a name for the position")
            return
        position.name = name
        self.positions.append(position)
        self.comboBox_positions.addItem(name)
        self.comboBox_positions.setCurrentIndex(self.comboBox_positions.count() - 1)
        self.lineEdit_position_name.setText("")
        logging.info(f"Added position {name}")

    def delete_position(self):
        del self.positions[self.comboBox_positions.currentIndex()]
        name = self.comboBox_positions.currentIndex()
        self.comboBox_positions.removeItem(self.comboBox_positions.currentIndex())
        logging.info(f"Removed position {name}")

    def update_saved_position(self):
        position = self.microscope.get_stage_position()
        position.name = self.comboBox_positions.currentText()
        self.positions[self.comboBox_positions.currentIndex()] = position
        self.select_position()
        logging.info(f"Updated position {self.comboBox_positions.currentText()}")

    def go_to_saved_position(self):
        self.microscope.move_stage_absolute(self.positions[self.comboBox_positions.currentIndex()])
        self.update_ui_after_movement()
        logging.info(f"Moved to position {self.comboBox_positions.currentIndex()}")

    def export_positions(self):
        protocol_path = _get_save_file_ui(msg="Select or create file")
        if protocol_path == '':
            return
        dict_position = []
        for position in self.positions:
            dict_position.append(position.__to_dict__())
        with open(os.path.join(Path(protocol_path).with_suffix(".yaml")), "w") as f:
            yaml.safe_dump(dict_position, f, indent=4, default_flow_style=False)

        logging.info("Positions saved to file")


    def import_positions(self):
        protocol_path = _get_file_ui(msg="Select or create file")
        if protocol_path == '':
            napari.utils.notifications.show_info("No file selected, positions not loaded")
            return
        with open(protocol_path, "r") as f:
            dict_positions = yaml.safe_load(f)
        for dict_position in dict_positions:
            position = FibsemStagePosition.__from_dict__(dict_position)
            self.positions.append(position)
            self.comboBox_positions.addItem(position.name)

    # def minimap(self):
    #     x = []
    #     y = []
    #     labels = []
    #     pil_image = None
    #     current_position = self.microscope.get_stage_position()
    #     x.append(deepcopy(current_position.x)*constants.SI_TO_MICRO)
    #     y.append(deepcopy(current_position.y)*constants.SI_TO_MICRO)
    #     labels.append("Current Position")
    #     for position in self.positions:
    #         x.append(deepcopy(position.x)*constants.SI_TO_MICRO)
    #         y.append(deepcopy(position.y)*constants.SI_TO_MICRO)
    #         labels.append(deepcopy(position.name))
    #     import pandas as pd
    #     df = pd.DataFrame({'x': x, 'y': y, 'labels': labels})
    #     import plotly.express as px
    #     import plotly.io as pio
    #     fig = px.scatter(df, color="labels", labels={'color': 'Position'}, x = 'x', y = 'y', width=400, height=400)
    #     fig.update_traces(
    #             marker=dict(size=8, symbol="cross"),
    #             selector=dict(mode="markers"),
    #         )
        
    #     if self.checkBox_auto_scaling.isChecked():
    #         fig.update_layout(legend=dict(
    #             orientation="h",
    #             yanchor="bottom",
    #             y=1.02,
    #             xanchor="right",
    #             x=1
    #         ),
    #         margin=dict(l=5, r=5, t=5, b=5),
    #         legend_title_text=None,
    #         xaxis_title=None,
    #             yaxis_title=None,
    #         )
            
    #     else:
    #         range = [-self.spinBox_grid_radius.value(), self.spinBox_grid_radius.value()]
    #         fig.update_layout(legend=dict(
    #             orientation="h",
    #             yanchor="bottom",
    #             y=1.02,
    #             xanchor="right",
    #             x=1
    #         ),
    #         margin=dict(l=5, r=5, t=5, b=5),
    #         legend_title_text=None,
    #         xaxis_title=None,
    #         yaxis_title=None,
    #         xaxis=dict(range=range),
    #         yaxis=dict(range=range)
    #         )

    #     image_from_plot = fig.to_image(format="png", engine="kaleido")


    #     pil_image = Image.open(io.BytesIO(image_from_plot))
    #     # Convert the PIL image to a QImage
    #     image_qt = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    #     # Convert the QImage to a QPixmap 
    #     qpixmap = QPixmap.fromImage(image_qt)
    #     self.label_minimap.setPixmap(qpixmap)


    def update_ui_after_movement(self):
        # disable taking images after movement here
        if self.checkBox_movement_acquire_electron.isChecked():
            self.image_widget.take_image(BeamType.ELECTRON)
        if self.checkBox_movement_acquire_ion.isChecked():
            self.image_widget.take_image(BeamType.ION)
        self.update_ui()
    
    def _stage_position_moved(self, pos: FibsemStagePosition):
        self.update_ui_after_movement()




    def move_flat_to_beam(self):

        beam_type = BeamType.ION if self.sender() == self.pushButton_move_flat_ion else BeamType.ELECTRON

        self.microscope.move_flat_to_beam(settings=self.settings, beam_type=beam_type)

        self.update_ui_after_movement()



def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMovementWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
