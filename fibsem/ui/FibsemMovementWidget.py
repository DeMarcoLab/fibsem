import io
import logging
import os
import time
import traceback
from copy import deepcopy
from pathlib import Path

import napari
import napari.utils.notifications
from napari.qt.threading import thread_worker
import numpy as np
import yaml
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from fibsem import config as cfg
from fibsem import constants, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (BeamType, FibsemStagePosition,
                               MicroscopeSettings, MovementMode, Point, FibsemImage)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMovementWidget
from fibsem.ui.utils import _get_file_ui, _get_save_file_ui
from fibsem.imaging._tile import _plot_positions, _minimap 
from fibsem.ui import utils as ui_utils 
from fibsem.ui.utils import message_box_ui
from matplotlib.backends.backend_agg import FigureCanvasAgg
from fibsem.ui import _stylesheets
import fibsem.utils as utils

def log_status_message(step: str):
    logging.debug(
        f"STATUS | Movement Widget | {step}"
    )


class FibsemMovementWidget(FibsemMovementWidget.Ui_Form, QtWidgets.QWidget):
    move_signal = QtCore.pyqtSignal()
    movement_notification_signal = QtCore.pyqtSignal(str)

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
        self.parent = parent
        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget

        self.setup_connections()
        self.image_widget.picture_signal.connect(self.update_ui)
        self.positions = []
        self.minimap_image = None
        settings_dict = utils.load_yaml(cfg.SYSTEM_PATH)
        if bool(settings_dict["load_positions_on_startup"]):
            self.import_positions(cfg.POSITION_PATH)
        self.update_ui()

    def setup_connections(self):

        # set ui elements
        self.comboBox_movement_mode.addItems([mode.name for mode in MovementMode])
        self.comboBox_movement_stage_coordinate_system.addItems(["SPECIMEN", "RAW"])

        # buttons
        self.pushButton_move.clicked.connect(self.move_to_position)
        self.pushButton_move.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_continue.clicked.connect(self.continue_pressed)
        self.pushButton_move_flat_ion.clicked.connect(self.move_flat_to_beam)
        self.pushButton_move_flat_ion.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
        self.pushButton_move_flat_electron.clicked.connect(self.move_flat_to_beam)
        self.pushButton_load_image_minimap.clicked.connect(self.load_image)
        self.pushButton_move_flat_electron.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)

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
        self.pushButton_save_position.clicked.connect(lambda: self.add_position(position=None))
        self.pushButton_save_position.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove_position.clicked.connect(self.delete_position)
        self.pushButton_remove_position.setStyleSheet(_stylesheets._RED_PUSHBUTTON_STYLE)
        self.pushButton_go_to.clicked.connect(lambda: self.go_to_saved_position(None))
        self.pushButton_go_to.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
        self.pushButton_export.clicked.connect(self.export_positions)
        self.pushButton_export.setStyleSheet(_stylesheets._GRAY_PUSHBUTTON_STYLE)
        self.pushButton_import.clicked.connect(self.import_positions)
        self.pushButton_import.setStyleSheet(_stylesheets._GRAY_PUSHBUTTON_STYLE)
        self.pushButton_update_position.clicked.connect(self.update_saved_position)
        self.pushButton_update_position.setStyleSheet(_stylesheets._ORANGE_PUSHBUTTON_STYLE)

        self.movement_notification_signal.connect(self.update_moving_ui)

    def auto_eucentric_correction(self):

        print("auto eucentric")

    def continue_pressed(self):
        print("continue pressed")

    def _toggle_interactions(self, enable: bool, caller: str = None):
        
        self.pushButton_move.setEnabled(enable)
        self.pushButton_move_flat_ion.setEnabled(enable)
        self.pushButton_move_flat_electron.setEnabled(enable)
        self.pushButton_go_to.setEnabled(enable)
        if caller is None:
            self.parent.milling_widget._toggle_interactions(enable, caller="movement")
            self.parent.image_widget._toggle_interactions(enable, caller="movement")
        if enable:
            self.pushButton_move.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_ion.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_electron.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
            self.pushButton_go_to.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
        else:
            self.pushButton_move.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_ion.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_electron.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_go_to.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)

    def move_to_position(self):
        worker = self.move_worker()
        worker.finished.connect(self.run_moving_finished)
        # worker.yielded.connect(self.update_moving_ui)
        worker.start()
    
    @thread_worker
    def move_worker(self):
        self._toggle_interactions(False)
        stage_position = self.get_position_from_ui()
        self.movement_notification_signal.emit(f"Moving to {stage_position}")
        self.microscope.move_stage_absolute(stage_position)
        log_status_message(f"MOVED_TO_{stage_position}")
        self.movement_notification_signal.emit("Move finished, taking new images")
        self.update_ui_after_movement()

    def run_moving_finished(self):
        self._toggle_interactions(True)

    def update_moving_ui(self, msg: str):
        logging.info(msg)
        napari.utils.notifications.notification_manager.records.clear()
        napari.utils.notifications.show_info(msg)

    def update_ui(self):

        stage_position: FibsemStagePosition = self.microscope.get_stage_position()

        self.doubleSpinBox_movement_stage_x.setValue(stage_position.x * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_y.setValue(stage_position.y * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_z.setValue(stage_position.z * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_rotation.setValue(np.rad2deg(stage_position.r))
        self.doubleSpinBox_movement_stage_tilt.setValue(np.rad2deg(stage_position.t))

        # NOTE (pc): temporary to reduce number of updates    
        if self.sender() is None:
            self.minimap()

    
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
        worker = self._double_click_worker(layer, event)
        worker.finished.connect(self.run_moving_finished)
        # worker.yielded.connect(self.update_moving_ui)
        worker.start()

    @thread_worker
    def _double_click_worker(self, layer, event):
        
        if event.button != 1 or "Shift" in event.modifiers:
            return
        self._toggle_interactions(False)
        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)
        self.movement_notification_signal.emit("Click to move in progress")
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
        logging.debug(
            f"Movement: {self.movement_mode.name} | COORD {coords} | {point.__to_dict__()} | {beam_type}"
        )
        log_status_message(f"MOVING_{self.movement_mode.name}_BY_{point.x:.2e}, {point.y:.2e} | {beam_type}")
        self.movement_notification_signal.emit("Moving stage ")
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
        self.movement_notification_signal.emit("Move finished, taking new images")
        self.update_ui_after_movement()

    def select_position(self):
        if self.comboBox_positions.currentIndex() != -1:
            position = self.positions[self.comboBox_positions.currentIndex()]
            self.label_current_position.setText(f"x={position.x*constants.METRE_TO_MILLIMETRE:.3f}, y={position.y*constants.METRE_TO_MILLIMETRE:.3f}, z={position.z*constants.METRE_TO_MILLIMETRE:.3f}, r={position.r*constants.RADIANS_TO_DEGREES:.1f}, t={position.t*constants.RADIANS_TO_DEGREES:.1f}")

    def add_position(self, position: FibsemStagePosition = None):

        if not isinstance(position, FibsemStagePosition):
            position = self.microscope.get_stage_position()
            name = self.lineEdit_position_name.text()
            if name == "":
                napari.utils.notifications.show_warning("Please enter a name for the position")
                return
            position.name = name
        self.positions.append(deepcopy(position))
        self.comboBox_positions.addItem(position.name)
        self.comboBox_positions.setCurrentIndex(self.comboBox_positions.count() - 1)
        self.lineEdit_position_name.setText("")
        logging.info(f"Added position {position.name}")
        self.minimap()

    def delete_position(self):
        del self.positions[self.comboBox_positions.currentIndex()]
        name = self.comboBox_positions.currentText()
        self.comboBox_positions.removeItem(self.comboBox_positions.currentIndex())
        logging.info(f"Removed position {name}")
        self.minimap()

    def update_saved_position(self):
        position = self.microscope.get_stage_position()
        position.name = self.comboBox_positions.currentText()
        self.positions[self.comboBox_positions.currentIndex()] = position
        self.select_position()
        logging.info(f"Updated position {self.comboBox_positions.currentText()}")
        self.minimap()

    def go_to_saved_position(self, pos:FibsemStagePosition = None):
        worker = self.go_to_saved_position_worker(pos)
        worker.finished.connect(self.run_moving_finished)
        # worker.yielded.connect(self.update_moving_ui)
        worker.start()
    
    @thread_worker
    def go_to_saved_position_worker(self, pos: FibsemStagePosition = None):

        if pos is None:
            pos = self.positions[self.comboBox_positions.currentIndex()]
        self._toggle_interactions(False)
        self.movement_notification_signal.emit(f"Moving to saved position {pos}")
        self.microscope._safe_absolute_stage_movement(pos)
        logging.info(f"Moved to position {pos}")
        self.update_ui_after_movement()


    def export_positions(self):

        protocol_path = _get_save_file_ui(msg="Select or create file")
        if protocol_path == '':
            return
        response = message_box_ui(text="Do you want to overwrite the file ? Click no to append the new positions to the existing file.", title="Overwrite ?", buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        
        dict_position = []
        if not response:
            with open(protocol_path, 'r') as yaml_file:
                dict_position = yaml.safe_load(yaml_file)

        for position in self.positions:
            dict_position.append(position.__to_dict__())
        with open(os.path.join(Path(protocol_path).with_suffix(".yaml")), "w") as f:
            yaml.safe_dump(dict_position, f, indent=4, default_flow_style=False)

        logging.info("Positions saved to file")


    def import_positions(self, path: str = None):
        if not isinstance(path, str):
            protocol_path = _get_file_ui(msg="Select or create file")
        else: 
            protocol_path = path
        if protocol_path == '':
            napari.utils.notifications.show_info("No file selected, positions not loaded")
            return
        with open(protocol_path, "r") as f:
            dict_positions = yaml.safe_load(f)
        for dict_position in dict_positions:
            position = FibsemStagePosition.__from_dict__(dict_position)
            self.positions.append(position)
            self.comboBox_positions.addItem(position.name)
        self.minimap()

    def load_image(self):

        path = ui_utils._get_file_ui( msg="Select image to load", path=cfg.DATA_TILE_PATH, _filter="Image Files (*.tif *.tiff)", parent=self)

        if path == "":
            napari.utils.notifications.show_info(f"No file selected..")
            return

        
        image = FibsemImage.load(path)
        if image.metadata is None:
            napari.utils.notifications.show_error(f"Could not load image {path}. Make sure it is an OpenFibsem Image.")
            return

        self.minimap_image = image
        self.minimap()

    def minimap(self):

        if self.minimap_image is None:
            return
        
        current_position = self.microscope.get_stage_position()
        current_position.name = "Current Position"
        positions = deepcopy(self.positions)
        positions.insert(0, current_position)
        
        qpixmap = _minimap(self.minimap_image, positions)

        self.label_minimap.setPixmap(qpixmap)


    def update_ui_after_movement(self): # TODO: PPP Refactor
        # disable taking images after movement here
        if self.checkBox_movement_acquire_electron.isChecked() and self.checkBox_movement_acquire_ion.isChecked():
            self.image_widget.take_reference_images()
            while self.image_widget.TAKING_IMAGES:
                # logging.info(f"TAKING_IMAGES: {self.image_widget.TAKING_IMAGES}")
                time.sleep(0.2)
            self.update_ui()
            return
        if self.checkBox_movement_acquire_electron.isChecked():
            self.image_widget.take_image(BeamType.ELECTRON)
        while self.image_widget.TAKING_IMAGES:
                time.sleep(0.2)
        if self.checkBox_movement_acquire_ion.isChecked():
            self.image_widget.take_image(BeamType.ION)
        while self.image_widget.TAKING_IMAGES:
                time.sleep(0.2)    
        self.update_ui()
    
    def _stage_position_moved(self, pos: FibsemStagePosition):
        # self.update_ui_after_movement()
        self.update_ui() # TODO: fix taking images after movement

    def move_flat_to_beam(self):
        beam_type = BeamType.ION if self.sender() == self.pushButton_move_flat_ion else BeamType.ELECTRON
        worker = self.move_flat_to_beam_worker(beam_type)
        worker.finished.connect(self.run_moving_finished)
        # worker.yielded.connect(self.update_moving_ui)
        worker.start()

    @thread_worker
    def move_flat_to_beam_worker(self, beam_type):
        self._toggle_interactions(False)
        self.movement_notification_signal.emit(f"Moving flat to {beam_type.name} beam")
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
