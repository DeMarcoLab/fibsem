
import logging
from copy import deepcopy
from typing import Dict, List, Optional

import napari
import napari.utils.notifications
import numpy as np
import yaml
from napari.qt.threading import thread_worker
from PyQt5 import QtCore, QtWidgets

import fibsem.utils as utils
from fibsem import config as cfg
from fibsem import constants, conversions
from fibsem.constants import DEGREE_SYMBOL
from fibsem.microscope import FibsemMicroscope, ThermoMicroscope
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    Point,
)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMovementWidget as FibsemMovementWidgetUI
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    DISABLED_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    ORANGE_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)
from fibsem.ui.utils import (
    message_box_ui,
    open_existing_file_dialog,
    open_save_file_dialog,
)


def to_pretty_string(position: FibsemStagePosition) -> str:
    xstr = f"x={position.x*constants.METRE_TO_MILLIMETRE:.3f}"
    ystr = f"y={position.y*constants.METRE_TO_MILLIMETRE:.3f}"
    zstr = f"z={position.z*constants.METRE_TO_MILLIMETRE:.3f}"
    rstr = f"r={position.r*constants.RADIANS_TO_DEGREES:.1f}"
    tstr = f"t={position.t*constants.RADIANS_TO_DEGREES:.1f}"
    return f"{xstr}, {ystr}, {zstr}, {rstr}, {tstr}"

def to_pretty_string_short(position: FibsemStagePosition) -> str:
    xstr = f"X:{position.x*constants.METRE_TO_MILLIMETRE:.2f}"
    ystr = f"Y:{position.y*constants.METRE_TO_MILLIMETRE:.2f}"
    zstr = f"Z:{position.z*constants.METRE_TO_MILLIMETRE:.2f}"
    rstr = f"R:{position.r*constants.RADIANS_TO_DEGREES:.1f}"
    tstr = f"T:{position.t*constants.RADIANS_TO_DEGREES:.1f}"
    return f"{xstr}, {ystr}, {zstr}, {rstr}, {tstr}"

class FibsemMovementWidget(FibsemMovementWidgetUI.Ui_Form, QtWidgets.QWidget):
    saved_positions_updated_signal = QtCore.pyqtSignal(object)  # TODO: investigate the use of this signal
    movement_progress_signal = QtCore.pyqtSignal(dict) # TODO: consolidate

    def __init__(
        self,
        microscope: FibsemMicroscope,
        viewer: napari.Viewer,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent
        
        self.microscope = microscope
        self.viewer = viewer
        self.image_widget: FibsemImageSettingsWidget = parent.image_widget

        self.positions: List[FibsemStagePosition] = []
        
        autoload_positions: bool = True # TODO: enable this
        if autoload_positions:
            self.import_positions(cfg.POSITION_PATH)

        self.setup_connections()

    def setup_connections(self):

        # buttons
        self.pushButton_move.clicked.connect(lambda: self.move_to_position(None))
        self.pushButton_move_flat_ion.clicked.connect(self.move_flat_to_beam)
        self.pushButton_move_flat_electron.clicked.connect(self.move_flat_to_beam)
        self.pushButton_refresh_stage_position_data.clicked.connect(self.update_ui)

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_double_click_callbacks.append(self._double_click)
        self.image_widget.ib_layer.mouse_double_click_callbacks.append(self._double_click)

        # disable ui elements
        self.label_movement_instructions.setText("Double click to move. Alt + Double Click in the Ion Beam to Move Vertically")

        # positions
        self.comboBox_positions.currentIndexChanged.connect(self.current_selected_position_changed)
        self.pushButton_save_position.clicked.connect(self.add_position)
        self.pushButton_remove_position.clicked.connect(self.delete_position)
        self.pushButton_go_to.clicked.connect(self.move_to_selected_saved_position)
        self.pushButton_update_position.clicked.connect(self.update_saved_position)
        self.pushButton_export.clicked.connect(self.export_positions)
        self.pushButton_import.clicked.connect(self.import_positions)

        # signals
        self.movement_progress_signal.connect(self.handle_movement_progress_update)
        self.image_widget.acquisition_progress_signal.connect(self.handle_acquisition_update)
        self.saved_positions_updated_signal.connect(self.update_saved_positions_ui)

        # set custom tilt limits for the compustage
        if self.microscope is not None and isinstance(self.microscope, (ThermoMicroscope, DemoMicroscope)):
            if self.microscope.stage_is_compustage:
                self.doubleSpinBox_movement_stage_tilt.setMinimum(-195.0)
                self.doubleSpinBox_movement_stage_tilt.setMaximum(15)

                # NOTE: these values are expressed in mm in the UI, hence the conversion
                # set x, y, z step sizes to be 1 um
                self.doubleSpinBox_movement_stage_x.setSingleStep(1e-6 * constants.SI_TO_MILLI)
                self.doubleSpinBox_movement_stage_y.setSingleStep(1e-6 * constants.SI_TO_MILLI)
                self.doubleSpinBox_movement_stage_z.setSingleStep(1e-6 * constants.SI_TO_MILLI)

                # TODO: get the true limits from the microscope
                self.doubleSpinBox_movement_stage_x.setMinimum(-999.9e-6 * constants.SI_TO_MILLI)
                self.doubleSpinBox_movement_stage_x.setMaximum(999.9e-6 * constants.SI_TO_MILLI)
                self.doubleSpinBox_movement_stage_y.setMinimum(-377.8e-6 * constants.SI_TO_MILLI)
                self.doubleSpinBox_movement_stage_y.setMaximum(377.8e-6 * constants.SI_TO_MILLI)

        # stylesheets
        self.pushButton_move.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.pushButton_move_flat_ion.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_move_flat_electron.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_move_to_milling_angle.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_refresh_stage_position_data.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.pushButton_save_position.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove_position.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.pushButton_go_to.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_export.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.pushButton_import.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.pushButton_update_position.setStyleSheet(ORANGE_PUSHBUTTON_STYLE)
        
        # display orientation values on tooltips
        self.pushButton_move_flat_electron.setText("Move to SEM Orientation")
        self.pushButton_move_flat_ion.setText("Move to FIB Orientation")
        sem = self.microscope.get_orientation("SEM")
        fib = self.microscope.get_orientation("FIB")
        milling = self.microscope.get_orientation("MILLING")
        sem_str = f"R:{sem.r*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}, T:{sem.t*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}"
        fib_str = f"R:{fib.r*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}, T:{fib.t*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}"
        milling_str = f"R:{milling.r*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}, T:{milling.t*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}"
        self.pushButton_move_flat_electron.setToolTip(sem_str)
        self.pushButton_move_flat_ion.setToolTip(fib_str)
        self.pushButton_move_to_milling_angle.setToolTip(milling_str)

        # milling angle controls
        self.doubleSpinBox_milling_angle.setValue(self.microscope.system.stage.milling_angle) # deg
        self.doubleSpinBox_milling_angle.setSuffix(DEGREE_SYMBOL)
        self.doubleSpinBox_milling_angle.setSingleStep(1.0)
        self.doubleSpinBox_milling_angle.setDecimals(1)
        self.doubleSpinBox_milling_angle.setRange(0, 45)
        self.doubleSpinBox_milling_angle.setToolTip("The milling angle is the difference between the stage and the fib viewing angle.")
        self.doubleSpinBox_milling_angle.setKeyboardTracking(False)
        self.doubleSpinBox_milling_angle.valueChanged.connect(self._update_milling_angle)
        self.pushButton_move_to_milling_angle.clicked.connect(lambda: self.move_to_orientation("MILLING"))

        self.display_stage_position_overlay()

        # update the UI
        self.update_ui()

    def _toggle_interactions(self, enable: bool, caller: str = None):
        """Toggle the interactions in the widget depending on microscope state"""
        self.pushButton_move.setEnabled(enable)
        self.pushButton_move_flat_ion.setEnabled(enable)
        self.pushButton_move_flat_electron.setEnabled(enable)
        self.pushButton_move_to_milling_angle.setEnabled(enable)
        self.doubleSpinBox_milling_angle.setEnabled(enable)
        self.pushButton_go_to.setEnabled(enable)
        if caller is None:
            self.parent.milling_widget._toggle_interactions(enable, caller="movement")
            self.parent.image_widget._toggle_interactions(enable, caller="movement")
        if enable:
            self.pushButton_move.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_ion.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_electron.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
            self.pushButton_move_to_milling_angle.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
            self.pushButton_go_to.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        else:
            self.pushButton_move.setStyleSheet(DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_ion.setStyleSheet(DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_move_flat_electron.setStyleSheet(DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_move_to_milling_angle.setStyleSheet(DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_go_to.setStyleSheet(DISABLED_PUSHBUTTON_STYLE)

    def handle_movement_progress_update(self, ddict: dict) -> None:
        """Handle movement progress updates from the microscope"""
        
        msg = ddict.get("msg", None)
        if msg is not None:
            logging.debug(msg)
            napari.utils.notifications.notification_manager.records.clear()
            napari.utils.notifications.show_info(msg)

        is_finished = ddict.get("finished", False)
        if is_finished:
            logging.info("Movement finished")
            # TODO: handle the signals
            self.display_stage_position_overlay()

    def display_stage_position_overlay(self):
        """Display the stage position as text overlay on the image widget"""
        if isinstance(self.microscope, TescanMicroscope):
            return  # Tescan systems do not support stage position display yet
        
        try:
            # NOTE: this crashes for tescan systems?
            pos = self.microscope.get_stage_position()
            orientation = self.microscope.get_stage_orientation()
            milling_angle = self.microscope.get_current_milling_angle()
        except Exception as e:
            logging.warning(f"Error getting stage position: {e}")
            return

        # add text layer, showing the stage position in cyan
        points = np.array([
            [self.image_widget.eb_layer.data.shape[0] + 50, 500],
            [self.image_widget.eb_layer.data.shape[0] + 50, self.image_widget.ib_layer.translate[1] + 150]
            ])
        text = {
            "string": [
                f"STAGE: {to_pretty_string_short(pos)} [{orientation}]",
                f"MILLING ANGLE: {milling_angle:.1f} {DEGREE_SYMBOL}"
                ],
            "color": "cyan",
            "font_size": 20,
        }
        try:
            self.viewer.layers['stage_position'].data = points
            self.viewer.layers['stage_position'].text = text
        except KeyError:    
            self.viewer.add_points(
                data=points,
                name="stage_position",
                size=20,
                text=text,
                edge_width=7,
                edge_width_is_relative=False,
                edge_color="transparent",
                face_color="transparent",
            )   

    def handle_acquisition_update(self, ddict: dict):
        """Handle acquisition updates from the image widget"""
        is_finished = ddict.get("finished", False)
        if is_finished:
            self.update_ui()

    def update_ui(self):
        """Update the UI with the current stage position and saved positions"""
        stage_position: FibsemStagePosition = self.microscope.get_stage_position()

        self.doubleSpinBox_movement_stage_x.setValue(stage_position.x * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_y.setValue(stage_position.y * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_z.setValue(stage_position.z * constants.SI_TO_MILLI)
        self.doubleSpinBox_movement_stage_rotation.setValue(np.degrees(stage_position.r))
        self.doubleSpinBox_movement_stage_tilt.setValue(np.degrees(stage_position.t))

        HAS_SAVED_POSITIONS = bool(len(self.positions))
        # self.pushButton_save_position.setVisible(HAS_SAVED_POSITIONS) # always visible
        self.pushButton_remove_position.setEnabled(HAS_SAVED_POSITIONS)
        self.pushButton_go_to.setVisible(HAS_SAVED_POSITIONS)
        self.pushButton_update_position.setVisible(HAS_SAVED_POSITIONS)
        self.pushButton_export.setVisible(HAS_SAVED_POSITIONS)
        self.comboBox_positions.setVisible(HAS_SAVED_POSITIONS)
        self.pushButton_import.setVisible(HAS_SAVED_POSITIONS)
        self.label_saved_positions.setVisible(HAS_SAVED_POSITIONS)
        self.label_current_position.setVisible(HAS_SAVED_POSITIONS)
        self.lineEdit_position_name.setVisible(HAS_SAVED_POSITIONS)

        # update the current position label
        self.display_stage_position_overlay()

    def update_ui_after_movement(self, retake: bool = True): # TODO: PPP Refactor
        # disable taking images after movement here
        if retake is False:
            self.update_ui()
            return
        if self.checkBox_movement_acquire_electron.isChecked() and self.checkBox_movement_acquire_ion.isChecked():
            self.image_widget.acquire_reference_images()
            return
        if self.checkBox_movement_acquire_electron.isChecked():
            # self.image_widget.acquire_image(BeamType.ELECTRON)
            logging.warning("Acquiring electron image after movement has been disabled temporarily. Please only acquire both images after movement")
        elif self.checkBox_movement_acquire_ion.isChecked():
            # self.image_widget.acquire_image(BeamType.ION)
            logging.warning("Acquiring ion image after movement has been disabled temporarily. Please only acquire both images after movement")
        else: 
            self.update_ui()

    def _update_milling_angle(self):
        """Update the milling angle in the microscope and the UI"""
        milling_angle = self.doubleSpinBox_milling_angle.value() # deg
        self.microscope.system.stage.milling_angle = milling_angle

        # set tooltip (dynamically)
        milling = self.microscope.get_orientation("MILLING")
        milling_str = f"R:{milling.r*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}, T:{milling.t*constants.RADIANS_TO_DEGREES:.1f} {DEGREE_SYMBOL}"
        self.pushButton_move_to_milling_angle.setToolTip(milling_str)

        # refresh stage display
        self.display_stage_position_overlay()

#### MOVEMENT

    def move_to_position(self, stage_position: Optional[FibsemStagePosition] = None):
        """Move the stage to the position specified in the UI"""
        if stage_position is None:
            stage_position = self.get_position_from_ui()
        self._move_to_absolute_position(stage_position)

    def _move_to_absolute_position(self, stage_position: FibsemStagePosition):
        """Move the stage to the specified position"""
        self._toggle_interactions(enable=False)
        worker = self.absolute_movement_worker(stage_position=stage_position)
        worker.finished.connect(self.move_stage_finished)
        worker.start()
    
    @thread_worker
    def absolute_movement_worker(self, stage_position: FibsemStagePosition) -> None:
        """Worker function to move the stage to the specified position"""
        self.movement_progress_signal.emit({"msg": f"Moving to {stage_position}"})
        self.microscope.safe_absolute_stage_movement(stage_position)
        self.movement_progress_signal.emit({"msg": "Move finished, taking new images"})
        self.update_ui_after_movement()

    def move_stage_finished(self):
        """Handle the completion of a stage movement"""
        self.movement_progress_signal.emit({"finished": True})
        if self.image_widget.is_acquiring:
            return
        self._toggle_interactions(enable=True)

    def get_position_from_ui(self):
        """Get the stage position from the UI"""

        stage_position = FibsemStagePosition(
            x=self.doubleSpinBox_movement_stage_x.value() * constants.MILLI_TO_SI,
            y=self.doubleSpinBox_movement_stage_y.value() * constants.MILLI_TO_SI,
            z=self.doubleSpinBox_movement_stage_z.value() * constants.MILLI_TO_SI,
            r=np.radians(self.doubleSpinBox_movement_stage_rotation.value()),
            t=np.radians(self.doubleSpinBox_movement_stage_tilt.value()),
            coordinate_system="RAW",
        )

        return stage_position

    def _double_click(self, layer, event):
        """Callback for double-click mouse events on the image widget"""
        self._toggle_interactions(enable= False)

        worker = self._double_click_worker(layer, event)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def _double_click_worker(self, layer, event):
        """Thread worker for double-click mouse events on the image widget"""
        if event.button != 1 or "Shift" in event.modifiers:
            return
        
        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)
        self.movement_progress_signal.emit({"msg": "Click to move in progress..."})

        if beam_type is None:
            napari.utils.notifications.show_info(
                "Clicked outside image dimensions. Please click inside the image to move."
            )
            return

        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), 
            image=image.data, 
            pixelsize=image.metadata.pixel_size.x,
        )

        # move
        vertical_move = True if "Alt" in event.modifiers else False
        movement_mode = "Vertical" if vertical_move else "Stable"

        logging.debug({
            "msg": "stage_movement",                    # message type
            "movement_mode": movement_mode,             # movement mode
            "beam_type": beam_type.name,                # beam type
            "dm": point.to_dict(),                      # shift in microscope coordinates
            "coords": {"x": coords[1], "y": coords[0]}, # coords in image coordinates
        })
        
        self.movement_progress_signal.emit({"msg": "Moving stage..."})
        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and vertical_move:
            self.microscope.vertical_move(dx=point.x, dy=point.y
            )

        else:
            # corrected stage movement
            self.microscope.stable_move(
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )
        self.movement_progress_signal.emit({"msg": "Move finished, updating UI"})
        self.update_ui_after_movement()

    def move_flat_to_beam(self)-> None:
        """Move to the specifed beam orientation"""
        beam_type = BeamType.ION if self.sender() == self.pushButton_move_flat_ion else BeamType.ELECTRON
        self._toggle_interactions(False)
        worker = self.move_flat_to_beam_worker(beam_type)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def move_flat_to_beam_worker(self, beam_type: BeamType) -> None:
        """Threaded worker function to move the stage to the specified beam"""
        self.movement_progress_signal.emit({"msg": f"Moving flat to {beam_type.name} beam"})
        self.microscope.move_flat_to_beam(beam_type=beam_type)
        self.update_ui_after_movement()

    # TODO: migrate to this from move_flat_to_beam
    def move_to_orientation(self, orientation: str)-> None:
        """Move to the specifed orientation"""
        if orientation not in ["SEM", "FIB", "MILLING"]:
            raise ValueError(f"Invalid orientation: {orientation}")
        self._toggle_interactions(False)
        worker = self.move_to_orientation_worker(orientation)
        worker.finished.connect(self.move_stage_finished)
        worker.start()

    @thread_worker
    def move_to_orientation_worker(self, orientation: str) -> None:
        """Threaded worker function to move the stage to the specified orientation"""
        self.movement_progress_signal.emit({"msg": f"Moving flat to {orientation}"})
        stage_orientation = self.microscope.get_orientation(orientation)
        self.microscope.safe_absolute_stage_movement(stage_orientation)
        self.update_ui_after_movement()

#### SAVED POSITIONS

    def current_selected_position_changed(self):
        """Callback for when the selected position is changed"""
        current_index = self.comboBox_positions.currentIndex()
        if current_index == -1:
            return
        position = self.positions[current_index]
        self.label_current_position.setText(to_pretty_string(position))
        self.lineEdit_position_name.setText(position.name)

    def add_position(self) -> None:
        """Add the current stage position to the saved positions"""

        position = self.microscope.get_stage_position()
        name = f"Position {len(self.positions):02d}"
        position.name = name
        self.positions.append(deepcopy(position))
        self.saved_positions_updated_signal.emit(self.positions)
        logging.info(f"Added position {position.name}")

    def delete_position(self):
        """Delete the currently selected position"""
        current_index = self.comboBox_positions.currentIndex()
        position = self.positions.pop(current_index)
        self.saved_positions_updated_signal.emit(self.positions)
        logging.info(f"Removed position {position.name}")

    def update_saved_positions_ui(self, positions: List[FibsemStagePosition]) -> None:
        """Update the positions in the widget"""

        self.comboBox_positions.clear()
        for position in positions:
            self.comboBox_positions.addItem(position.name)

        # set index to the last position
        self.comboBox_positions.setCurrentIndex(len(positions) - 1)
        self.lineEdit_position_name.setText(positions[-1].name)
        self.update_ui()

    def update_saved_position(self):
        """Update the currently selected saved position to the current stage position"""
        current_index = self.comboBox_positions.currentIndex()
        if current_index == -1:
            napari.utils.notifications.show_warning("Please select a position to update")
            return

        position: FibsemStagePosition = self.microscope.get_stage_position()
        position.name = self.lineEdit_position_name.text()
        if position.name == "":
            napari.utils.notifications.show_warning("Please enter a name for the position")
            return

        # TODO: add dialog confirmation
        self.positions[current_index] = deepcopy(position)

        logging.info(f"Updated position {position}")
        self.saved_positions_updated_signal.emit(self.positions) # redraw combobox, TODO: need to re-select the current index

    def move_to_selected_saved_position(self) -> None: 
        """Move the stage to the selected saved position"""
        current_index = self.comboBox_positions.currentIndex()
        stage_position = self.positions[current_index]
        self._move_to_absolute_position(stage_position)

    def export_positions(self):

        path = open_save_file_dialog(msg="Select or create file", 
            path=cfg.POSITION_PATH, 
            _filter="YAML Files (*.yaml)")
        
        if path == '':
            napari.utils.notifications.show_info("No file selected, positions not saved")
            return
        
        response = message_box_ui(text="Do you want to overwrite the file ? Click no to append the new positions to the existing file.", 
            title="Overwrite ?", buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        
        # save positions
        utils.save_positions(self.positions, path, overwrite=response)

        logging.info(f"Positions saved to {path}")

    def import_positions(self, path: str = None):
        """Import saved positions from a file"""
        
        if path is None:
            path = open_existing_file_dialog(msg="Select a positions file", path=cfg.POSITION_PATH, _filter="YAML Files (*.yaml)")
        
        if path == "":
            napari.utils.notifications.show_info("No file selected, positions not loaded")
            return

        def load_saved_positions_from_yaml(path: str = None) -> List[FibsemStagePosition]:
            with open(path, "r") as f:
                ddict = yaml.safe_load(f)
            return [FibsemStagePosition.from_dict(pdict) for pdict in ddict]

        self.positions = load_saved_positions_from_yaml(path)
        self.saved_positions_updated_signal.emit(self.positions)