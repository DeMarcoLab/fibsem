
import logging
import time
from copy import deepcopy
from typing import Dict, List, Union, Optional

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Layer
from napari.qt.threading import thread_worker
from PyQt5 import QtCore, QtWidgets

from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.microscope import (
    DemoMicroscope,
    FibsemMicroscope,
    TescanMicroscope,
    ThermoMicroscope,
)
from fibsem.milling import FibsemMillingStage, get_strategy, mill_stages
from fibsem.milling.patterning.patterns2 import (
    MILLING_PATTERN_NAMES,
    BasePattern,
    CirclePattern,
    FiducialPattern,
    LinePattern,
    RectanglePattern,
    get_pattern,
)
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemImage,
    FibsemMillingSettings,
    MicroscopeSettings,
    Point,
    calculate_fiducial_area_v2,
)
from fibsem.ui import _stylesheets as stylesheets
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMillingWidget
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    remove_all_napari_shapes_layers,
    convert_pattern_to_napari_circle,
    convert_pattern_to_napari_rect,
    validate_pattern_placement,
)

_UNSCALED_VALUES  = ["rotation", "size_ratio", "scan_direction", "cleaning_cross_section", 
                     "number", "passes", "n_rectangles", "overlap", "inverted", "use_side_patterns",
                     "n_columns", "n_rows", "cross_section", "time"]
_ANGLE_KEYS = ["rotation"]
_LINE_KEYS = ["start_x", "start_y", "end_x", "end_y"]

MILLING_WIDGET_INSTRUCTIONS = """Controls:
Shift + Left Click to Move Selected Pattern
Ctrl + Shift + Left Click to Move All Patterns
Press Run Milling to Start Milling"""

def scale_value_for_display(key: str, value: Union[float, int], scale: float) -> Union[float, int]:
    if key not in _UNSCALED_VALUES:
        return value * scale    
    return value

def log_status_message(stage: FibsemMillingStage, step: str):
    logging.debug(
        f"STATUS | Milling Widget | {stage.name} | {step}"
    )


# ned to re-write
# if no milling stage, dont show milling settings or pattern settings
# add groups for milling settings, pattern settings
# add drift correction block
# make everything into scollable


# advanced settings/hidden by default
# external events should stay the same



# milling operations
# run, pause/resume, stop


class FibsemMillingWidget(FibsemMillingWidget.Ui_Form, QtWidgets.QWidget):
    milling_position_changed = QtCore.pyqtSignal()
    _milling_finished = QtCore.pyqtSignal()
    milling_notification = QtCore.pyqtSignal(str)
    _progress_bar_update = QtCore.pyqtSignal(object)
    _progress_bar_start = QtCore.pyqtSignal(object)
    _progress_bar_quit = QtCore.pyqtSignal()
    # TODO: consolidate these into a single signal

    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None, # TODO: deprecate external settings
        viewer: napari.Viewer = None,
        image_widget: FibsemImageSettingsWidget = None,
        parent=None,
    ):
        super(FibsemMillingWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent
        self.microscope = microscope
        
        self.viewer = viewer
        self.image_widget = image_widget

        # default milling protocol
        self.protocol = utils.load_yaml(cfg.PROTOCOL_PATH)
        
        self.current_milling_stage: Optional[FibsemMillingStage] = None
        self.milling_stages: List[FibsemMillingStage] = []
        self.milling_pattern_layers: List[Layer] = []

        self.setup_connections()

        self.good_copy_pattern = None

        self.UPDATING_PATTERN:bool = False
        self.CAN_MOVE_PATTERN: bool = True
        self._STOP_MILLING: bool = False

    def setup_connections(self):
        """Setup the connections for the milling widget."""


        IS_THERMO = isinstance(self.microscope, ThermoMicroscope)
        IS_TESCAN = isinstance(self.microscope, TescanMicroscope)

        if isinstance(self.microscope, DemoMicroscope):
            IS_THERMO, IS_TESCAN = True, False
        
        # MILLING SETTINGS
        # general
        AVAILABLE_MILLING_MODES: List[str] = ["Serial", "Parallel"] # TODO: make microscope method
        self.AVAILABLE_SCAN_DIRECTIONS = self.microscope.get_available_values(key="scan_direction")
        self.comboBox_patterning_mode.addItems(AVAILABLE_MILLING_MODES)
        self.comboBox_patterning_mode.currentIndexChanged.connect(self.update_milling_settings_from_ui)
        self.doubleSpinBox_hfw.valueChanged.connect(self.update_milling_settings_from_ui)
        
        # ThermoFisher Only 
        self.label_application_file.setVisible(IS_THERMO)
        self.comboBox_application_file.setVisible(IS_THERMO)
        self.doubleSpinBox_milling_current.setVisible(IS_THERMO)
        self.label_milling_current.setVisible(IS_THERMO)
        self.label_voltage.setVisible(IS_THERMO)
        self.spinBox_voltage.setVisible(IS_THERMO) # TODO: set this to the available voltages
        if IS_THERMO:
            AVAILABLE_APPLICATION_FILES = self.microscope.get_available_values("application_file")
            self.comboBox_application_file.addItems(AVAILABLE_APPLICATION_FILES)
            # milling currents: TODO: make this a combobox with available values
            self.AVAILABLE_MILLING_CURRENTS = self.microscope.get_available_values("current", BeamType.ION)
            min_current = self.AVAILABLE_MILLING_CURRENTS[0] * constants.SI_TO_NANO
            max_current = self.AVAILABLE_MILLING_CURRENTS[-1] * constants.SI_TO_NANO
            self.doubleSpinBox_milling_current.setRange(min_current, max_current)
            self.doubleSpinBox_milling_current.setDecimals(4)
            self.comboBox_application_file.currentIndexChanged.connect(self.update_milling_settings_from_ui)
            self.doubleSpinBox_milling_current.valueChanged.connect(self.update_milling_settings_from_ui)
            self.spinBox_voltage.valueChanged.connect(self.update_milling_settings_from_ui)
        
        # Tescan Only
        AVAILABLE_PRESETS = self.microscope.get_available_values("presets")
        self.label_rate.setVisible(IS_TESCAN)
        self.label_spot_size.setVisible(IS_TESCAN)
        self.label_dwell_time.setVisible(IS_TESCAN)
        self.doubleSpinBox_rate.setVisible(IS_TESCAN)
        self.doubleSpinBox_spot_size.setVisible(IS_TESCAN)
        self.doubleSpinBox_dwell_time.setVisible(IS_TESCAN)   
        self.comboBox_preset.setVisible(IS_TESCAN)
        self.label_preset.setVisible(IS_TESCAN)
        self.label_spacing.setVisible(IS_TESCAN)
        self.doubleSpinBox_spacing.setVisible(IS_TESCAN)
        if IS_TESCAN:   
            self.comboBox_preset.addItems(AVAILABLE_PRESETS)
            self.doubleSpinBox_rate.valueChanged.connect(self.update_milling_settings_from_ui)
            self.doubleSpinBox_spot_size.valueChanged.connect(self.update_milling_settings_from_ui)
            self.doubleSpinBox_dwell_time.valueChanged.connect(self.update_milling_settings_from_ui)  
            self.comboBox_preset.currentIndexChanged.connect(self.update_milling_settings_from_ui)
            self.doubleSpinBox_spacing.valueChanged.connect(self.update_milling_settings_from_ui)

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_drag_callbacks.append(self._single_click)
        self.image_widget.ib_layer.mouse_drag_callbacks.append(self._single_click)

        # new patterns
        self.comboBox_patterns.addItems(MILLING_PATTERN_NAMES)
        self.comboBox_patterns.currentIndexChanged.connect(self.update_current_selected_pattern)
    
        # milling stages
        self.pushButton_add_milling_stage.clicked.connect(self.add_milling_stage)
        self.pushButton_remove_milling_stage.clicked.connect(self.remove_milling_stage)
        self.comboBox_milling_stage.currentIndexChanged.connect(self.update_milling_stage_ui)
        
        # update ui
        self.milling_notification.connect(self.update_milling_ui)

        # run milling
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        
        # stop milling
        self.pushButton_stop_milling.clicked.connect(self.stop_milling)
        self.pushButton_stop_milling.setVisible(False)
        self.pushButton_pause_milling.setVisible(False) # TODO: implement pause / resume

        # set styles
        self.pushButton_add_milling_stage.setStyleSheet(stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove_milling_stage.setStyleSheet(stylesheets._RED_PUSHBUTTON_STYLE)
        self.pushButton_run_milling.setStyleSheet(stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_stop_milling.setStyleSheet(stylesheets._RED_PUSHBUTTON_STYLE)

        # progress bar # TODO: fix this (redo)
        self.progressBar_milling.setVisible(False)
        self._progress_bar_update.connect(self.update_progress_bar)
        self._progress_bar_start.connect(self.start_progress_thread)
        self._progress_bar_quit.connect(self._quit_progress_bar)

        # last
        self.doubleSpinBox_centre_x.setKeyboardTracking(False)
        self.doubleSpinBox_centre_y.setKeyboardTracking(False)
        self.doubleSpinBox_centre_x.valueChanged.connect(self.redraw_patterns)
        self.doubleSpinBox_centre_y.valueChanged.connect(self.redraw_patterns)

        # options
        self.checkBox_show_milling_patterns.setChecked(True)
        self.checkBox_show_milling_crosshair.setChecked(True)
        self.checkBox_show_milling_crosshair.stateChanged.connect(self.redraw_patterns)
        self.checkBox_show_milling_patterns.stateChanged.connect(self.toggle_pattern_visibility)
            
        self.label_milling_instructions.setText(MILLING_WIDGET_INSTRUCTIONS)
        self.show_milling_stage_widgets()

        # external signals
        self.image_widget.viewer_update_signal.connect(self.update_ui) # update the ui when the image is updated

    def update_milling_settings_from_ui(self):
        settings = self.get_milling_settings_from_ui()
        index = self.comboBox_milling_stage.currentIndex()
        if index != -1:
            self.milling_stages[index].milling = settings

    def add_milling_stage(self):
        logging.info("Adding milling stage")

        num = len(self.milling_stages) + 1
        name = f"Milling Stage {num}"
        pattern = RectanglePattern.from_dict(self.protocol["patterns"]["Rectangle"])
        milling_stage = FibsemMillingStage(name=name, num=num, 
                                           pattern=pattern, 
                                           strategy=get_strategy())
        self.milling_stages.append(milling_stage)
        self.comboBox_milling_stage.addItem(name)
        self.comboBox_milling_stage.setCurrentText(name)
        napari.utils.notifications.show_info(f"Added {name}.")
        log_status_message(self.milling_stages[-1], "CREATED_STAGE")

    def remove_milling_stage(self):
        logging.info("Removing milling stage")

        current_index = self.comboBox_milling_stage.currentIndex()
        if current_index == -1:
            return
        log_status_message(self.milling_stages[current_index], "REMOVED_STAGE")
        self.milling_stages.pop(current_index)
        self.comboBox_milling_stage.removeItem(current_index)
        napari.utils.notifications.show_info("Removed milling stage.")     


    def clear_all_milling_stages(self):
        """Remove all milling stages from the widget."""
        self.milling_stages = []
        self.comboBox_milling_stage.currentIndexChanged.disconnect()
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.currentIndexChanged.connect(self.update_milling_stage_ui)

        remove_all_napari_shapes_layers(self.viewer) # remove all shape layers

    def set_milling_stages(self, milling_stages: List[FibsemMillingStage]) -> None:
        logging.debug(f"Setting milling stages: {len(milling_stages)}")
        self.milling_stages = milling_stages
        
        # very explicitly set what is happening
        self.comboBox_milling_stage.currentIndexChanged.disconnect()
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.addItems([stage.name for stage in self.milling_stages])
        self.comboBox_milling_stage.currentIndexChanged.connect(self.update_milling_stage_ui)
        
        self.comboBox_patterns.currentIndexChanged.disconnect()
        self.comboBox_patterns.setCurrentText(self.milling_stages[0].pattern.name)
        self.comboBox_patterns.currentIndexChanged.connect(self.update_current_selected_pattern)
        
        logging.debug(f"Set milling stages: {len(milling_stages)}")
        self.update_milling_stage_ui()
        # self.update_ui()

    def get_milling_stages(self):
        """Get the milling stages from the widget."""
        return self.milling_stages

    def get_point_from_ui(self):
        """Get the pattern position from the UI."""
        point = Point(x=self.doubleSpinBox_centre_x.value() * constants.MICRO_TO_SI, 
                      y=self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI)

        return point

    def show_milling_stage_widgets(self):
        """Show / Enable the milling stage widgets if a milling stage is selected."""
        show = self.current_milling_stage is not None
        self.scrollArea_milling_stage.setVisible(show)
        
        # disable milling buttons if no milling stage
        self.pushButton_run_milling.setEnabled(show)
        self.pushButton_remove_milling_stage.setEnabled(show)

    def update_milling_stage_ui(self):
        """Update the milling stage UI when the milling stage is changed."""

        # get the selected milling stage
        current_index = self.comboBox_milling_stage.currentIndex()
        if current_index == -1:
            remove_all_napari_shapes_layers(self.viewer)
            self.current_milling_stage = None
            self.show_milling_stage_widgets()
            return

        self.current_milling_stage: FibsemMillingStage = self.milling_stages[current_index]

        # update the milling and pattern settings
        self.set_milling_settings_ui()        
        self.setup_pattern_ui()
        # TODO: drift correction

        self.show_milling_stage_widgets()

    def redraw_patterns(self):
        """Redraw the patterns in the viewer."""
        if self.UPDATING_PATTERN:
            return

        # refresh the pattern
        self.update_ui()

    def update_milling_stage_from_ui(self):
                
        # get current milling stage
        current_index = self.comboBox_milling_stage.currentIndex()

        if current_index == -1:
            msg = "No milling stages defined, cannot draw patterns."
            logging.warning(msg)
            napari.utils.notifications.show_warning(msg)
            return

        milling_stage: FibsemMillingStage = self.milling_stages[current_index]

        # update milling settings
        milling_stage.milling = self.get_milling_settings_from_ui()

        # update pattern and define
        milling_stage.pattern = self.get_pattern_from_ui_v2()

        # TODO: drift correction
        # milling_stage.drift_correction = self.get_drift_correction_from_ui()

        napari.utils.notifications.show_info(f"Updated {milling_stage.name}.")
        return 


    def update_current_selected_pattern(self):
        """When the currently selected pattern is changed, update the current milling stage, and then reset the ui."""

        pattern_name = self.comboBox_patterns.currentText()

        logging.info(f"selected pattern: {pattern_name}")
        pattern = get_pattern(pattern_name, self.protocol["patterns"][pattern_name])
        self.current_milling_stage.pattern = pattern
        self.setup_pattern_ui()

    def setup_pattern_ui(self):
        """When the pattern is changed, setup the ui for the new pattern."""

        self.UPDATING_PATTERN = True

        # if the pattern is change? what happens
        # get the new currently selected pattern
        # set the new pattern to the current milling stage
        # update the ui 

        milling_stage = self.current_milling_stage # the currently selected milling stage
        if milling_stage is None:
            self.UPDATING_PATTERN = False
            return
        
        pattern: BasePattern = milling_stage.pattern
        point: Point = pattern.point

        # clear layout
        for i in reversed(range(self.gridLayout_patterns.count())):
            self.gridLayout_patterns.itemAt(i).widget().setParent(None)

        self.pattern_attribute_widgets: Dict[str, QtWidgets.QWidget] = {}

        # set widgets for each required key / value
        for i, key in enumerate(pattern.required_attributes):

            label = QtWidgets.QLabel(key.replace("_", " ").title())

            # default None
            val = getattr(pattern, key, None)

            if isinstance(val, (int, float)):
                # limits
                min_val = -1000 if key in _LINE_KEYS else 0

                control_widget = QtWidgets.QDoubleSpinBox()
                control_widget.setDecimals(3)
                control_widget.setSingleStep(0.001)
                control_widget.setRange(min_val, 1000)
                control_widget.setValue(0)
                control_widget.setKeyboardTracking(False)

                value = scale_value_for_display(key, getattr(pattern, key), constants.SI_TO_MICRO)
                control_widget.setValue(value)
                control_widget.valueChanged.connect(self.redraw_patterns)
            
            if isinstance(val, CrossSectionPattern):
                control_widget = QtWidgets.QComboBox()
                control_widget.addItems([section.name for section in CrossSectionPattern]) # TODO: store the options somewhere?
                control_widget.setCurrentText(getattr(pattern, key, CrossSectionPattern.Rectangle).name)
                control_widget.currentIndexChanged.connect(self.redraw_patterns)

            if key == "scan_direction": # TODO: migrate to Enum??
                control_widget = QtWidgets.QComboBox()
                control_widget.addItems(self.AVAILABLE_SCAN_DIRECTIONS)
                control_widget.setCurrentText(val)
                control_widget.currentIndexChanged.connect(self.redraw_patterns)

            if isinstance(val, bool):
                control_widget = QtWidgets.QCheckBox()
                control_widget.setChecked(bool(val))
                control_widget.stateChanged.connect(self.redraw_patterns)

            # add to grid layout
            self.gridLayout_patterns.addWidget(label, i, 0)
            self.gridLayout_patterns.addWidget(control_widget, i, 1)

            # store the widget
            self.pattern_attribute_widgets[key] = control_widget

            # attr:
                # float / int: QDoubleSpinBox
                # bool: QCheckBox
                # str: QLineEdit
                # enum: QComboBox
            # scaled?
            # display name
            # tooltip
            # advanced? (hidden by default)
            # events:
                # set
                # get
                # changed

            # TODO: add tooltips
            # TODO: add advanced flags (to hide options)
        
        # set the point values
        self.doubleSpinBox_centre_x.setValue(point.x * constants.SI_TO_MICRO)
        self.doubleSpinBox_centre_y.setValue(point.y * constants.SI_TO_MICRO)

        self.update_ui()

        self.UPDATING_PATTERN = False
    
    def get_pattern_from_ui_v2(self):

        pattern = self.current_milling_stage.pattern

        # get the updated pattern values from ui
        for i, key in enumerate(pattern.required_attributes):
            widget = self.pattern_attribute_widgets[key]

            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                value = scale_value_for_display(key, widget.value(), constants.MICRO_TO_SI)
            if isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()

                if key == "cross_section": # TODO: need a more general way to handle this
                    value = CrossSectionPattern[value]
            if isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()

            setattr(pattern, key, value)

        pattern.point = self.get_point_from_ui()
        
        # from pprint import pprint
        # pprint(pattern.to_dict())
        
        return pattern

    def _single_click(self, layer, event):
        """Callback for single click on image layer."""
        if event.button != 1 or 'Shift' not in event.modifiers or self.milling_stages == []:
            return

        if not self.CAN_MOVE_PATTERN:
            msg = "Pattern is not moveable."
            logging.info(msg)
            napari.utils.notifications.show_info(msg)
            return

        self.UPDATING_PATTERN = True

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)
        
        if beam_type is not BeamType.ION:
            napari.utils.notifications.show_info(
                f"Please right click on the {BeamType.ION.name} image to move pattern."
            )
            return
        
        point = conversions.image_to_microscope_image_coordinates(
                Point(x=coords[1], y=coords[0]), image.data, image.metadata.pixel_size.x,
            )
        
        point_clicked = deepcopy(point)

        # only move the pattern if milling widget is activate and beamtype is ion?
        renewed_patterns = []
        _patterns_valid = True

        current_stage_index = self.comboBox_milling_stage.currentIndex()

        diff = point - self.milling_stages[current_stage_index].pattern.point

        for idx, milling_stage in enumerate(self.milling_stages):
        # loop to check through all patterns to see if they are in bounds
            if 'Control' not in event.modifiers:
                if idx != current_stage_index: 
                    continue
            
            pattern_renew = deepcopy(milling_stage.pattern)

            if self.checkBox_relative_move.isChecked():
                point = Point(x=pattern_renew.point.x + diff.x, 
                              y=pattern_renew.point.y + diff.y)
            
            # if the pattern is a line, we also need to update start_x, start_y, end_x, end_y to move with the click
            if isinstance(pattern_renew, LinePattern):
                pattern_renew.start_x += diff.x
                pattern_renew.start_y += diff.y
                pattern_renew.end_x += diff.x
                pattern_renew.end_y += diff.y
                # TODO: this doesnt work if the line is rotated at all
            
            # update the pattern point
            pattern_renew.point = point

            pattern_is_valid = valid_pattern_location(pattern_renew, image=self.image_widget.ib_image)
            # TODO: if the line goes out of bounds it is not reset correcectly, need to fix this

            if not pattern_is_valid:
                logging.warning(f"Could not move Patterns, out of bounds at at {point}")
                napari.utils.notifications.show_warning("Patterns is not within the image.")
                _patterns_valid = False
                break
            else:
                renewed_patterns.append(pattern_renew)

        _redraw = False

        if _patterns_valid:

            if len(renewed_patterns) == len(self.milling_stages):
                for milling_stage, pattern_renew in zip(self.milling_stages, renewed_patterns):
                    milling_stage.pattern = pattern_renew
                _redraw = True
            elif len(renewed_patterns)>0:
                self.milling_stages[current_stage_index].pattern = renewed_patterns[0]
                _redraw = True

        if _redraw:
            self.doubleSpinBox_centre_x.setValue(point_clicked.x * constants.SI_TO_MICRO)
            self.doubleSpinBox_centre_y.setValue(point_clicked.y * constants.SI_TO_MICRO) # THIS TRIGGERS AN UPDATE
            logging.info(f"Moved patterns to {point} ")

            # if current pattern is Line, we need to update the ui elements
            if self.milling_stages[current_stage_index].pattern.name == "Line":
                # self.update_pattern_ui(milling_stage=self.milling_stages[current_stage_index])
                print("NOT IMPLEMENTED TODO: update line pattern ui")

            logging.debug({
                "msg": "move_milling_patterns",                                     # message type
                "pattern": self.milling_stages[current_stage_index].pattern.name,   # pattern name
                "dm": diff.to_dict(), # x, y                                        # metres difference 
                "beam_type": BeamType.ION.name,                                     # beam type
            })
            
            self.update_ui()
            self.milling_position_changed.emit()
            
        self.UPDATING_PATTERN = False

    
    # TODO: add a validation check for the patterns based on the image hfw, rather than napari shapes
   
    def set_milling_settings_ui(self) -> None:
        """Set the milling settings ui from the current milling stage."""
        milling = self.current_milling_stage.milling

        self.doubleSpinBox_milling_current.setValue(milling.milling_current * constants.SI_TO_NANO)
        self.comboBox_application_file.setCurrentText(milling.application_file)
        self.doubleSpinBox_rate.setValue(milling.rate*constants.SI_TO_NANO)
        self.doubleSpinBox_dwell_time.setValue(milling.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_spot_size.setValue(milling.spot_size * constants.SI_TO_MICRO)
        self.doubleSpinBox_hfw.setValue(milling.hfw * constants.SI_TO_MICRO)
        self.comboBox_preset.setCurrentText(str(milling.preset))
        self.spinBox_voltage.setValue(milling.milling_voltage)
        self.comboBox_patterning_mode.setCurrentText(milling.patterning_mode)

    def get_milling_settings_from_ui(self):
        """Get the Milling Settings from the UI."""
        current_amps = float(self.doubleSpinBox_milling_current.value()) * constants.NANO_TO_SI

        milling_settings = FibsemMillingSettings(
            milling_current=current_amps,
            application_file=self.comboBox_application_file.currentText(),
            rate=self.doubleSpinBox_rate.value()*1e-9,
            dwell_time = self.doubleSpinBox_dwell_time.value() * constants.MICRO_TO_SI,
            spot_size=self.doubleSpinBox_spot_size.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_hfw.value() * constants.MICRO_TO_SI,
            preset= self.comboBox_preset.currentText(),
            spacing=self.doubleSpinBox_spacing.value(),
            milling_voltage=self.spinBox_voltage.value(),
            patterning_mode=self.comboBox_patterning_mode.currentText()
        )

        return milling_settings

    def update_ui(self, milling_stages: List[FibsemMillingStage] = None):

        # force the milling hfw to match the current image hfw
        self.doubleSpinBox_hfw.setValue(self.image_widget.doubleSpinBox_image_hfw.value())
        
        t0 = time.time()

        self.update_milling_stage_from_ui()         # update milling stage from ui
        milling_stages = self.get_milling_stages() # get the latest milling stages from the ui

        t1 = time.time()

        if not milling_stages:
            remove_all_napari_shapes_layers(self.viewer)
            return

        # make milling stage a list if it is not
        if not isinstance(milling_stages, list):
            milling_stages = [milling_stages]

        # # check hfw threshold # TODO: do this check more seriously, rather than rule of thumb

        t2 = time.time()
        try:
            draw_crosshair = self.checkBox_show_milling_crosshair.isChecked()
            if not isinstance(self.image_widget.ib_image, FibsemImage):
                raise Exception("No Ion Image, cannot draw patterns. Please take an image.")
            if not isinstance(self.image_widget.eb_image, FibsemImage):
                raise Exception("No Electron Image, cannot draw patterns. Please take an image.") # TODO: this is unintuitive why this is required -> ui issue only
            # clear patterns then draw new ones
            self.milling_pattern_layers = draw_milling_patterns_in_napari(
                viewer=self.viewer, 
                ib_image=self.image_widget.ib_image, 
                eb_image=self.image_widget.eb_image, 
                milling_stages = milling_stages,
                draw_crosshair=draw_crosshair)  # TODO: add names and legend for this
        
        except Exception as e:
            napari.utils.notifications.show_error(f"Error drawing patterns: {e}")
            logging.error(e)
            return
        t2 = time.time()
        logging.debug(f"UPDATE_UI: GET: {t1-t0}, DRAW: {t2-t1}")
        self.viewer.layers.selection.active = self.image_widget.eb_layer

    def toggle_pattern_visibility(self):

        is_visible = self.checkBox_show_milling_patterns.isChecked()
        for layer in self.milling_pattern_layers:
            if layer in self.viewer.layers:
                self.viewer.layers[layer].visible = is_visible

    def _toggle_interactions(self, enabled: bool = True, caller: str = None, milling: bool = False):

        """Toggle microscope and pattern interactions."""
        self.pushButton_add_milling_stage.setEnabled(enabled)
        self.pushButton_remove_milling_stage.setEnabled(enabled)
        # self.pushButton_save_milling_stage.setEnabled(enabled)
        self.pushButton_run_milling.setEnabled(enabled)
        if enabled:
            self.pushButton_run_milling.setStyleSheet(stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.pushButton_run_milling.setText("Run Milling")
            self.pushButton_add_milling_stage.setStyleSheet(stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.pushButton_remove_milling_stage.setStyleSheet(stylesheets._RED_PUSHBUTTON_STYLE)
            self.pushButton_stop_milling.setVisible(False)
        elif milling:
            self.pushButton_run_milling.setStyleSheet(stylesheets._ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_run_milling.setText("Running...")
            self.pushButton_stop_milling.setVisible(True)
        else:
            self.pushButton_run_milling.setStyleSheet(stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_add_milling_stage.setStyleSheet(stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_remove_milling_stage.setStyleSheet(stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_stop_milling.setVisible(False)


        if caller is None:
            self.parent.image_widget._toggle_interactions(enabled, caller="milling")
            self.parent.movement_widget._toggle_interactions(enabled, caller="milling")


    def run_milling(self):
        
        worker = self.run_milling_step()
        worker.finished.connect(self.run_milling_finished)
        worker.start()
    
    def stop_milling(self):
        """Request milling stop."""
        self._STOP_MILLING = True
        self.microscope.stop_milling()
        self._quit_progress_bar()
        self.finish_progress_bar()
    
    def start_progress_thread(self,info):

        est_time = info['estimated_time']  # TODO: add some extra time to this to account for changing currents etc
        idx = info['idx']
        total = info['total']

        self.progressBar_milling.setVisible(True)
        self.progressBar_milling.setValue(0)
        self.progressBar_milling.setStyleSheet("QProgressBar::chunk "
                          "{"
                          "background-color: green;"
                          "}")
        self.progressBar_milling.setFormat(f"Milling stage {idx+1} of {total}: {est_time:.1f}s")

        # info = [idx, total, est_time]

        self.progress_bar_worker = self.start_progress_bar(info)
        self.progress_bar_worker.finished.connect(self.finish_progress_bar)

        self.progress_bar_worker.start()

    @thread_worker
    def start_progress_bar(self,info):
        
        est_time = info['estimated_time']
        info['progress_percent'] = 0
 
        i = 0
        # time.sleep(2)
        inc = 0.5
        while i < est_time:
            time.sleep(inc)
            progress_percent = (i+inc)/est_time
            info['progress_percent'] = progress_percent
            info['est_time'] = est_time - i
            self._progress_bar_update.emit(info)
            i += inc 
            yield


    def update_progress_bar(self, info):
        
        value = info['progress_percent']
        idx = info['idx']
        total = info['total']
        est_time = info['est_time']
        t_m = int(est_time // 60)
        t_s = int(est_time % 60)

        self.progressBar_milling.setVisible(True)
        self.progressBar_milling.setValue(value*100)
        self.progressBar_milling.setFormat(f"Milling Stage {idx+1}/{total}: {t_m:02d}:{t_s:02d} remaining...")

    def finish_progress_bar(self):
        self.progressBar_milling.setVisible(False)
        self.progressBar_milling.setValue(0)

    def _quit_progress_bar(self):
        self.progress_bar_worker.quit()

    @thread_worker
    def run_milling_step(self):

        milling_stages = self.get_milling_stages()
        self._toggle_interactions(enabled=False,milling=True)
        
        # new milling interface
        mill_stages(microscope=self.microscope, 
                            stages=milling_stages, 
                            parent_ui=self)
        return

    def update_milling_ui(self, msg: str):
        logging.info(msg)
        napari.utils.notifications.notification_manager.records.clear()
        napari.utils.notifications.show_info(msg)

    def run_milling_finished(self):

        # take new images and update ui
        self._toggle_interactions(enabled=True)
        self.image_widget.take_reference_images()
        self.update_ui()
        self._milling_finished.emit()
        self._quit_progress_bar()
        self.finish_progress_bar()
        self._STOP_MILLING = False


def valid_pattern_location(pattern: BasePattern, image: FibsemImage) -> bool:
    """Check if the pattern is within the image bounds."""

    if isinstance(pattern, FiducialPattern):
        _,flag = calculate_fiducial_area_v2(image=image, 
                                            fiducial_centre = deepcopy(pattern.point), 
                                            fiducial_length = pattern.height)
        
        if flag:
            napari.utils.notifications.show_warning("Fiducial reduce area is not within the image.")
            return False
        else:
            return True    
    
    for pattern_settings in pattern.define():
        if isinstance(pattern_settings, CirclePattern):
            napari_shape = convert_pattern_to_napari_circle(pattern_settings=pattern_settings, image=image)
        else:
            napari_shape = convert_pattern_to_napari_rect(pattern_settings=pattern_settings, image=image)

        is_valid_placement = validate_pattern_placement(resolution=image.data.shape[::-1], shape=napari_shape)
        if not is_valid_placement:
            return False
    
    return True


def main():
    # millings_stages = [
    # FibsemMillingStage(
    #     name="Milling Stage X",
    #     num = 1,
    #     milling = FibsemMillingSettings(hfw=400e-6),
    #     pattern = patterns.get_pattern("Trench"),
    # ),
    #     FibsemMillingStage(
    #     name="Milling Stage 2",
    #     num = 2,
    #     milling = FibsemMillingSettings(hfw=200e-6),
    #     pattern = patterns.get_pattern("Horseshoe"),
    # )
    # ]
    viewer = napari.Viewer(ndisplay=2)
    milling_widget = FibsemMillingWidget()
    viewer.window.add_dock_widget(
        milling_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()