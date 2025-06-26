import logging
import math
import os
from copy import deepcopy
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Layer
from napari.qt.threading import thread_worker
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QListWidgetItem
from scipy.ndimage import median_filter

from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.microscope import (
    FibsemMicroscope,
    ThermoMicroscope,
)
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.milling import (
    FibsemMillingStage,
    MillingAlignment,
    estimate_total_milling_time,
    get_strategy,
    mill_stages,
)
from fibsem.milling.patterning.patterns2 import (
    DEFAULT_MILLING_PATTERN,
    MILLING_PATTERN_NAMES,
    BasePattern,
    FiducialPattern,
    LinePattern,
    get_pattern,
)
from fibsem.milling.strategy import DEFAULT_STRATEGY_NAME, get_strategy_names
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemImage,
    FibsemMillingSettings,
    ImageSettings,
    MillingState,
    Point,
)
from fibsem.ui import stylesheets as stylesheets
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    is_pattern_placement_valid,
    remove_all_napari_shapes_layers,
)
from fibsem.ui.qtdesigner_files import FibsemMillingWidget as FibsemMillingWidgetUI
from fibsem.utils import format_duration

# 3D Correlation Widget
CORRELATION_THREEDCT_AVAILABLE = False
try:
    from tdct.app import CorrelationUI
    logging.info("CorrelationUI imported from tdct.app")
    CORRELATION_THREEDCT_AVAILABLE = True
except ImportError as e:
    logging.debug(f"Could not import CorrelationUI from tdct.app: {e}")
    CorrelationUI = None

UNSCALED_VALUES = [
    "rotation",
    "size_ratio",
    "scan_direction",
    "cleaning_cross_section",
    "number",
    "passes",
    "n_rectangles",
    "overlap",
    "inverted",
    "use_side_patterns",
    "n_columns",
    "n_rows",
    "cross_section",
    "time",
    "milling_interval", 
    "max_milling_cycles",
    "sem_frame_integration",
    "fib_frame_integration",
    "overtilt",
]
LINE_KEYS = ["start_x", "start_y", "end_x", "end_y"]

MILLING_WIDGET_INSTRUCTIONS = """Controls:
Shift + Left Click to Move Selected Pattern
Ctrl + Shift + Left Click to Move All Patterns
Press Run Milling to Start Milling"""

def scale_value_for_display(key: str, value: Union[float, int], scale: float) -> Union[float, int]:
    if key not in UNSCALED_VALUES:
        return value * scale    
    return value



# TODO: make these more generic for other units, or use pint
def _format_beam_current_as_str(val: float):
    scale = 1
    unit = "A"
    if val < 1e-9:
        scale = 1e12
        unit = "pA"
    elif val < 1e-6:
        scale = 1e9
        unit = "nA"
    elif val < 1e-3:
        scale = 1e6
        unit = "uA"

    return f"{math.ceil(val*scale)} {unit}"
    
def _parse_beam_current_str(val: str) -> float:
    scale = 1
    unit = "A"
    if "pA" in val:
        scale = 1e-12
    elif "nA" in val:
        scale = 1e-9
    elif "uA" in val:
        scale = 1e-6
    return float(val.split(" ")[0]) * scale

class WheelBlocker(QObject):
    """Event filter that blocks wheel events"""
    
    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel:
            return True  # Block the wheel event
        return super().eventFilter(watched, event)

class FibsemMillingWidget(FibsemMillingWidgetUI.Ui_Form, QtWidgets.QWidget):
    milling_position_changed = QtCore.pyqtSignal()
    milling_progress_signal = QtCore.pyqtSignal(dict) # TODO: replace with pysygnal (pure-python signal)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        viewer: napari.Viewer,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent
        self.microscope = microscope

        self.viewer = viewer
        self.image_widget: FibsemImageSettingsWidget = parent.image_widget
        
        # correlation widget
        self.correlation_widget: CorrelationUI = None

        self.UPDATING_PATTERN: bool = False
        self.CAN_MOVE_PATTERN: bool = True
        self.STOP_MILLING: bool = False
        self.MILLING_STAGES_INITIALISED: bool = False

        self.current_milling_stage: Optional[FibsemMillingStage] = None
        self.milling_stages: List[FibsemMillingStage] = []
        self.milling_pattern_layers: List[Layer] = []

        self.pattern_attribute_widgets: Dict[str, Tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}
        self.strategy_config_widgets: Dict[str, QtWidgets.QWidget] = {}

        self.setup_connections()
        # TODO: migrate to MILLING_WORKFLOWS: Dict[str, List[FibsemMillingStage]]

    def setup_connections(self):
        """Setup the connections for the milling widget."""

        is_thermo = isinstance(self.microscope, ThermoMicroscope)
        is_tescan = isinstance(self.microscope, TescanMicroscope)

        if isinstance(self.microscope, DemoMicroscope) or (not is_thermo and not is_tescan):
            is_thermo, is_tescan = True, False
        
        # MILLING SETTINGS
        # general
        AVAILABLE_MILLING_MODES: List[str] = ["Serial", "Parallel"] # TODO: make microscope method
        self.AVAILABLE_SCAN_DIRECTIONS = self.microscope.get_available_values(key="scan_direction")
        self.comboBox_patterning_mode.addItems(AVAILABLE_MILLING_MODES)
        self.comboBox_patterning_mode.currentIndexChanged.connect(self.update_milling_settings_from_ui)
        self.doubleSpinBox_hfw.valueChanged.connect(self.update_milling_settings_from_ui)
        self.checkBox_milling_acquire_images.stateChanged.connect(self.update_milling_settings_from_ui)
        self.checkBox_milling_acquire_images.stateChanged.connect(self.toggle_imaging_visibility)
        
        # ThermoFisher Only 
        self.label_application_file.setVisible(is_thermo)
        self.comboBox_application_file.setVisible(is_thermo)
        self.comboBox_milling_current.setVisible(is_thermo)
        self.label_milling_current.setVisible(is_thermo)
        self.label_voltage.setVisible(is_thermo)
        self.spinBox_voltage.setVisible(is_thermo) # TODO: set this to the available voltages
        if is_thermo:
            AVAILABLE_APPLICATION_FILES = self.microscope.get_available_values("application_file")
            self.comboBox_application_file.addItems(AVAILABLE_APPLICATION_FILES)

            self.AVAILABLE_MILLING_CURRENTS = self.microscope.get_available_values("current", BeamType.ION)
            for current in self.AVAILABLE_MILLING_CURRENTS:
                label = _format_beam_current_as_str(current)
                self.comboBox_milling_current.addItem(label, current)
            self.comboBox_application_file.currentIndexChanged.connect(self.update_milling_settings_from_ui)
            self.comboBox_milling_current.currentIndexChanged.connect(self.update_milling_settings_from_ui)
            self.spinBox_voltage.valueChanged.connect(self.update_milling_settings_from_ui)

        # Tescan Only
        AVAILABLE_PRESETS = self.microscope.get_available_values("presets", BeamType.ION)
        self.label_rate.setVisible(is_tescan)
        self.label_spot_size.setVisible(is_tescan)
        self.label_dwell_time.setVisible(is_tescan)
        self.doubleSpinBox_rate.setVisible(is_tescan)
        self.doubleSpinBox_spot_size.setVisible(is_tescan)
        self.doubleSpinBox_dwell_time.setVisible(is_tescan)   
        self.comboBox_preset.setVisible(is_tescan)
        self.label_preset.setVisible(is_tescan)
        self.label_spacing.setVisible(is_tescan)
        self.doubleSpinBox_spacing.setVisible(is_tescan)
        if is_tescan:   
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

        # strategy
        self.comboBox_strategy_name.addItems(get_strategy_names())
        self.comboBox_strategy_name.setCurrentText(DEFAULT_STRATEGY_NAME)
        self.comboBox_strategy_name.currentIndexChanged.connect(self.update_current_selected_strategy) # TODO: connect event
        # TODO: auto-update drift correction and strategy on value changes

        # image acquisition
        self.comboBox_imaging_resolution.addItems(cfg.STANDARD_RESOLUTIONS)
        self.comboBox_imaging_resolution.setCurrentText(cfg.DEFAULT_STANDARD_RESOLUTION)
        self.groupBox_imaging.setVisible(False)

        # imaging updates
        self.doubleSpinBox_imaging_dwell_time.valueChanged.connect(self.update_ui)
        self.doubleSpinBox_imaging_fov.valueChanged.connect(self.update_ui)
        self.comboBox_imaging_resolution.currentIndexChanged.connect(self.update_ui)
        self.checkBox_imaging_use_autocontrast.stateChanged.connect(self.update_ui)
        self.checkBox_imaging_save.stateChanged.connect(self.update_ui)

        # milling stages
        self.pushButton_add_milling_stage.clicked.connect(self.add_milling_stage)
        self.pushButton_remove_milling_stage.clicked.connect(self.remove_milling_stage)
        self.comboBox_milling_stage.currentIndexChanged.connect(self.update_milling_stage_ui)
        self.listWidget_active_milling_stages.itemSelectionChanged.connect(self.on_selection_changed)
        self.listWidget_active_milling_stages.itemChanged.connect(self.on_stage_checked)
        # update ui
        self.milling_progress_signal.connect(self.handle_milling_progress_update)

        # run milling
        self.pushButton_run_milling.clicked.connect(self.run_milling)
        
        # stop milling
        self.pushButton_stop_milling.clicked.connect(self.stop_milling)
        self.pushButton_pause_milling.clicked.connect(self.pause_resume_milling)
        self.pushButton_stop_milling.setVisible(False)
        self.pushButton_pause_milling.setVisible(False)

        # set styles
        self.pushButton_add_milling_stage.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove_milling_stage.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_run_milling.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_stop_milling.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        self.pushButton_pause_milling.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.progressBar_milling.setStyleSheet(stylesheets.PROGRESS_BAR_GREEN_STYLE)
        self.progressBar_milling_stages.setStyleSheet(stylesheets.PROGRESS_BAR_BLUE_STYLE)
        
        # progress bar
        self.progressBar_milling.setVisible(False)
        self.progressBar_milling_stages.setVisible(False)
        self.label_milling_information.setVisible(False)

        # last
        self.doubleSpinBox_centre_x.setKeyboardTracking(False)
        self.doubleSpinBox_centre_y.setKeyboardTracking(False)
        self.doubleSpinBox_centre_x.valueChanged.connect(self.redraw_patterns)
        self.doubleSpinBox_centre_y.valueChanged.connect(self.redraw_patterns)
        self.doubleSpinBox_centre_x.setSuffix(" µm")
        self.doubleSpinBox_centre_y.setSuffix(" µm")

        # options
        self.checkBox_show_milling_patterns.setChecked(True)
        self.checkBox_show_milling_crosshair.setChecked(True)
        self.checkBox_show_milling_crosshair.stateChanged.connect(self.redraw_patterns)
        self.checkBox_show_milling_patterns.stateChanged.connect(self.toggle_pattern_visibility)
        self.checkBox_show_advanced.stateChanged.connect(self.show_advanced_settings)
            
        self.label_milling_instructions.setText(MILLING_WIDGET_INSTRUCTIONS)
        self.show_milling_stage_widgets()

        # feature flags
        DRIFT_CORRECTION_ENABLED = False
        self.pushButton_alignment_edit_area.setVisible(False) # TODO: enable setting alignment area via ui
        self.checkBox_alignment_interval_enabled.setVisible(DRIFT_CORRECTION_ENABLED)
        self.label_alignment_interval.setVisible(DRIFT_CORRECTION_ENABLED)
        self.doubleSpinBox_alignment_interval.setVisible(DRIFT_CORRECTION_ENABLED)
        self.checkBox_alignment_enabled.toggled.connect(self.update_milling_settings_from_ui)

        # external signals
        self.image_widget.viewer_update_signal.connect(self.update_ui) # update the ui when the image is updated

        self.pushButton_open_correlation.clicked.connect(self.open_3d_correlation_widget)
        self.pushButton_open_correlation.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.pushButton_open_correlation.setVisible(CORRELATION_THREEDCT_AVAILABLE)

        # block scroll wheel events for pattern combobox. 
        # QUERY: implement for other comboboxes?
        self.wheel_blocker = WheelBlocker()
        self.comboBox_patterns.installEventFilter(self.wheel_blocker)

    def on_selection_changed(self):
        """Selection changed callback for the milling stages list widget."""
        selected_items = self.listWidget_active_milling_stages.selectedItems()
        if selected_items:
            # get the index of the selected item
            index = self.listWidget_active_milling_stages.currentRow()
            self.comboBox_milling_stage.setCurrentIndex(index)
            # QUERY: remove combobox, just use list widget?
            # QUERY: support renaming stages
    
    def on_stage_checked(self, item: QListWidgetItem):
        # TODO: add a check to see if to enable/disable milling
        self.update_ui()

    def update_milling_settings_from_ui(self):
        """Update the milling settings from the UI."""
        index = self.comboBox_milling_stage.currentIndex()
        if index == -1:
            return 
        settings = self.get_milling_settings_from_ui()
        self.milling_stages[index].milling = settings
        self.update_ui()

    def add_milling_stage(self):
        logging.info("Adding milling stage")

        num = len(self.milling_stages) + 1
        name = f"Milling Stage {num}"
        pattern = DEFAULT_MILLING_PATTERN()
        strategy = get_strategy(DEFAULT_STRATEGY_NAME)
        milling_stage = FibsemMillingStage(name=name, 
                                           num=num, 
                                           pattern=pattern, 
                                           strategy=strategy)
        self.milling_stages.append(deepcopy(milling_stage))
        self.comboBox_milling_stage.addItem(name)
        self.comboBox_milling_stage.setCurrentText(name)
        napari.utils.notifications.show_info(f"Added {name}.")
        self.update_selected_milling_stages_ui()
        self.update_ui()

    def update_selected_milling_stages_ui(self):
        # update the combobox and list widget
        # TODO: sync with milling_stages
        # TODO: convert to signal

        # clear the list widget
        self.listWidget_active_milling_stages.clear()

        for stage in self.milling_stages:
            txt = f"{stage.name} - {stage.pattern.name} ({format_duration(stage.estimated_time)})"
            item = QListWidgetItem(txt)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.listWidget_active_milling_stages.addItem(item)

    def remove_milling_stage(self):
        logging.info("Removing milling stage")
        # self.image_widget.clear_alignment_area() # DISABLED

        current_index = self.comboBox_milling_stage.currentIndex()
        if current_index == -1:
            return
        self.milling_stages.pop(current_index)
        self.comboBox_milling_stage.removeItem(current_index)
        napari.utils.notifications.show_info("Removed milling stage.")
        self.update_selected_milling_stages_ui()
        self.update_ui()

    def clear_all_milling_stages(self):
        """Remove all milling stages from the widget."""
        self.milling_stages = []
        self.comboBox_milling_stage.currentIndexChanged.disconnect()
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.currentIndexChanged.connect(self.update_milling_stage_ui)
        self.update_selected_milling_stages_ui()

        remove_all_napari_shapes_layers(self.viewer) # remove all shape layers
        # self.image_widget.clear_alignment_area() # DISABLED

    def set_milling_stages(self, milling_stages: List[FibsemMillingStage]) -> None:
        """Set the milling stages in the widget and update the UI."""
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
        self.update_selected_milling_stages_ui()
        self.update_ui()

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
        self.listWidget_active_milling_stages.setVisible(show)
        self.label_milling_estimated_time.setVisible(show)

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

        self.MILLING_STAGES_INITIALISED = False
        # update the milling stage UI
        self.set_milling_settings_ui()        
        self.set_milling_strategy_ui()
        self.set_milling_alignment_ui()
        self.set_pattern_settings_ui()
        self.set_imaging_settings_ui()
        self.MILLING_STAGES_INITIALISED = True
        self.show_milling_stage_widgets()

        self.update_ui()

    def set_milling_strategy_ui(self):
        """Set the milling strategy UI from the current milling stage."""
        strategy = self.current_milling_stage.strategy

        self.comboBox_strategy_name.setCurrentText(strategy.name)
        
        # clear the grid layout
        self.clear_grid_layout(self.gridLayout_strategy)
        
        for i, key in enumerate(strategy.config.required_attributes):
            label = QtWidgets.QLabel(key.replace("_", " ").title())    

            # default None
            val = getattr(strategy.config, key, None)

            if isinstance(val, (int, float)) and not isinstance(val, bool):
                # limits
                min_val = -1000

                control_widget = QtWidgets.QDoubleSpinBox()
                control_widget.setDecimals(1)
                control_widget.setSingleStep(0.001)
                control_widget.setRange(min_val, 1000)
                control_widget.setValue(0)
                control_widget.setKeyboardTracking(False)

                val = scale_value_for_display(key, val, constants.SI_TO_MICRO) # TODO: confirm
                if key in ["overtilt"]:
                    control_widget.setSuffix(" °")
                control_widget.setValue(val)
            if isinstance(val, str):
                control_widget = QtWidgets.QLineEdit()
                control_widget.setText(val)
            if isinstance(val, bool):
                control_widget = QtWidgets.QCheckBox()
                control_widget.setChecked(bool(val))
            if isinstance(val, (tuple, list)):
                # dont handle for now
                if "resolution" in key:
                    control_widget = QtWidgets.QComboBox()
                    control_widget.addItems(cfg.STANDARD_RESOLUTIONS)
                    control_widget.setCurrentText(f"{val[0]}x{val[1]}") # TODO: check if in list

            # TODO: add support for scaling, str, bool, etc.
            # TODO: attached events

            # add to grid layout
            self.gridLayout_strategy.addWidget(label, i, 0)
            self.gridLayout_strategy.addWidget(control_widget, i, 1)

            # store the widget
            self.strategy_config_widgets[key] = (label, control_widget)

        self.show_advanced_settings()

    def get_milling_strategy_from_ui(self):
        """Get the milling strategy from the UI."""
        strategy = self.current_milling_stage.strategy

        # get the updated pattern values from ui
        for i, key in enumerate(strategy.config.required_attributes):
            
            if key not in self.strategy_config_widgets:
                continue

            label, widget = self.strategy_config_widgets[key]

            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                value = scale_value_for_display(key, widget.value(), constants.MICRO_TO_SI) # TODO: support other scales
            if isinstance(widget, QtWidgets.QLineEdit):
                value = widget.text()
            if isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()
            if isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()
                if "resolution" in key:
                    value = tuple(map(int, value.split("x")))
            # TODO: add support for scaling, str, bool, etc.

            setattr(strategy.config, key, value)

        return strategy

    def set_milling_alignment_ui(self):
        """Set the drift correction UI from the current milling stage."""
        milling_alignment = self.current_milling_stage.alignment

        self.checkBox_alignment_enabled.setChecked(milling_alignment.enabled)
        self.checkBox_alignment_interval_enabled.setChecked(milling_alignment.interval_enabled)
        self.doubleSpinBox_alignment_interval.setValue(milling_alignment.interval)

        # display the alignment area
        # self.image_widget.toggle_alignment_area(milling_alignment.rect, editable=False)

    def get_milling_alignment_from_ui(self):
        """Get the drift correction settings from the UI."""
        milling_alignment = self.current_milling_stage.alignment
        milling_alignment.enabled=self.checkBox_alignment_enabled.isChecked()
        milling_alignment.interval_enabled=self.checkBox_alignment_interval_enabled.isChecked()
        milling_alignment.interval=self.doubleSpinBox_alignment_interval.value()
        #milling_alignment.rect=self.image_widget.get_alignment_area(), # TODO: get the alignment area from the image widget
        return milling_alignment

    def toggle_imaging_visibility(self):
        """Toggle the visibility of the imaging settings."""
        visible = self.checkBox_milling_acquire_images.isChecked()
        self.groupBox_imaging.setVisible(visible)

    def get_imaging_settings_from_ui(self):
        """Get the imaging settings from the UI."""

        imaging_settings = self.current_milling_stage.imaging

        # imaging settings (only a partial update)
        resolution = list(map(int, self.comboBox_imaging_resolution.currentText().split("x")))
        imaging_settings.resolution=resolution
        imaging_settings.dwell_time=self.doubleSpinBox_imaging_dwell_time.value() * constants.MICRO_TO_SI
        imaging_settings.hfw=self.doubleSpinBox_imaging_fov.value() * constants.MICRO_TO_SI
        imaging_settings.autocontrast=self.checkBox_imaging_use_autocontrast.isChecked()
        imaging_settings.save = self.checkBox_imaging_save.isChecked()

        return imaging_settings

    def set_imaging_settings_ui(self):
        """Set the imaging settings UI from the current milling stage."""
        imaging_settings = self.current_milling_stage.imaging
        res = imaging_settings.resolution
        res_str = f"{res[0]}x{res[1]}"
        if res_str not in cfg.STANDARD_RESOLUTIONS:
            logging.warning(f"Resolution {res_str} not in standard resolutions, using default {cfg.DEFAULT_STANDARD_RESOLUTION}.")
            res_str = cfg.DEFAULT_STANDARD_RESOLUTION
        self.comboBox_imaging_resolution.setCurrentText(res_str)
        self.doubleSpinBox_imaging_dwell_time.setValue(imaging_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_imaging_fov.setValue(imaging_settings.hfw * constants.SI_TO_MICRO)
        self.checkBox_imaging_use_autocontrast.setChecked(imaging_settings.autocontrast)
        self.checkBox_imaging_save.setChecked(imaging_settings.save)

    def redraw_patterns(self):
        """Redraw the patterns in the viewer."""
        if self.UPDATING_PATTERN:
            return

        # refresh the pattern
        self.update_ui()

    def update_milling_stage_from_ui(self) -> None:
        """Get the updated milling stage from the UI."""

        # get current milling stage
        current_index = self.comboBox_milling_stage.currentIndex()

        if current_index == -1:
            return

        milling_stage: FibsemMillingStage = self.milling_stages[current_index]

        # update all milling stage settings from the UI
        milling_stage.milling = self.get_milling_settings_from_ui()
        milling_stage.pattern = self.get_pattern_from_ui_v2()
        milling_stage.alignment = self.get_milling_alignment_from_ui()
        milling_stage.strategy = self.get_milling_strategy_from_ui()
        milling_stage.imaging = self.get_imaging_settings_from_ui() # NOTE: this is shared across all stages?

    def clear_grid_layout(self, grid_layout: QtWidgets.QGridLayout) -> None:
        # clear layout
        for i in reversed(range(grid_layout.count())):
            grid_layout.itemAt(i).widget().setParent(None)

    def update_current_selected_strategy(self):
        """Update the current selected strategy."""

        strategy_name = self.comboBox_strategy_name.currentText()
        logging.info(f"selected strategy: {strategy_name}")
        strategy = get_strategy(strategy_name)
        self.current_milling_stage.strategy = strategy
        self.set_milling_strategy_ui()

    def update_current_selected_pattern(self):
        """When the currently selected pattern is changed, update the current milling stage, and then reset the ui."""

        pattern_name = self.comboBox_patterns.currentText()

        logging.info(f"selected pattern: {pattern_name}")
        pattern = get_pattern(pattern_name)
        self.current_milling_stage.pattern = pattern
        self.set_pattern_settings_ui()

    def set_pattern_settings_ui(self):
        """When the pattern is changed, setup the ui for the new pattern."""

        # if the pattern is change? what happens
        # get the new currently selected pattern
        # set the new pattern to the current milling stage
        # update the ui 

        milling_stage = self.current_milling_stage # the currently selected milling stage
        if milling_stage is None:
            return

        self.UPDATING_PATTERN = True
        
        pattern: BasePattern = milling_stage.pattern
        point: Point = pattern.point

        # check if the combo box is set to the correct pattern, if not, set it without triggering the event
        current_pattern_name = self.comboBox_patterns.currentText()
        if pattern.name != current_pattern_name:
            self.comboBox_patterns.currentIndexChanged.disconnect()
            self.comboBox_patterns.setCurrentText(pattern.name)
            self.comboBox_patterns.currentIndexChanged.connect(self.update_current_selected_pattern)

        # clear the grid layout
        self.clear_grid_layout(self.gridLayout_patterns)


        # set widgets for each required key / value
        for i, key in enumerate(pattern.required_attributes):

            label = QtWidgets.QLabel(key.replace("_", " ").title())

            # default None
            val = getattr(pattern, key, None)

            if isinstance(val, (int, float)):
                # limits
                min_val = -1000 # if key in LINE_KEYS else 0 # why not?

                control_widget = QtWidgets.QDoubleSpinBox()
                control_widget.setDecimals(2)
                control_widget.setSingleStep(0.001)
                control_widget.setRange(min_val, 1000)
                control_widget.setValue(0)
                control_widget.setKeyboardTracking(False)

                # set the suffix (unit)
                # TODO: move to patterns2.py
                units_map = {
                             "time": " s", 
                             "rotation": "°"}
                
                unit = units_map.get(key, " µm")
                if key not in UNSCALED_VALUES or key in units_map:
                    control_widget.setSuffix(unit)
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
            self.pattern_attribute_widgets[key] = (label, control_widget)

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

        self.show_advanced_settings()
        self.update_ui()

        self.UPDATING_PATTERN = False
    
    def show_advanced_settings(self):
        """Toggle the advanced settings visibility."""
        show = self.checkBox_show_advanced.isChecked()
        
        pattern = self.current_milling_stage.pattern 
        for i, key in enumerate(pattern.advanced_attributes):
            if key in self.pattern_attribute_widgets:
                label, widget = self.pattern_attribute_widgets[key]
                label.setVisible(show)
                widget.setVisible(show)
        
        strategy_config = self.current_milling_stage.strategy.config 
        for i, key in enumerate(strategy_config.advanced_attributes):
            if key in self.strategy_config_widgets:
                label, widget = self.strategy_config_widgets[key]
                label.setVisible(show)
                widget.setVisible(show)

    def get_pattern_from_ui_v2(self):

        pattern = self.current_milling_stage.pattern

        # get the updated pattern values from ui
        for i, key in enumerate(pattern.required_attributes):
            label, widget = self.pattern_attribute_widgets[key]

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

        current_stage_index = self.comboBox_milling_stage.currentIndex()

        # no milling stage selected
        if current_stage_index == -1:
            return

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)

        if beam_type is not BeamType.ION:
            napari.utils.notifications.show_warning("Patterns can only be placed inside the FIB image.")
            return
        
        point = conversions.image_to_microscope_image_coordinates(
                coord=Point(x=coords[1], y=coords[0]), # yx required
                image=image.data,
                pixelsize=image.metadata.pixel_size.x,
            )
        point_clicked = deepcopy(point)
        
        # conditions to move:
        #   all moved patterns are within the fib image
        renewed_patterns = []
        all_patterns_are_valid: bool = True
        moving_all_patterns: bool = bool('Control' in event.modifiers)
        use_relative_movement: bool = self.checkBox_relative_move.isChecked()

        diff = point - self.milling_stages[current_stage_index].pattern.point

        self.UPDATING_PATTERN = True

        # loop to check through all patterns to see if they are in bounds
        for idx, milling_stage in enumerate(self.milling_stages):
            if not moving_all_patterns:
                if idx != current_stage_index: 
                    continue

            pattern_renew = deepcopy(milling_stage.pattern)

            if use_relative_movement:
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

            pattern_is_valid = is_pattern_placement_valid(pattern=pattern_renew, 
                                                          image=self.image_widget.ib_image)
            # TODO: if the line goes out of bounds it is not reset correctly, need to fix this

            if not pattern_is_valid:
                all_patterns_are_valid = False
                msg = f"{milling_stage.name} pattern is not within the FIB image."
                logging.warning(msg)
                napari.utils.notifications.show_warning(msg)
                break
            renewed_patterns.append(pattern_renew)

        if all_patterns_are_valid:

            if moving_all_patterns: # moving all patterns
                for milling_stage, pattern_renew in zip(self.milling_stages, renewed_patterns):
                    milling_stage.pattern = pattern_renew
            else: # only moving selected pattern
                self.milling_stages[current_stage_index].pattern = renewed_patterns[0]

            # update the ui
            self.doubleSpinBox_centre_x.setValue(point_clicked.x * constants.SI_TO_MICRO)
            self.doubleSpinBox_centre_y.setValue(point_clicked.y * constants.SI_TO_MICRO) # THIS TRIGGERS AN UPDATE
            logging.debug(f"Moved patterns to {point}")

            # if current pattern is Line, we need to update the ui elements
            if isinstance(self.milling_stages[current_stage_index].pattern, LinePattern):
                # self.update_pattern_ui(milling_stage=self.milling_stages[current_stage_index])
                logging.error("NOT IMPLEMENTED TODO: update line pattern ui")

            logging.debug({
                "msg": "move_milling_patterns",                                     # message type
                "pattern": self.milling_stages[current_stage_index].pattern.name,   # pattern name
                "dm": diff.to_dict(), # x, y                                        # metres difference 
                "beam_type": BeamType.ION.name,                                     # beam type
            })
            
            # redraw the patterns
            self.update_ui()
            self.milling_position_changed.emit()
            
        self.UPDATING_PATTERN = False

    # TODO: add a validation check for the patterns based on the image hfw, rather than napari shapes
   
    def set_milling_settings_ui(self) -> None:
        """Set the milling settings ui from the current milling stage."""
        milling = self.current_milling_stage.milling

        idx = self.comboBox_milling_current.findData(milling.milling_current)
        if idx == -1 and not isinstance(self.microscope, TescanMicroscope):
            # get the closest value
            milling_currents = [self.comboBox_milling_current.itemData(i) for i in range(self.comboBox_milling_current.count())]
            closest_value = min(milling_currents,key=lambda x: abs(x - milling.milling_current))
            idx = self.comboBox_milling_current.findData(closest_value)
        self.comboBox_milling_current.setCurrentIndex(idx)
        self.comboBox_application_file.setCurrentText(milling.application_file)
        self.doubleSpinBox_rate.setValue(milling.rate * constants.SI_TO_NANO)
        self.doubleSpinBox_dwell_time.setValue(milling.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_spot_size.setValue(milling.spot_size * constants.SI_TO_MICRO)
        self.doubleSpinBox_hfw.setValue(milling.hfw * constants.SI_TO_MICRO)
        self.comboBox_preset.setCurrentText(str(milling.preset))
        self.spinBox_voltage.setValue(int(milling.milling_voltage))
        self.comboBox_patterning_mode.setCurrentText(milling.patterning_mode)
        self.checkBox_milling_acquire_images.setChecked(milling.acquire_images)

    def get_milling_settings_from_ui(self):
        """Get the Milling Settings from the UI."""

        # get the milling current
        current_amps = 20e-12 # default value
        if isinstance(self.microscope, (ThermoMicroscope, DemoMicroscope)):
            current_amps = self.comboBox_milling_current.currentData()

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
            patterning_mode=self.comboBox_patterning_mode.currentText(),
            acquire_images=self.checkBox_milling_acquire_images.isChecked(),
        )

        return milling_settings

    def update_estimated_time(self, milling_stages: List[FibsemMillingStage]) -> None:
        # need to update the selectbox
        for idx, st in enumerate(self.milling_stages):
            txt = f"{st.name} - {st.pattern.name} ({format_duration(st.estimated_time)})"
            if idx < self.listWidget_active_milling_stages.count():
                self.listWidget_active_milling_stages.item(idx).setText(txt)

        # estimate the total milling time
        total_time = estimate_total_milling_time(milling_stages)
        self.label_milling_estimated_time.setText(f"Estimated Milling Time: {format_duration(total_time)}")

    def get_selected_milling_stages(self) -> Optional[FibsemMillingStage]:
        """Return the milling stages that are checked in the list widget."""
        checked_indexes = []
        for i in range(self.listWidget_active_milling_stages.count()):
            item = self.listWidget_active_milling_stages.item(i)
            if item.checkState() == Qt.Checked:
                checked_indexes.append(i)

        # get the checked milling stages
        checked_milling_stages = [self.milling_stages[idx] for idx in checked_indexes 
                                  if idx < len(self.milling_stages)]

        return checked_milling_stages

    def run_milling(self):
        """Run the selected milling stages."""

        # disable ui interactions
        self._toggle_interactions(enabled=False, milling=True)

        # get the selected milling stages
        self.update_milling_stage_from_ui()
        selected_milling_stages = self.get_selected_milling_stages()

        # start milling
        worker = self.run_milling_step(selected_milling_stages)
        worker.finished.connect(self.run_milling_finished)
        worker.start()
       
    @thread_worker
    def run_milling_step(self, milling_stages: List[FibsemMillingStage]) -> None:
        """Threaded worker to run the milling stages."""
        # new milling interface
        mill_stages(microscope=self.microscope, 
                            stages=milling_stages, 
                            parent_ui=self)
        return

    def run_milling_finished(self):

        # take new images and update ui
        self._toggle_interactions(enabled=True)
        self.image_widget.acquire_reference_images()
        self.update_ui() # TODO: convert to signal for image acquisition
        self.milling_progress_signal.emit({"finished": True, "msg": "Milling Finished."})
        self.STOP_MILLING = False

    def stop_milling(self):
        """Request milling stop."""
        self.STOP_MILLING = True
        self.microscope.stop_milling()
        self.milling_progress_signal.emit({"finished": True, "msg": "Milling Stopped by User."})

    def pause_resume_milling(self):
        """Request milling pause / resume."""
        milling_state = self.microscope.get_milling_state()
        if milling_state is MillingState.RUNNING:
            self.microscope.pause_milling()
            self.pushButton_pause_milling.setText("Resume Milling")
            self.pushButton_pause_milling.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_run_milling.setStyleSheet(stylesheets.GRAY_PUSHBUTTON_STYLE)
            self.pushButton_run_milling.setText("Milling Paused")
            self.milling_progress_signal.emit({"msg": "Milling Paused by User."})
        if milling_state is MillingState.PAUSED:
            self.microscope.resume_milling()
            # self.pushButton_pause_milling.setText("Pause Milling")
            # self.pushButton_pause_milling.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
            # self.pushButton_run_milling.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            # self.pushButton_run_milling.setText("Running...")
            self._toggle_interactions(enabled=False, milling=True)
            self.milling_progress_signal.emit({"msg": "Milling Resumed by User."})

    def update_progress_bar(self, progress_info: dict) -> None:
        """Update the milling progress bar."""

        state = progress_info.get('state', None)

        # start milling stage progress bar
        if state == "start":
            current_stage = progress_info.get('current_stage', 0)
            total_stages = progress_info.get('total_stages', 1)
            self.progressBar_milling.setVisible(True)
            self.progressBar_milling_stages.setVisible(True)
            self.progressBar_milling.setValue(0)
            self.progressBar_milling.setRange(0, 100)
            self.progressBar_milling.setFormat("Preparing Milling Conditions...")
            self.progressBar_milling_stages.setRange(0, 100)
            self.progressBar_milling_stages.setValue(int((current_stage+1)/total_stages*100))
            self.progressBar_milling_stages.setFormat(f"Milling Stage: {current_stage+1}/{total_stages}")
        
        # update
        if state == "update":
            logging.debug(progress_info)

            estimated_time = progress_info.get('estimated_time', None)
            remaining_time = progress_info.get('remaining_time', None)
            start_time = progress_info.get('start_time', None)
            
            if any([estimated_time is None, remaining_time is None]):
                return
            
            # calculate the percent complete
            percent_complete = int((1 - (remaining_time / estimated_time)) * 100)
            self.progressBar_milling.setValue(percent_complete)
            self.progressBar_milling.setFormat(f"Current Stage: {format_duration(remaining_time)} remaining...")

        # finished
        if state == "finished":
            self.progressBar_milling.setVisible(False)
            self.progressBar_milling_stages.setVisible(False)

    def handle_milling_progress_update(self, ddict: Dict[str, Union[str, int, float]]) -> None:
        """Handler the milling update. Displays messages to user, updates progress bar, etc."""

        logging.debug(f"Progress Update: {ddict}")

        msg = ddict.get("msg", None)
        if msg is not None:
            logging.debug(msg)
            self.label_milling_information.setVisible(True)
            self.label_milling_information.setText(msg)

        progress_info = ddict.get("progress", None)
        if progress_info is not None:
            self.update_progress_bar(progress_info)
            
        # finsihed milling
        if ddict.get("finished", False):
            self.MILLING_IS_FINISHED = True
            self.update_progress_bar({"state": "finished"})

    def _toggle_interactions(self, enabled: bool = True, caller: str = None, milling: bool = False):
        """Toggle microscope and pattern interactions."""

        self.pushButton_add_milling_stage.setEnabled(enabled)
        self.pushButton_remove_milling_stage.setEnabled(bool(enabled and self.milling_stages))
        self.pushButton_run_milling.setEnabled(bool(enabled and self.milling_stages))
        self.pushButton_open_correlation.setEnabled(enabled)
        if enabled:
            self.pushButton_run_milling.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
            self.pushButton_run_milling.setText("Run Milling")
            self.pushButton_add_milling_stage.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
            self.pushButton_remove_milling_stage.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
            self.pushButton_stop_milling.setVisible(False)
        elif milling:
            self.pushButton_stop_milling.setVisible(True)
            self.pushButton_pause_milling.setVisible(True)
            self.pushButton_run_milling.setText("Running...")
            self.pushButton_pause_milling.setText("Pause Milling")
            self.pushButton_run_milling.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_pause_milling.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        else:
            self.pushButton_run_milling.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_add_milling_stage.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_remove_milling_stage.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_stop_milling.setVisible(False)
            self.pushButton_pause_milling.setVisible(False)

        # disable the milling stage widgets while milling
        self.groupBox_aligment.setEnabled(not milling)
        self.groupBox_milling_settings.setEnabled(not milling)
        self.groupBox_pattern_settings.setEnabled(not milling)
        self.groupBox_strategy.setEnabled(not milling)
        self.groupBox_imaging.setEnabled(not milling)

        if caller is None:
            self.image_widget._toggle_interactions(enabled, caller="milling")
            self.parent.movement_widget._toggle_interactions(enabled, caller="milling")

    def update_ui(self):
        """Update the milling stages from the UI and try to redraw the patterns."""
        if not self.MILLING_STAGES_INITIALISED:
            return

        # force the milling hfw to match the current image hfw (TODO: do this more elegantly)
        self.doubleSpinBox_hfw.setValue(self.image_widget.doubleSpinBox_image_hfw.value())

        # get the selected milling stages
        self.update_milling_stage_from_ui()                 # update milling stage from ui
        milling_stages = self.get_selected_milling_stages() # get the selected milling stages from the ui

        # remove all patterns if there are no milling stages
        if not milling_stages:
            remove_all_napari_shapes_layers(self.viewer)
            return

        # make milling stage a list if it is not
        if not isinstance(milling_stages, list):
            milling_stages = [milling_stages]

        # update the estimated time
        self.update_estimated_time(milling_stages)

        # pprint(milling_stages[0].to_dict())
        # check hfw threshold # TODO: do this check more seriously, rather than rule of thumb
        try:
            draw_crosshair = self.checkBox_show_milling_crosshair.isChecked()
            if not isinstance(self.image_widget.ib_image, FibsemImage):
                raise Exception("No FIB Image, cannot draw patterns. Please take an image.")
            if self.image_widget.ib_layer is None:
                raise Exception("No FIB Image Layer, cannot draw patterns. Please take an image.")

            # TODO: upgrades. 
            # Don't redraw the patterns each time, only update the position if it has changed
            # Only draw the patterns that are visible 
            # add names and legends

            # clear patterns then draw new ones
            self.milling_pattern_layers = draw_milling_patterns_in_napari(
                viewer=self.viewer,
                image_layer=self.image_widget.ib_layer,
                milling_stages=milling_stages,
                pixelsize=self.image_widget.ib_image.metadata.pixel_size.x,
                draw_crosshair=draw_crosshair,
            )  

        except Exception as e:
            napari.utils.notifications.show_error(f"Error drawing patterns: {e}")
            logging.error(e)

        self.image_widget.restore_active_layer_for_movement()

    def toggle_pattern_visibility(self):
        """Toggle the visibility of the milling patterns layers."""
        is_visible = self.checkBox_show_milling_patterns.isChecked()
        for layer in self.milling_pattern_layers:
            if layer in self.viewer.layers:
                self.viewer.layers[layer].visible = is_visible


#### CORRELATION

    def open_3d_correlation_widget(self):
        """Open the correlation widget, connect relevant signals."""
        if not CORRELATION_THREEDCT_AVAILABLE:
            napari.utils.notifications.show_warning(
                "3D Correlation UI not available. Please install tdct."
            )
            return

        # attempt to close the windows
        try:
            if self.correlation_widget is not None:
                self.correlation_widget.close()
                self.correlation_widget = None
        except Exception as e:
            logging.warning(f"Error closing correlation widget: {e}")

        # snapshot existing layers
        self.existing_layers = [layer.name for layer in self.viewer.layers]
        for layer_name in self.existing_layers:
            self.viewer.layers[layer_name].visible = False

        self.correlation_widget = CorrelationUI(viewer=self.viewer, parent_ui=self)
        self.correlation_widget.continue_pressed_signal.connect(self.handle_correlation_continue_signal)

        # load fib image, if available
        fib_image: FibsemImage = self.image_widget.ib_image
        if fib_image is not None:
            md = fib_image.metadata.image_settings
            filename = os.path.join(md.path, md.filename)
            self.correlation_widget.set_project_path(str(md.path))
            self.correlation_widget.load_fib_image(image=median_filter(fib_image.data, size=3), 
                                                    pixel_size=fib_image.metadata.pixel_size.x, 
                                                    filename=filename)
        # else:
            # napari.utils.notifications.show_warning("No FIBSEM image available...")
            # self.correlation_widget.set_project_path(self.experiment.path)

        # create a button to run correlation
        self.viewer.window.add_dock_widget(
            self.correlation_widget,
            area="right",
            name="3DCT Correlation",
            tabify=True
        )

    def handle_correlation_continue_signal(self, data: dict):
        """Handle the correlation signal from the correlation widget."""
        logging.info(f"correlation-data: {data}")

        poi = data.get("poi", None)
        if poi is None:
            return
        point = Point(x=poi[0], y=poi[1])

        for milling_stage in self.milling_stages:
            # don't move fiducial patterns
            if isinstance(milling_stage.pattern, FiducialPattern):
                continue
            milling_stage.pattern.point = point
        logging.debug(f"Moved patterns to {point}")

        # update the ui element correpsonding to the point, if not fiducial
        if not isinstance(self.current_milling_stage.pattern, FiducialPattern):
            self.doubleSpinBox_centre_x.setValue(point.x * constants.SI_TO_MICRO)
            self.doubleSpinBox_centre_y.setValue(point.y * constants.SI_TO_MICRO) # THIS TRIGGERS AN UPDATE

        self.update_ui()
        self.milling_position_changed.emit()

        # restore layers visibility
        for layer_name in self.existing_layers:
            self.viewer.layers[layer_name].visible = True

        # remove tab widget, delete object
        self.viewer.window.remove_dock_widget(self.correlation_widget)
        self.correlation_widget = None
        self.existing_layers = []