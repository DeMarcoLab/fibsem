import copy
import logging
import os
from typing import Dict, List, Tuple, Any

import napari
import napari.utils.notifications
from napari.layers import Image as NapariImageLayer
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QMainWindow,
)

from fibsem import config as cfg
from fibsem import conversions, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (
    FibsemMillingStage,
    get_milling_stages,
    get_protocol_from_stages,
    get_strategy,
)
from fibsem.milling.patterning.patterns2 import (
    BasePattern,
    LinePattern,
    TrenchPattern,
)
from fibsem.milling.patterning import (
    MILLING_PATTERN_NAMES,
    get_pattern,
)
from fibsem.milling.strategy import (
    MillingStrategy,
    get_strategy_names,
)
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    Enum,
    FibsemImage,
    Point,
)
from fibsem.ui import stylesheets
from fibsem.ui.FibsemMillingWidget import WheelBlocker
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    is_pattern_placement_valid,
)
from fibsem.ui.napari.utilities import is_position_inside_layer
from fibsem.utils import format_value

MILLING_SETTINGS_GUI_CONFIG = {
    "patterning_mode": {
        "label": "Patterning Mode",
        "type": str,
        "items": ["Serial", "Parallel"],
        "tooltip": "The mode of patterning used for milling.",
    },
    "milling_current": {
        "label": "Milling Current",
        "units": "A",
        "tooltip": "The current used for milling.",
        "items": "dynamic",
    },
    "milling_voltage": {
        "label": "Milling Voltage",
        "units": "V",
        "items": "dynamic",
        "tooltip": "The voltage used for milling.",
    },
    "hfw": {
        "label": "Field of View",
        "type": float,
        "units": "µm",
        "scale": 1e6,
        "default": 150.0,
        "minimum": 20.0,
        "maximum": 950.0,
        "step": 10.0,
        "decimals": 2,
        "tooltip": "The horizontal field width (fov) for milling.",
    },
    "application_file": {
        "label": "Application File",
        "type": str,
        "items":  "dynamic",
        "tooltip": "The ThermoFisher application file for milling.",
    },
    "acquire_images": {
        "label": "Acquire After Milling",
        "type": bool,
        "default": True,
        "tooltip": "Acquire images after milling.",
    },
    "rate": {
        "label": "Milling Rate",
        "units": "mm3/s",
        "scale": 1e9,
        "tooltip": "The milling rate in mm³/s.",
    },
    "preset": {
        "label": "Milling Preset",
        "type": str,
        "items": "dynamic",
        "tooltip": "The preset for milling parameters.",
    },
    "dwell_time": {
        "label": "Dwell Time",
        "units": "us",
        "scale": 1e6,
        # "default": 0.1,
        # "minimum": 0.01,
        # "maximum": 10.0,
        # "step": 0.01,
        "decimals": 2,
        "tooltip": "The dwell time for each point in the milling pattern.",
    },
    "spot_size": {
        "label": "Spot Size",
        "type": float,
        "units": "um",
        "scale": 1e6,
        # "default": 10.0,
        # "minimum": 1.0,
        # "maximum": 100.0,
        # "step": 1.0,
        "decimals": 2,
        "tooltip": "The spot size for the ion beam during milling.",
    },
}

MILLING_PATTERN_GUI_CONFIG = {
    "width": {
        "label": "Width",
        "tooltip": "The width of the milling pattern.",
    },
    "height": {
        "label": "Height",
        "tooltip": "The height of the milling pattern.",
    },
    "depth": {
        "label": "Depth",
        "tooltip": "The depth of the milling pattern.",
    },
    "rotation": {
        "label": "Rotation",
        "type": float,
        "scale": None,
        "units": "°",
        "minimum": 0.0,
        "maximum": 360.0,
        "step": 1.0,
        "tooltip": "The rotation angle of the milling pattern.",
    },
    "time": {
        "label": "Time",
        "units": "s",
        "scale": None,
        "tooltip": "The time for which the milling pattern will be applied.",
    },
    "cross_section": {
        "label": "Cross Section",
        "type": CrossSectionPattern,
        "items": [cs for cs in CrossSectionPattern],
        "tooltip": "The type of cross section for the milling pattern.",
    },
    "scan_direction": {
        "label": "Scan Direction",
        "type": str,
        "items": "dynamic",
        "tooltip": "The scan direction for the milling pattern.",
    },
    "upper_trench_height": {
        "label": "Upper Trench Height",
        "tooltip": "The height of the upper trench in the milling pattern.",
    }, 
    "lower_trench_height": {
        "label": "Lower Trench Height",
        "tooltip": "The height of the lower trench in the milling pattern.",
    },
    "fillet": {
        "label": "Fillet",
        "tooltip": "The fillet radius for the milling pattern.",
    },
    "spacing": {
        "label": "Spacing",
        "tooltip": "The spacing between the trenches in the milling pattern.",
    },
    "side_width": {
        "label": "Side Width",
        "tooltip": "The width of the sides in the milling pattern.",
    },
    "passes": {
        "label": "Passes",
        "scale": None,
        "units": "",
        "tooltip": "The number of passes for the milling pattern.",
    },
    "n_rows": {
        "label": "Rows",
        "type": int,
        "units": "",
        "minimum": 1,
        "maximum": 100,
        "step": 1,
        "scale": None,
        "tooltip": "The number of rows in the array.",
    },
    "n_columns": {
        "label": "Columns",
        "type": int,
        "units": "",
        "minimum": 1,
        "maximum": 100,
        "step": 1,
        "scale": None,
        "tooltip": "The number of columns in the array.",
    },
}

MILLING_STRATEGY_GUI_CONFIG = {
    "overtilt": {
        "label": "Overtilt",
        "type": float,
        "units": "°",
        "scale": None,
        "minimum": 0.0,
        "maximum": 10,
        "step": 1.0,
        "decimals": 2,
        "tooltip": "The overtilt angle for the milling strategy.",
    },  
    "resolution": {
        "label": "Resolution",
        "type": List[int],
        "items": cfg.STANDARD_RESOLUTIONS_LIST,
        "tooltip": "The imaging resolution for the milling strategy.",}
    }

MILLING_ALIGNMENT_GUI_CONFIG = {
    "enabled": {
        "label": "Initial Alignment",
        "type": bool,
        "default": True,
        "tooltip": "Enable initial milling alignment between imaging and milling current.",
    },
}

MILLING_IMAGING_GUI_CONFIG = {
    "resolution": {
        "label": "Image Resolution",
        "type": List[int],
        "items": cfg.STANDARD_RESOLUTIONS_LIST,
        "tooltip": "The resolution for the acquired images.",
    },
    "hfw": {
        "label": "Horizontal Field Width",
        "units": "µm",
        "scale": 1e6,
        "tooltip": "The horizontal field width for the acquired images.",
    },
    "dwell_time": {
        "label": "Dwell Time",
        "units": "µs",
        "scale": 1e6,
        "tooltip": "The dwell time for each pixel in the acquired images.",
    },
    "autocontrast": {
        "label": "Autocontrast",
        "type": bool,
        "default": True,
        "tooltip": "Enable autocontrast for the acquired images.",
    },
}

DEFAULT_PARAMETERS: Dict[str, any] = {
    "type": float,
    "units": "µm",
    "scale": 1e6,
    "minimum": 0.0,
    "maximum": 1000.0,
    "step": 0.01,
    "decimals": 2,
    "tooltip": "Default parameter for milling settings.",
}

GUI_CONFIG: Dict[str, Dict] = {
    "milling": MILLING_SETTINGS_GUI_CONFIG,
    "pattern": MILLING_PATTERN_GUI_CONFIG,
    "strategy": MILLING_STRATEGY_GUI_CONFIG,
    "alignment": MILLING_ALIGNMENT_GUI_CONFIG,
    "imaging": MILLING_IMAGING_GUI_CONFIG,}

# mapping from milling settings to microscope parameters
PARAMETER_MAPPING = {
    "milling_current": "current",
    "milling_voltage": "voltage",
}




# MILLING_STAGE_1:
    # MILLING_SETTINGS
    # MILLING_PATTERN
    # MILLING_STRATEGY
    # MILLING_ALIGNMENT
    # MILLING_ACQUISITION
# MILLING_STAGE_2:
#    ...

# TODO: 
# what to do when no microscope available???
# milling stage name?

# DONE:
# point controls for pattern
# when we change selection of pattern or strategy, need to also update milling_stage object
# parameter return
# strategy selection
# multi-stages
# advanced settings display

def pretty_name(milling_stage: FibsemMillingStage) -> str:
    milling_current = milling_stage.milling.milling_current
    mc = format_value(val=milling_current, unit="A", precision=1)
    txt = f"{milling_stage.name} - {milling_stage.pattern.name} ({mc})"
    return txt

class FibsemMillingStageWidget(QWidget):
    _milling_stage_changed = pyqtSignal(FibsemMillingStage)

    def __init__(self, 
                 microscope: FibsemMicroscope, # TODO: don't require this!, but if its available, use it for dynamic items 
                 milling_stage: FibsemMillingStage, 
                 manufacturer: str = "TFS", 
                 parent=None):
        super().__init__(parent)

        self.parameters: Dict[str, Dict[str, Tuple[QLabel, QWidget, float]]] = {} # param: label, control, scale
        self.microscope = microscope
        self._milling_stage = milling_stage
        self._manufacturer = manufacturer  # Manufacturer for dynamic items (TFS, TESCAN)

        self._create_widgets()
        self._initialise_widgets()

    def _create_widgets(self):
        """Create the main widgets for the milling stage editor."""
        self.milling_widget = QGroupBox(self)
        self.milling_widget.setTitle("Settings")
        self.milling_widget.setObjectName("widget-milling-settings")
        self.milling_widget.setLayout(QGridLayout())

        self.pattern_widget = QGroupBox(self)
        self.pattern_widget.setTitle("Pattern")
        self.pattern_widget.setObjectName("widget-milling-pattern")
        self.pattern_widget.setLayout(QGridLayout())

        self.alignment_widget = QGroupBox(self)
        self.alignment_widget.setTitle("Alignment")
        self.alignment_widget.setObjectName("widget-milling-alignment")
        self.alignment_widget.setLayout(QGridLayout())
        
        self.strategy_widget = QGroupBox(self)
        self.strategy_widget.setTitle("Strategy")
        self.strategy_widget.setObjectName("widget-milling-strategy")
        self.strategy_widget.setLayout(QGridLayout())

        self.acquisition_widget = QGroupBox(self)
        self.acquisition_widget.setTitle("Acquisition")
        self.acquisition_widget.setObjectName("widget-milling-acquisition")
        self.acquisition_widget.setLayout(QGridLayout())

        # create label and combobox
        label = QLabel(self)
        label.setText("Name")
        self.comboBox_selected_pattern = QComboBox(self)
        self.comboBox_selected_pattern.addItems(MILLING_PATTERN_NAMES)
        self.wheel_blocker1 = WheelBlocker()
        self.comboBox_selected_pattern.installEventFilter(self.wheel_blocker1)
        self.comboBox_selected_pattern.currentTextChanged.connect(self._on_pattern_changed)
        self.pattern_widget.layout().addWidget(label, 0, 0, 1, 1)
        self.pattern_widget.layout().addWidget(self.comboBox_selected_pattern, 0, 1, 1, 1)

        # create strategy widget
        label = QLabel(self)
        label.setText("Name")
        self.comboBox_selected_strategy = QComboBox(self)
        self.strategy_widget.layout().addWidget(label, 0, 0, 1, 1)
        self.strategy_widget.layout().addWidget(self.comboBox_selected_strategy, 0, 1, 1, 1)

        self.comboBox_selected_strategy.addItems(get_strategy_names())
        self.wheel_blocker2 = WheelBlocker()
        self.comboBox_selected_strategy.installEventFilter(self.wheel_blocker2)
        self.comboBox_selected_strategy.currentTextChanged.connect(self._on_strategy_changed)

        # Create the widgets list to hold all the widgets
        self._widgets = [
            self.milling_widget,
            self.pattern_widget,
            self.alignment_widget,
            self.strategy_widget,
            self.acquisition_widget
        ]

        # Add the widgets to the main layout
        self.gridlayout = QGridLayout(self)
        label = QLabel(self)
        label.setText("Milling Stage:")
        label.setObjectName("label-milling-stage-name")
        self.lineEdit_milling_stage_name = QLineEdit(self)
        self.lineEdit_milling_stage_name.setText(self._milling_stage.name)
        self.lineEdit_milling_stage_name.setObjectName("lineEdit-name-stage")
        self.lineEdit_milling_stage_name.setToolTip("The name of the milling stage.")
        self.lineEdit_milling_stage_name.editingFinished.connect(self._update_setting)
        self.gridlayout.addWidget(label, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.lineEdit_milling_stage_name, 0, 1, 1, 1)
        for widget in self._widgets:
            self.gridlayout.addWidget(widget, self.gridlayout.rowCount(), 0, 1, 2)

    def _initialise_widgets(self):
        """Initialise the widgets with the current milling stage settings."""
        # MILLING SETTINGS
        milling_parames = self._milling_stage.milling.get_parameters(self._manufacturer)
        self._create_controls(self.milling_widget, milling_parames, "milling", GUI_CONFIG["milling"].copy())

        # PATTERN
        self.comboBox_selected_pattern.blockSignals(True)
        self.comboBox_selected_pattern.setCurrentText(self._milling_stage.pattern.name)
        self.comboBox_selected_pattern.blockSignals(False)
        self._update_pattern_widget(self._milling_stage.pattern)  # Set default pattern

        # ALIGNMENT
        alignment = self._milling_stage.alignment
        alignment_params = {"enabled": alignment.enabled}
        self._create_controls(self.alignment_widget, alignment_params, "alignment", GUI_CONFIG["alignment"].copy())

        # STRATEGY
        self.comboBox_selected_strategy.blockSignals(True)
        self.comboBox_selected_strategy.setCurrentText(self._milling_stage.strategy.name)
        self.comboBox_selected_strategy.blockSignals(False)
        self._update_strategy_widget(self._milling_stage.strategy)  # Set default strategy

        # IMAGING
        image_settings = self._milling_stage.imaging
        imaging_params = {"resolution": image_settings.resolution,
                            "hfw": image_settings.hfw,
                            "dwell_time": image_settings.dwell_time,
                            "autocontrast": image_settings.autocontrast,
                            "save": image_settings.save,}

        self._create_controls(self.acquisition_widget, imaging_params, "imaging", GUI_CONFIG["imaging"].copy())

        # special case for hiding acquisition controls
        control: QCheckBox = self.parameters["milling"]["acquire_images"][1]
        control.toggled.connect(lambda checked: self.acquisition_widget.setVisible(checked))
        self.acquisition_widget.setVisible(control.isChecked())

    def toggle_advanced_settings(self, show: bool):
        """Toggle the visibility of advanced settings."""
        ms = self._milling_stage
        wp = self.parameters
        for param in ms.pattern.advanced_attributes:

            label, control, _ = wp["pattern"].get(param, (None, None, None))
            if label:
                label.setVisible(show)
            if control:
                control.setVisible(show)
        for param in ms.strategy.config.advanced_attributes:
            label, control, _ = wp["strategy.config"].get(param, (None, None, None))
            if label:
                label.setVisible(show)
            if control:
                control.setVisible(show)
        for param in ms.milling.advanced_attributes:
            label, control, _ = wp["milling"].get(param, (None, None, None))
            if label:
                label.setVisible(show)
            if control:
                control.setVisible(show)
        # QUERY: should we also hide imaging, alignment?
        # consider strategy as advanced, so hide it as well
        self.strategy_widget.setVisible(show)

    def clear_widget(self, widget: QWidget, row_threshold: int = -1):
        """Clear the widget's layout, removing all items below a certain row threshold."""

        items_to_remove = []
        grid_layout: QGridLayout = widget.layout()

        # iterate through the items in the grid layout
        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if item is not None:
                row, col, rowspan, colspan = grid_layout.getItemPosition(i)
                if row > row_threshold:
                    items_to_remove.append(item)

        # Remove the items
        for item in items_to_remove:
            grid_layout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()

    def _on_pattern_changed(self, pattern_name: str):
        # TODO: convert the comboBox_selected_pattern to use currentData, 
        # that way we can pass the pattern object directly (and restore it from the previous state)
        pattern = get_pattern(pattern_name)

        self._milling_stage.pattern = pattern  # Update the milling stage's strategy

        self._update_pattern_widget(pattern)
        self._milling_stage_changed.emit(self._milling_stage)  # Emit signal to notify changes

    def _update_pattern_widget(self, pattern: BasePattern):
        """Update the pattern widget with the selected pattern's parameters."""

        params = {k: getattr(pattern, k) for k in pattern.required_attributes if hasattr(pattern, k)}
        params["point"] = pattern.point  # add point as a special case

        self._create_controls(self.pattern_widget, params, "pattern", GUI_CONFIG["pattern"].copy())

    def _on_strategy_changed(self, strategy_name: str):
        """Update the strategy widget with the selected strategy's parameters."""
        strategy = get_strategy(strategy_name, {"config": {}})

        # update strategy and widget
        self._milling_stage.strategy = strategy
        self._update_strategy_widget(strategy)
        self._milling_stage_changed.emit(self._milling_stage)  # Emit signal to notify changes
        
    def _update_strategy_widget(self, strategy: MillingStrategy[Any]):
        """Update the strategy widget with the selected strategy's parameters."""
        params = {k: getattr(strategy.config, k) for k in strategy.config.required_attributes}

        self._create_controls(self.strategy_widget, params, "strategy.config", GUI_CONFIG["strategy"].copy())

    def _create_controls(self, widget: QWidget, params: Dict[str, any], cls: str, config: Dict[str, any]):
        """Create controls for the given parameters and add them to the widget."""

        # clear previous controls
        if cls == "pattern":
            self.clear_widget(self.pattern_widget, row_threshold=0)
        if cls == "strategy.config":
            self.clear_widget(self.strategy_widget, row_threshold=0)

        self.parameters[cls] = {}
        grid_layout: QGridLayout = widget.layout()

        # point controls (special case). but why do they have to be?
        if cls == "pattern":
            gui_config = config.get("point", {})
            label_text = gui_config.get("label", "Point")
            minimum = gui_config.get("minimum", DEFAULT_PARAMETERS["minimum"])
            maximum = gui_config.get("maximum", DEFAULT_PARAMETERS["maximum"])
            step_size   = gui_config.get("step", DEFAULT_PARAMETERS["step"])
            units = gui_config.get("units", DEFAULT_PARAMETERS["units"])
            scale = gui_config.get("scale", DEFAULT_PARAMETERS["scale"])
            decimals = gui_config.get("decimals", DEFAULT_PARAMETERS["decimals"])

            # points are a special case? 
            pt_label = QLabel(self)
            pt_label.setText(label_text)
            pt_label.setObjectName(f"label-{cls}-point")
            pt_label.setToolTip(gui_config.get("tooltip", "Point coordinates for the milling pattern."))

            hbox_layout = QHBoxLayout()
            for attr in ["x", "y"]:
                # create double spin boxes for point coordinates
                control = QDoubleSpinBox(self)
                control.setSuffix(f" {units}")
                control.setRange(-1000, 1000)
                control.setSingleStep(step_size)
                value = getattr(params["point"], attr)
                if scale is not None:
                    value *= scale
                control.setValue(value)
                control.setObjectName(f"control-pattern-point.{attr}")
                control.setKeyboardTracking(False)
                control.setDecimals(decimals)
                control.valueChanged.connect(self._update_setting)

                self.parameters[cls][f"point.{attr}"] = (pt_label, control, scale)
                hbox_layout.addWidget(control)

            # add both point controls to widget, set the padding to 0 to match other visual
            point_widget = QWidget(self)
            point_widget.setObjectName(f"point-widget-{cls}")
            point_widget.setToolTip(gui_config.get("tooltip", "Point coordinates for the milling pattern."))
            hbox_layout.setContentsMargins(0, 0, 0, 0)
            point_widget.setContentsMargins(0, 0, 0, 0)
            point_widget.setLayout(hbox_layout)

            # add to the grid layout
            row = widget.layout().rowCount()
            grid_layout.addWidget(pt_label, row, 0, 1, 1)
            grid_layout.addWidget(point_widget, row, 1, 1, 1)

        for name, value in params.items():

            # get the GUI configuration for the parameter
            gui_config = config.get(name, {})
            label_text = gui_config.get("label", name.replace("_", " ").title())
            scale = gui_config.get("scale", DEFAULT_PARAMETERS["scale"])
            units = gui_config.get("units", DEFAULT_PARAMETERS["units"])
            minimum = gui_config.get("minimum", DEFAULT_PARAMETERS["minimum"])
            maximum = gui_config.get("maximum", DEFAULT_PARAMETERS["maximum"])
            step_size = gui_config.get("step", DEFAULT_PARAMETERS["step"])
            decimals = gui_config.get("decimals", DEFAULT_PARAMETERS["decimals"])
            items = gui_config.get("items", [])

            # set label text
            label = QLabel(label_text)

            # add combobox controls
            if items:
                if items == "dynamic":
                    items = self.microscope.get_available_values(PARAMETER_MAPPING.get(name, name), BeamType.ION)

                control = QComboBox()
                for item in items:
                    if isinstance(item, (float, int)):
                        item_str = format_value(val=item,
                                                unit=units,
                                                precision=gui_config.get("decimals", 1))
                    elif isinstance(item, Enum):
                        item_str = item.name
                    elif "resolution" in name:
                        item_str = f"{item[0]}x{item[1]}"
                    else:
                        item_str = str(item)
                    control.addItem(item_str, item)

                if isinstance(value, tuple) and len(value) == 2:
                    value = list(value)  # Convert tuple to list for easier handling

                # find the closest match to the current value (should only be used for numerical values)
                idx = control.findData(value)
                if idx == -1:
                    # get the closest value
                    closest_value = min(items, key=lambda x: abs(x - value))
                    idx = control.findData(closest_value)
                if idx == -1:
                    logging.debug(f"Warning: No matching item or nearest found for {name} with value {value}. Using first item.")
                    idx = 0
                control.setCurrentIndex(idx)

            # add line edit controls
            elif isinstance(value, str):
                control = QLineEdit()
                control.setText(value)
            # add checkbox controls
            elif isinstance(value, bool):
                control = QCheckBox()
                control.setChecked(value)
            elif isinstance(value, (float, int)):

                control = QDoubleSpinBox()
                if units is not None:
                    control.setSuffix(f' {units}')
                if scale is not None:
                    value = value * scale
                if minimum is not None:
                    control.setMinimum(minimum)
                if maximum is not None:
                    control.setMaximum(maximum)
                if step_size is not None:
                    control.setSingleStep(step_size)
                if decimals is not None:
                    control.setDecimals(decimals)
                control.setValue(value)
                control.setKeyboardTracking(False)
            else:
                continue

            # Set tooltip for both label and control
            if tooltip := gui_config.get("tooltip"):
                label.setToolTip(tooltip)
                control.setToolTip(tooltip)

            grid_layout.addWidget(label, grid_layout.rowCount(), 0)
            grid_layout.addWidget(control, grid_layout.rowCount() - 1, 1)

            label.setObjectName(f"label-{cls}-{name}")
            control.setObjectName(f"control-{cls}-{name}")
            self.parameters[cls][name] = (label, control, scale)

            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._update_setting)
            elif isinstance(control, QLineEdit):
                control.textChanged.connect(self._update_setting)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._update_setting)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(self._update_setting)

    # add callback to update settings when control value changes
    def _update_setting(self):
        obj = self.sender()
        if not obj:
            return
        obj_name = obj.objectName()
        _, cls, name = obj_name.split("-", 2)

        if isinstance(obj, QComboBox):
            value = obj.currentData()
        elif isinstance(obj, QLineEdit):
            value = obj.text()
        elif isinstance(obj, QCheckBox):
            value = obj.isChecked()
        elif isinstance(obj, (QSpinBox, QDoubleSpinBox)):
            value = obj.value()
            # apply scale if defined
            scale = self.parameters[cls][name][2]
            if scale is not None:
                value /= scale
        else:
            return

        # update the milling_stage object
        if hasattr(self._milling_stage, cls):

            # special case for pattern point
            if "point" in name:
                if "x" in name:
                    setattr(self._milling_stage.pattern.point, "x", value)
                elif "y" in name:
                    setattr(self._milling_stage.pattern.point, "y", value)
            elif cls == "name":
                setattr(self._milling_stage, "name", value)
            else:
                setattr(getattr(self._milling_stage, cls), name, value)
        elif hasattr(self._milling_stage, "strategy") and cls == "strategy.config":
            # Special case for strategy config
            setattr(self._milling_stage.strategy.config, name, value)
        else:
            logging.debug(f"Warning: {cls} not found in milling_stage object. Cannot update {name}.")

        self._milling_stage_changed.emit(self._milling_stage)  # notify changes

    def get_milling_stage(self) -> FibsemMillingStage:
        return self._milling_stage

    def set_point(self, point: Point) -> None:
        """Set the point for the milling pattern."""

        # Update the point controls
        control: QDoubleSpinBox
        for attr in ["x", "y"]:
            label, control, scale = self.parameters["pattern"][f"point.{attr}"]
            value = getattr(point, attr) * scale
            control.setValue(value)

class FibsemMillingStageEditorWidget(QWidget):
    _milling_stages_updated = pyqtSignal(list)
    """A widget to edit the milling stage settings."""

    def __init__(self,
                 viewer: napari.Viewer,
                 microscope: FibsemMicroscope,
                 milling_stages: List[FibsemMillingStage],
                 parent=None):
        super().__init__(parent)

        self.microscope = microscope
        self._milling_stages = milling_stages
        self._background_milling_stages: List[FibsemMillingStage] = []
        self.is_updating_pattern = False

        self.viewer = viewer
        self.image: FibsemImage = FibsemImage.generate_blank_image(hfw=80e-6)
        self.image_layer: NapariImageLayer = self.viewer.add_image(data=self.image.data, name="FIB Image")
        self._widgets: List[FibsemMillingStageWidget] = []

        # widget controls
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # required to resize
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # add widget for scroll content
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        # add a list widget to hold the milling stages, with re-ordering support
        self.list_widget_milling_stages = QListWidget(self)
        self.list_widget_milling_stages.setDragDropMode(QListWidget.InternalMove)
        self.list_widget_milling_stages.setDefaultDropAction(Qt.MoveAction)
        self.list_widget_milling_stages.setMaximumHeight(100)
        model = self.list_widget_milling_stages.model()
        model.rowsMoved.connect(self._reorder_milling_stages)

        # add milling widgets for each milling stage
        for milling_stage in self._milling_stages:
            self._add_milling_stage_widget(milling_stage)

        # add/remove buttons for milling stages
        self.pushButton_add = QPushButton("Add Milling Stage", self)
        self.pushButton_add.clicked.connect(lambda: self._add_milling_stage(None))
        self.pushButton_add.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_remove = QPushButton("Remove Selected Stage", self)
        self.pushButton_remove.clicked.connect(self._remove_selected_milling_stage)
        self.pushButton_remove.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pushButton_add)
        button_layout.addWidget(self.pushButton_remove)

        # clear/set milling stages buttons (for development mode)
        DEVELOPMENT_MODE = False
        if DEVELOPMENT_MODE:
            self._tmp_milling_stages = copy.deepcopy(milling_stages)
            pushButton_add = QPushButton("Set Milling Stages", self)
            pushButton_add.clicked.connect(lambda: self.set_milling_stages(self._tmp_milling_stages))
            pushButton_add.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
            pushButton_remove = QPushButton("Clear Milling Stages", self)
            pushButton_remove.clicked.connect(self.clear_milling_stages)
            pushButton_remove.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)

            pushButton_show = QPushButton("Show Milling Stages", self)
            pushButton_show.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
            pushButton_show.clicked.connect(self.update_milling_stage_display)

            button_layout2 = QHBoxLayout()
            button_layout2.addWidget(pushButton_add)
            button_layout2.addWidget(pushButton_remove)

        # add checkboxes for show advanced settings, show milling crosshair, show milling patterns
        self.checkBox_show_advanced_settings = QCheckBox("Show Advanced Settings", self)
        self.checkBox_show_advanced_settings.setChecked(False)
        self.checkBox_show_advanced_settings.setToolTip("Show advanced settings for milling stages.")
        self.checkBox_show_milling_crosshair = QCheckBox("Show Milling Crosshair", self)
        self.checkBox_show_milling_crosshair.setChecked(True)
        self.checkBox_show_milling_crosshair.setToolTip("Show the milling crosshair in the viewer.")
        self.checkBox_show_milling_patterns = QCheckBox("Show Milling Patterns", self)
        self.checkBox_show_milling_patterns.setChecked(True)
        self.checkBox_show_milling_patterns.setToolTip("Show the milling patterns in the viewer.")
        
        # callbacks for checkboxes
        self.checkBox_show_advanced_settings.stateChanged.connect(self._toggle_advanced_settings)
        self.checkBox_show_milling_crosshair.stateChanged.connect(self.update_milling_stage_display)
        self.checkBox_show_milling_patterns.stateChanged.connect(self._toggle_pattern_visibility)
        
        # grid layout for checkboxes
        self._grid_layout_checkboxes = QGridLayout()
        self._grid_layout_checkboxes.addWidget(self.checkBox_show_advanced_settings, 0, 0, 1, 1)
        self._grid_layout_checkboxes.addWidget(self.checkBox_show_milling_crosshair, 0, 1, 1, 1)
        self._grid_layout_checkboxes.addWidget(self.checkBox_show_milling_patterns, 1, 0, 1, 1)
        # not adding non-relative movement checkbox for now

        # add widgets to main widget/layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addLayout(button_layout)
        if DEVELOPMENT_MODE:
            self.main_layout.addLayout(button_layout2)
            self.main_layout.addWidget(pushButton_show)
        self.main_layout.addLayout(self._grid_layout_checkboxes)
        self.main_layout.addWidget(self.list_widget_milling_stages)

        # connect signals
        self.list_widget_milling_stages.itemSelectionChanged.connect(self._on_selected_stage_changed)
        self.list_widget_milling_stages.itemChanged.connect(self.update_milling_stage_display)
        self.list_widget_milling_stages.itemChanged.connect(self._on_milling_stage_updated)
        self.viewer.mouse_drag_callbacks.append(self._on_single_click)

        # set initial selection to the first item
        if self.list_widget_milling_stages.count() > 0:
            self.list_widget_milling_stages.setCurrentRow(0)

        self._toggle_advanced_settings(self.checkBox_show_advanced_settings.checkState())

    def _toggle_advanced_settings(self, state: int):
        """Toggle the visibility of advanced settings in the milling stage editor."""

        show_advanced = bool(state == Qt.Checked)

        for widget in self._widgets:
            widget.toggle_advanced_settings(show_advanced)

    def _toggle_pattern_visibility(self, state: int):
        """Toggle the visibility of milling patterns in the viewer."""
        visible = bool(state == Qt.Checked)
        if self.milling_pattern_layers:
            for layer in self.milling_pattern_layers:
                if layer in self.viewer.layers:
                    self.viewer.layers[layer].visible = visible

    def _reorder_milling_stages(self, parent, start, end, destination, row):
        """Sync the object list when UI is reordered"""
        logging.info(f"Reordering milling stages: start={start}, end={end}, destination={destination}, row={row}")        
        
        # get
        dest_index = row if row < start else row - (end - start + 1)
        
        # Move objects in the list
        objects_to_move = self._milling_stages[start:end+1]
        del self._milling_stages[start:end+1]
        
        for i, obj in enumerate(objects_to_move):
            self._milling_stages.insert(dest_index + i, obj)
        
        logging.info(f"Objects reordered: {[obj.name for obj in self._milling_stages]}")

        # when we re-order, we need to re-order the widgets as well
        dest_widgets = self._widgets[start:end+1]
        del self._widgets[start:end+1]
        for i, widget in enumerate(dest_widgets):
            self._widgets.insert(dest_index + i, widget)

        self.update_milling_stage_display()
        self._on_milling_stage_updated()

    def _remove_selected_milling_stage(self):
        """Remove the selected milling stage from the list widget."""
        selected_items = self.list_widget_milling_stages.selectedItems()
        if not selected_items:
            logging.info("No milling stage selected for removal.")
            return

        for item in selected_items:
            index = self.list_widget_milling_stages.row(item)
            self.list_widget_milling_stages.takeItem(index)
            # also remove the corresponding widget
            if index < len(self._widgets):
                widget = self._widgets.pop(index)
                widget.deleteLater()
            
            self._milling_stages.pop(index)  # Remove from the milling stages list
            logging.info(f"Removed item: {item.text()} at index {index}")

        self._on_milling_stage_updated()
        self.update_milling_stage_display()

    def clear_milling_stages(self):
        """Clear all milling stages from the editor."""
        self._milling_stages.clear()
        self.list_widget_milling_stages.clear()

        # clear previous widgets
        for widget in self._widgets:
            widget.deleteLater()
        self._widgets.clear()
        
        self.update_milling_stage_display()

    def set_milling_stages(self, milling_stages: List[FibsemMillingStage]):
        """Set the milling stages to be displayed in the editor."""

        self.clear_milling_stages()  # Clear existing milling stages
        self._milling_stages = copy.deepcopy(milling_stages)
        for milling_stage in self._milling_stages:
            self._add_milling_stage_widget(milling_stage)

        # select the first milling stage if available
        if self._milling_stages:
            self.list_widget_milling_stages.setCurrentRow(0)
    
    def set_background_milling_stages(self, milling_stages: List[FibsemMillingStage]):
        """Set the background milling stages to be displayed in the editor."""
        self._background_milling_stages = copy.deepcopy(milling_stages)

    def _update_list_widget_text(self):
        """Update the text of the list widget items to reflect the current milling stages."""
        for i, milling_stage in enumerate(self._milling_stages):
            if i < self.list_widget_milling_stages.count():
                item = self.list_widget_milling_stages.item(i)
                # update the text of the item
                if item:
                    item.setText(pretty_name(milling_stage))

    def _add_milling_stage(self, milling_stage: FibsemMillingStage = None):
        """Add a new milling stage to the editor."""
        
        # create a default milling stage if not provided
        if milling_stage is None:
            num = len(self._milling_stages) + 1
            milling_stage = FibsemMillingStage(name=f"Milling Stage {num}", num=num)

        # Create a new widget for the milling stage
        logging.info(f"Added new milling stage: {milling_stage.name}")
        self._milling_stages.append(milling_stage)  # Add to the milling stages list

        self._add_milling_stage_widget(milling_stage)

        self.list_widget_milling_stages.setCurrentRow(self.list_widget_milling_stages.count()-1)
        self._on_milling_stage_updated()

    def _add_milling_stage_widget(self, milling_stage: FibsemMillingStage):
        """Add a milling stage widget to the editor."""

        # create milling stage widget, connect signals
        ms_widget = FibsemMillingStageWidget(microscope=self.microscope, 
                                            milling_stage=milling_stage)
        ms_widget._milling_stage_changed.connect(self.update_milling_stage_display)
        ms_widget._milling_stage_changed.connect(self._on_milling_stage_updated)
        ms_widget._milling_stage_changed.connect(self._update_list_widget_text)

        # create related list widget item
        # TODO: migrate to setData, so we can store the milling stage object directly
        item = QListWidgetItem(pretty_name(milling_stage))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.list_widget_milling_stages.addItem(item)

        # add the widgets
        self.scroll_layout.addWidget(ms_widget)
        self._widgets.append(ms_widget)

    def _on_selected_stage_changed(self):
        """Handle the selection change in the list widget."""
        selected_items = self.list_widget_milling_stages.selectedItems()
        if not selected_items:
            # hide all widgets
            for widget in self._widgets:
                widget.hide()

        # hide all widgets, except selected (only single-selection supported)
        index = self.list_widget_milling_stages.currentRow()
        for i, widget in enumerate(self._widgets):
            widget.setVisible(i==index)
        
        show_adv = self.checkBox_show_advanced_settings.checkState()
        self._widgets[index].toggle_advanced_settings(show_adv)

        # refresh display
        self.update_milling_stage_display()

    def _get_selected_milling_stages(self) -> List[FibsemMillingStage]:
        """Return the milling stages that are selected (checked) in the list widget."""
        checked_indexes = []
        for i in range(self.list_widget_milling_stages.count()):
            item = self.list_widget_milling_stages.item(i)
            if item.checkState() == Qt.Checked:
                checked_indexes.append(i)

        milling_stages = [
            widget.get_milling_stage()
            for i, widget in enumerate(self._widgets)
            if i in checked_indexes
        ]
        return milling_stages

    def get_milling_stages(self) -> List[FibsemMillingStage]:
        """Public method to get the currently selected milling stages."""
        return self._get_selected_milling_stages()

    def update_milling_stage_display(self):
        """Update the display of milling stages in the viewer."""
        if self.is_updating_pattern:
            return # block updates while updating patterns

        milling_stages = self.get_milling_stages()

        if not milling_stages:
            try:
                for layer in self.milling_pattern_layers:
                    if layer in self.viewer.layers:
                        self.viewer.layers.remove(layer)
            except Exception as e:
                logging.debug(f"Error removing milling pattern layers: {e}")
            self.milling_pattern_layers = []
            return

        logging.info(f"Selected milling stages:, {[stage.name for stage in milling_stages]}")

        if self.image is None:
            image = FibsemImage.generate_blank_image(hfw=milling_stages[0].milling.hfw)
            self.set_image(image)

        self.milling_pattern_layers = draw_milling_patterns_in_napari(
            viewer=self.viewer,
            image_layer=self.image_layer,
            milling_stages=milling_stages,
            pixelsize=self.image.metadata.pixel_size.x,
            draw_crosshair=self.checkBox_show_milling_crosshair.isChecked(),
            background_milling_stages=self._background_milling_stages,
        )

    def set_image(self, image: FibsemImage) -> None:
        """Set the image for the milling stage editor."""

        self.image = image
        try:
            self.image_layer.data = image.data
        except Exception as e:
            self.image_layer = self.viewer.add_image(name="FIB Image",
                                                     data=image.data,
                                                     opacity=0.7)
        self.update_milling_stage_display()

    def _on_single_click(self, viewer: napari.Viewer, event):
        
        if event.button != 1 or 'Shift' not in event.modifiers or self._milling_stages == []:
            return
        
        if not self.image_layer:
            logging.warning("No target layer found for the click event.")
            return

        if not is_position_inside_layer(event.position, self.image_layer):
            logging.warning("Click position is outside the image layer.")
            return

        current_idx = self.list_widget_milling_stages.currentRow()

        if current_idx < 0 or current_idx >= len(self._milling_stages):
            logging.warning("No milling stage selected or index out of range.")
            return

        # convert from image coordinates to microscope coordinates
        coords = self.image_layer.world_to_data(event.position)
        point_clicked = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=coords[1], y=coords[0]), # yx required
            image=self.image.data,
            pixelsize=self.image.metadata.pixel_size.x,
        )

        # conditions to move:
        #   all moved patterns are within the fib image
        new_points: List[Point] = []
        has_valid_patterns: bool = True
        is_moving_all_patterns: bool = bool('Control' in event.modifiers)
        use_relative_movement: bool = True

        # calculate the difference between the clicked point and the current pattern point (used for relative movement)
        diff = point_clicked - self._milling_stages[current_idx].pattern.point

        # loop to check through all patterns to see if they are in bounds
        for idx, milling_stage in enumerate(self._milling_stages):
            if not is_moving_all_patterns:
                if idx != current_idx:
                    continue

            pattern_renew = copy.deepcopy(milling_stage.pattern)

            # special case: if the pattern is a line, we also need to update start_x, start_y, end_x, end_y to move with the click
            if isinstance(pattern_renew, LinePattern):
                pattern_renew.start_x += diff.x
                pattern_renew.start_y += diff.y
                pattern_renew.end_x += diff.x
                pattern_renew.end_y += diff.y

                # TODO: resolve line special cases
                # this doesnt work if the line is rotated at all
                # if the line goes out of bounds it is not reset correctly, need to fix this

            # update the pattern point
            point = pattern_renew.point + diff if use_relative_movement else point_clicked
            pattern_renew.point = point
            
            # test if the pattern is within the image bounds
            if not is_pattern_placement_valid(pattern=pattern_renew, image=self.image):
                has_valid_patterns = False
                msg = f"{milling_stage.name} pattern is not within the FIB image."
                logging.warning(msg)
                napari.utils.notifications.show_warning(msg)
                break
            # otherwise, add the new point to the list
            new_points.append(copy.deepcopy(point))

        if has_valid_patterns:
    
            # block redraw until all patterns are updated
            self.is_updating_pattern = True
            if is_moving_all_patterns:
                for idx, new_point in enumerate(new_points):
                    self._widgets[idx].set_point(new_point)
            else: # only moving selected pattern
                self._widgets[current_idx].set_point(point_clicked)

        self.is_updating_pattern = False
        self._on_milling_stage_updated()
        self.update_milling_stage_display()  # force refresh the milling stages display

    def _on_milling_stage_updated(self, milling_stage: FibsemMillingStage = None):
        """Callback when a milling stage is updated."""

        # If we are currently updating the pattern, we don't want to emit the signal
        if self.is_updating_pattern:
            return

        milling_stages = self.get_milling_stages()
        self._milling_stages_updated.emit(milling_stages)


from autolamella.structures import AutoLamellaProtocol, Experiment, Lamella
# TODO: move to autolamella
class AutoLamellaProtocolEditorWidget(QWidget):
    """A widget to edit the AutoLamella protocol."""
    
    def __init__(self, 
                 viewer: napari.Viewer,
                 microscope: FibsemMicroscope,
                 protocol: AutoLamellaProtocol,
                 experiment: Experiment = None,
                 parent=None):
        super().__init__(parent)
        self.parent = parent
        self.viewer = viewer
        self.microscope = microscope
        self.experiment = experiment
        self.protocol = protocol
        self.background_milling_stages: List[FibsemMillingStage] = []

        self.setWindowTitle("AutoLamella Protocol Editor")
        self._create_widgets()
        self._initialise_widgets()

    def _create_widgets(self):
        """Create the widgets for the protocol editor."""
        self.milling_stage_editor = FibsemMillingStageEditorWidget(viewer=self.viewer, 
                                                            microscope=self.microscope, 
                                                            milling_stages=[],
                                                            parent=self)
        # lamella, milling controls
        self.label_selected_lamella = QLabel("Lamella")
        self.comboBox_selected_lamella = QComboBox()
        self.label_selected_milling = QLabel("Milling Stage")
        self.comboBox_selected_milling = QComboBox()
        self.checkbox_sync_positions = QCheckBox("Sync Trench Position for Rough Milling and Polishing")
        self.checkbox_sync_positions.setToolTip("If checked, the trench position will be synchronized for rough milling and polishing stages.")
        self.checkbox_sync_positions.setObjectName("checkbox-sync-positions")
        self.checkbox_sync_positions.setChecked(True)
        
        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.label_selected_lamella, 0, 0)
        self.grid_layout.addWidget(self.comboBox_selected_lamella, 0, 1)
        self.grid_layout.addWidget(self.label_selected_milling, 1, 0)
        self.grid_layout.addWidget(self.comboBox_selected_milling, 1, 1)
        self.grid_layout.addWidget(self.checkbox_sync_positions, 2, 0, 1, 2)

        # main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.milling_stage_editor)
        self.main_layout.addLayout(self.grid_layout)

    def _initialise_widgets(self):
        if self.experiment is not None:
            for pos in self.experiment.positions:
                self.comboBox_selected_lamella.addItem(pos.name, pos)
        
        for k in self.protocol.milling.keys():
            self.comboBox_selected_milling.addItem(k)

        self.comboBox_selected_lamella.currentIndexChanged.connect(self._on_selected_lamella_changed)
        self.comboBox_selected_milling.currentIndexChanged.connect(self._on_selected_milling_stage_changed)
        self.milling_stage_editor._milling_stages_updated.connect(self._on_milling_stages_updated)
        self.milling_stage_editor.scroll_area.setMinimumHeight(550)

        if self.experiment is not None:
            self._on_selected_lamella_changed()
        else:
            self.label_selected_lamella.setVisible(False)
            self.comboBox_selected_lamella.setVisible(False)
            self.image = FibsemImage.generate_blank_image(hfw=80e-6, random=True)
            self.milling_stage_editor.set_image(self.image)
            self._on_selected_milling_stage_changed()

    def _on_selected_lamella_changed(self):
        """Callback when the selected lamella changes."""
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()

        reference_image_path = os.path.join(selected_lamella.path, "ref_PositionReady.tif")
        if os.path.exists(reference_image_path):
           self.image = FibsemImage.load(reference_image_path)
        else:
            logging.warning(f"Reference image not found at {reference_image_path}. Generating blank image.")
            self.image = FibsemImage.generate_blank_image(hfw=150e-6, random=True)

        self.milling_stage_editor.set_image(self.image)
        self._on_selected_milling_stage_changed()

    def _on_selected_milling_stage_changed(self):
        """Callback when the selected milling stage changes."""
        selected_stage_name = self.comboBox_selected_milling.currentText()
        
        self.background_milling_stages = []
        if self.experiment is not None:
            selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
            protocol =  selected_lamella.protocol
            milling_stages = get_milling_stages(selected_stage_name, protocol)
            self._get_background_milling_stages(selected_stage_name, protocol)
        else:
            milling_stages = self.protocol.milling[selected_stage_name]
            for k, ms in self.protocol.milling.items():
                if k == selected_stage_name:
                    continue
                self.background_milling_stages.extend(ms)
            self.image = FibsemImage.generate_blank_image(hfw=milling_stages[0].milling.hfw, random=True)
            self.milling_stage_editor.set_image(self.image)

        self.milling_stage_editor.set_background_milling_stages(self.background_milling_stages)
        self.milling_stage_editor.set_milling_stages(milling_stages)

    def _get_background_milling_stages(self, stage_name: str, protocol: Dict[str, List]) -> List[FibsemMillingStage]:
        """Get the background (non-selected) milling stages for a given stage name."""
        self.background_milling_stages = []
        other_keys = [k for k in protocol.keys() if k != stage_name]
        for k in other_keys:
            self.background_milling_stages.extend(get_milling_stages(k, protocol))

    def _on_milling_stages_updated(self, milling_stages: List[FibsemMillingStage]):
        """Callback when the milling stages are updated."""

        pprotocol = get_protocol_from_stages(milling_stages)

        selected_stage_name = self.comboBox_selected_milling.currentText()

        if self.experiment is None:
            self.protocol.milling[selected_stage_name] = copy.deepcopy(milling_stages)
            return

        # idx = self.comboBox_selected_lamella.currentIndex()
        selected_lamella: Lamella = self.comboBox_selected_lamella.currentData()
        selected_lamella.protocol[selected_stage_name] = copy.deepcopy(pprotocol)

        # NOTE: this sync is awful, but it is a quick fix to sync the trench position
        # ideally, we want to display both rough/polsihing at the same time, and allow the user to edit them both
        # but need to migrate to tree-widget or similar to allow for multi-stage editing
        # leaving this for now
        point = None
        for ms in milling_stages:
            if isinstance(ms.pattern, TrenchPattern):
                point = ms.pattern.point
                break

        if point is None:
            logging.warning("No trench pattern found in the milling stages. Cannot sync positions.")

        # DONT SYNC POLISHING -> ROUGH, ONLY ROUGH -> POLISHING?        
        sync_positions = self.checkbox_sync_positions.isChecked()
        if sync_positions and point is not None:
            if selected_stage_name == "mill_rough":
                for ms in selected_lamella.protocol["mill_polishing"]:
                    if ms["pattern"]["name"] == TrenchPattern.name:
                        ms["pattern"]["point"] = copy.deepcopy(point.to_dict())
                        logging.info(f"Syncing polishing pattern {ms['name']} point to {point}")


        # save the experiment
        self.experiment.save()

        # update the main ui??? DO WE NEED TO DO THIS?
        # reference seems to go through?, maybe just refresh ui?
        # NOTE: this causes a de-sync if the workflow is running, 
        # we need to save the experiment to disk, then reload in the main ui / workflow thread
        # otherwise we get these weird desync issues
        if self.parent is not None and not self.parent.WORKFLOW_IS_RUNNING: 
            self.parent.update_experiment_signal.emit(self.experiment)

        # reset the background milling stages, force refresh ui
        self._get_background_milling_stages(selected_stage_name, selected_lamella.protocol)
        self.milling_stage_editor.set_background_milling_stages(self.background_milling_stages)
        self.milling_stage_editor.update_milling_stage_display()

        # TODO: this causes a double update, need to fix this

def show_protocol_editor(viewer: napari.Viewer, 
                         microscope: FibsemMicroscope, 
                         protocol: AutoLamellaProtocol, 
                         experiment: Experiment = None, 
                         parent: QWidget = None):
    """Show the AutoLamella protocol editor widget."""
    widget = AutoLamellaProtocolEditorWidget(viewer=viewer, 
                                             microscope=microscope, 
                                             experiment=experiment, 
                                             protocol=protocol, 
                                             parent=parent)
    viewer.window.add_dock_widget(widget, area='right', name='AutoLamella Protocol Editor')
    napari.run(max_loop_level=2)
    return widget

if __name__ == "__main__":
    
    from autolamella.structures import AutoLamellaProtocol

    microscope, settings = utils.setup_session()
    viewer = napari.Viewer()


    BASE_PATH = "/home/patrick/github/autolamella/autolamella/log/AutoLamella-2025-05-28-17-22/"
    EXPERIMENT_PATH = os.path.join(BASE_PATH, "experiment.yaml")
    PROTOCOL_PATH = os.path.join(BASE_PATH, "protocol.yaml")
    exp = Experiment.load(EXPERIMENT_PATH)
    protocol = AutoLamellaProtocol.load(PROTOCOL_PATH)

    widget=show_protocol_editor(viewer=viewer,
                         microscope=microscope, 
                         experiment=exp, 
                         protocol=protocol)



# TODO: re-sizing base image?? scale bar
# TODO: export protocol to yaml file
# TODO: re-fresh lamella list when lamella added/removed
# TODO: allow 'live' edits of the protocol while workflow is running? SCARY
# TODO: allow editing the 'master' protocol, so we can change the default milling stages
# TODO: show multiple-stage milling patterns in the viewer?
# TODO: what to do when we want to move multi-stages to the same position, e.g. rough-milling and polishing?
# - This may be breaking, and we need a way to handle it rather than moving them individually.
# QUERY: WHY ISN"T PROTOCOL PART OF EXPERIMENT?????