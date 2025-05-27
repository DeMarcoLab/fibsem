import copy
import logging
from typing import Dict, List, Tuple

import napari
import napari.utils.notifications
from napari.layers import Image as NapariImageLayer
from PyQt5.QtCore import Qt
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
)

from fibsem import config as cfg
from fibsem import conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import get_strategy
from fibsem.milling.patterning.patterns2 import (
    MILLING_PATTERN_NAMES,
    BasePattern,
    LinePattern,
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
    FibsemMillingSettings,
    ImageSettings,
    MillingAlignment,
    Point,
)
from fibsem.ui import stylesheets
from fibsem.ui.FibsemMillingWidget import WheelBlocker, get_default_milling_pattern
from fibsem.ui.napari.patterns import (
    draw_milling_patterns_in_napari,
    is_pattern_placement_valid,
)
from fibsem.ui.napari.utilities import is_position_inside_layer

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
        "tooltip": "The number of passes for the milling pattern.",
    }
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


PARAMETER_MAPPING = {
    "milling_current": "current",
    "milling_voltage": "voltage",
}

def format_value(val: float, unit: str = None, precision: int = 2) -> str:
    """Format a numerical value as a string with nearest SI unit."""
    if val < 1e-9:
        scale = 1e12
        prefix = "p"
    elif val < 1e-6:
        scale = 1e9
        prefix = "n"
    elif val < 1e-3:
        scale = 1e6
        prefix = "u"
    elif val < 1:
        scale = 1e3
        prefix = "m"
    elif val < 1e3:
        scale = 1
        prefix = ""
    elif val < 1e6:
        scale = 1e-3
        prefix = "k"
    elif val < 1e9:
        scale = 1e-6
        prefix = "M"
    elif val < 1e12:
        scale = 1e-9
        prefix = "G"
    else:
        scale = 1e-12
        prefix = "T"

    if unit is None:
        unit = ""
    return f"{val*scale:.{precision}f} {prefix}{unit}"




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
# point controls for pattern
# milling stage name?

# DONE:
# when we change selection of pattern or strategy, need to also update milling_stage object
# parameter return
# strategy selection
# multi-stages
# advanced settings display

from PyQt5.QtCore import Qt, pyqtSignal

from fibsem.milling import FibsemMillingStage


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
        label.setText(f"Milling Stage: {self._milling_stage.name}")
        label.setObjectName("label-milling-stage-name")
        self.gridlayout.addWidget(label, 0, 0, 1, 2)  # Add label to the top of the grid layout
        for widget in self._widgets:
            # widget.setLayout(QGridLayout())
            self.gridlayout.addWidget(widget, self.gridlayout.rowCount(), 0, 1, 2)

    def _initialise_widgets(self):
        """Initialise the widgets with the current milling stage settings."""
        # MILLING SETTINGS
        milling_parames = self._milling_stage.milling.get_parameters(self._manufacturer)
        self._create_controls(self.milling_widget, milling_parames, "milling", GUI_CONFIG["milling"].copy())

        # PATTERN
        # don't emit event on comboBox_selected_pattern change
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

    def clear_widget(self, widget: QWidget, row_threshold: int = -1):
        """Clear the widget's layout, removing all items below a certain row threshold."""

        items_to_remove = []
        grid_layout: QGridLayout = widget.layout()

        # iterate through the items in the grid layout
        for i in range(grid_layout.count()):
            item = grid_layout.itemAt(i)
            if item is not None:
                row, col, rowspan, colspan = grid_layout.getItemPosition(i)
                # print(f"Found widget at row {row}, col {col}")  # Debug info
                if row > row_threshold:
                    items_to_remove.append(item)
                    # print(f" {item.widget().objectName()} -> Will remove widget at row {row}, col {col}")
        
        # Remove the items
        for item in items_to_remove:
            grid_layout.removeItem(item)
            if item.widget():
                item.widget().deleteLater()
        
    def _on_pattern_changed(self, pattern_name: str):
        # TODO: comvert the comboBox_selected_pattern to use currentData, 
        # that way we can pass the pattern object directly (and restore it from the previous state)
        pattern = get_default_milling_pattern(pattern_name)

        self._milling_stage.pattern = pattern  # Update the milling stage's strategy

        self._update_pattern_widget(pattern)
        self._milling_stage_changed.emit(self._milling_stage)  # Emit signal to notify changes

    def _update_pattern_widget(self, pattern: BasePattern):
        """Update the pattern widget with the selected pattern's parameters."""

        params = {k: getattr(pattern, k) for k in pattern.required_attributes if hasattr(pattern, k)}
        params["point"] = pattern.point  # Add point as a special case

        self.clear_widget(self.pattern_widget, row_threshold=0)  # Clear previous controls
        self._create_controls(self.pattern_widget, params, "pattern", GUI_CONFIG["pattern"].copy())

    def _on_strategy_changed(self, strategy_name: str):
        """Update the strategy widget with the selected strategy's parameters."""
        strategy = get_strategy(strategy_name, {"config": {}})

        self._milling_stage.strategy = strategy  # Update the milling stage's strategy

        self._update_strategy_widget(strategy)  # Update the strategy widget
        self._milling_stage_changed.emit(self._milling_stage)  # Emit signal to notify changes
        
    def _update_strategy_widget(self, strategy: MillingStrategy):
        """Update the strategy widget with the selected strategy's parameters."""
        params = {k: getattr(strategy.config, k) for k in strategy.config.required_attributes}

        self.clear_widget(self.strategy_widget, row_threshold=0)  # Clear previous controls
        self._create_controls(self.strategy_widget, params, "strategy.config", GUI_CONFIG["strategy"].copy())

    def _create_controls(self, widget: QWidget, params: Dict[str, any], cls: str, config: Dict[str, any]):
        """Create controls for the given parameters and add them to the widget."""

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

            # print(name, value, gui_config)

            # add combobox controls
            if items:
                if items == "dynamic":
                    items = microscope.get_available_values(PARAMETER_MAPPING.get(name, name), BeamType.ION)

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
                        # print(f"Adding resolution item: {item_str} for {name}")
                    else:
                        item_str = str(item)
                    control.addItem(item_str, item)

                if isinstance(value, tuple) and len(value) == 2:
                    value = list(value)  # Convert tuple to list for easier handling

                # find the closest match to the current value (should only be used for numerical values)
                idx = control.findData(value)
                print(name, value, idx, items)
                if idx == -1:
                    # get the closest value
                    closest_value = min(items, key=lambda x: abs(x - value))
                    idx = control.findData(closest_value)
                if idx == -1:
                    print(f"Warning: No matching item or nearest found for {name} with value {value}. Using first item.")
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
                print("Setting value for control:", name, "to", value)
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

        print(f"Updating settings... {obj_name}, {cls}, {name} changed ")

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
            print(f"Unknown control type: {type(obj)}")
            return

        print(f"Setting {cls}.{name} to {value} (type: {type(value)})")

        # Update the milling_stage object
        if hasattr(self._milling_stage, cls):

            # special case for pattern point
            if "point" in name:
                if "x" in name:
                    setattr(self._milling_stage.pattern.point, "x", value)
                elif "y" in name:
                    setattr(self._milling_stage.pattern.point, "y", value)
            else:
                setattr(getattr(self._milling_stage, cls), name, value)
            print(f"Updated {cls}.{name} to {value}")
        elif hasattr(self._milling_stage, "strategy") and cls == "strategy.config":
            # Special case for strategy config
            setattr(self._milling_stage.strategy.config, name, value)
            print(f"Updated strategy.config.{name} to {value}")
        else:
            print(f"Warning: {cls} not found in milling_stage object. Cannot update {name}.")

        self._milling_stage_changed.emit(self._milling_stage)  # Emit signal to notify changes 
        print("-----------------------------------------------------")

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
    """A widget to edit the milling stage settings."""

    def __init__(self, viewer: napari.Viewer, microscope: FibsemMicroscope, milling_stages: List[FibsemMillingStage], parent=None):
        super().__init__(parent)
        self.microscope = microscope
        self._milling_stages = milling_stages

        self.viewer: napari.ViewerModel = viewer
        self.image: FibsemImage = FibsemImage.generate_blank_image(hfw=80e-6)
        self.image_layer: NapariImageLayer = self.viewer.add_image(data=self.image.data, name="FIB Image")

        self.UPDATING_PATTERN = False  # Flag to prevent recursive updates

        self.viewer.mouse_drag_callbacks.append(self._on_single_click)

        # Create the scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Important: allows the widget to resize
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create the widget that will contain all the scrollable content
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content.setContentsMargins(0, 0, 0, 0)  # Remove margins for better scrolling
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        self._widgets: List[FibsemMillingStageWidget] = []  # Store the widgets for each milling stage

        self.listWidget_active_milling_stages = QListWidget(self)

        # Add multiple widgets to the scroll content
        for milling_stage in self._milling_stages:
            self._add_milling_stage_widget(milling_stage)

        # re-ordering support
        self.listWidget_active_milling_stages.setDragDropMode(QListWidget.InternalMove)
        self.listWidget_active_milling_stages.setDefaultDropAction(Qt.MoveAction)
        model = self.listWidget_active_milling_stages.model()
        model.rowsMoved.connect(self._reorder_milling_stages)

        # Set the scroll content widget to the scroll area
        scroll_area.setWidget(self.scroll_content)
        scroll_area.setContentsMargins(0, 0, 0, 0)  # Remove margins for better scrolling
        # Add the scroll area to the main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(scroll_area)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # HBoxLayout with Add and Remove buttons
        pushButton_add = QPushButton("Add Milling Stage", self)
        pushButton_add.clicked.connect(lambda: self._add_milling_stage(None))
        pushButton_add.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        pushButton_remove = QPushButton("Remove Selected Stage", self)
        pushButton_remove.clicked.connect(self._remove_selected_milling_stage)
        pushButton_remove.setStyleSheet(stylesheets.RED_PUSHBUTTON_STYLE)
        button_layout = QHBoxLayout()
        button_layout.addWidget(pushButton_add)
        button_layout.addWidget(pushButton_remove)
        self.main_layout.addLayout(button_layout)

        # HBoxLayout with Set, Clear buttons for milling stages
        self._tmp_milling_stages = copy.deepcopy(milling_stages)
        pushButton_add = QPushButton("Set Milling Stages", self)
        pushButton_add.clicked.connect(lambda: self.set_milling_stages(self._tmp_milling_stages))
        pushButton_add.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        pushButton_remove = QPushButton("Clear Milling Stages", self)
        pushButton_remove.clicked.connect(self.clear_milling_stages)
        pushButton_remove.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
        button_layout = QHBoxLayout()
        button_layout.addWidget(pushButton_add)
        button_layout.addWidget(pushButton_remove)
        self.main_layout.addLayout(button_layout)

        pushButton = QPushButton("Show Milling Stages", self)
        pushButton.clicked.connect(self._show_milling_stages)
        self.main_layout.addWidget(pushButton)
        pushButton.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

        grid_layout = QGridLayout()
        # add checkboxes for show advanced settings, show milling crosshair, show milling patterns
        self.checkBox_show_advanced_settings = QCheckBox("Show Advanced Settings", self)
        self.checkBox_show_advanced_settings.setChecked(False)
        self.checkBox_show_advanced_settings.setToolTip("Show advanced settings for milling stages.")
        self.checkBox_show_advanced_settings.stateChanged.connect(self._toggle_advanced_settings)
        grid_layout.addWidget(self.checkBox_show_advanced_settings, 0, 0, 1, 1)
        self.checkBox_show_milling_crosshair = QCheckBox("Show Milling Crosshair", self)
        self.checkBox_show_milling_crosshair.setChecked(True)
        self.checkBox_show_milling_crosshair.setToolTip("Show the milling crosshair in the viewer.")
        self.checkBox_show_milling_crosshair.stateChanged.connect(self._show_milling_stages)
        grid_layout.addWidget(self.checkBox_show_milling_crosshair, 0, 1, 1, 1)
        self.checkBox_show_milling_patterns = QCheckBox("Show Milling Patterns", self)
        self.checkBox_show_milling_patterns.setChecked(True)
        self.checkBox_show_milling_patterns.setToolTip("Show the milling patterns in the viewer.")
        self.checkBox_show_milling_patterns.stateChanged.connect(self._toggle_pattern_visibility)
        grid_layout.addWidget(self.checkBox_show_milling_patterns, 1, 0, 1, 1)
        # NOT ADDING NON-RELATIVE MOVE FOR NOW
        self.main_layout.addLayout(grid_layout)

        # Add the list widget to the main layout
        self.main_layout.addWidget(self.listWidget_active_milling_stages)
        self.listWidget_active_milling_stages.setMaximumHeight(100)
        self.listWidget_active_milling_stages.itemSelectionChanged.connect(self._on_selected_stage_changed)
        self.listWidget_active_milling_stages.itemChanged.connect(self._on_milling_stage_check_changed)
        # set initial list widget selection to row 0
        if self.listWidget_active_milling_stages.count() > 0:
            self.listWidget_active_milling_stages.setCurrentRow(0)

        self._toggle_advanced_settings(self.checkBox_show_advanced_settings.checkState())
        self._show_milling_stages()

    def _toggle_advanced_settings(self, state: int):
        """Toggle the visibility of advanced settings in the milling stage editor."""
        print(f"Toggling advanced settings: {'Enabled' if state == Qt.Checked else 'Disabled'}")

        show_advanced = bool(state == Qt.Checked)

        for widget in self._widgets:

            ms = widget.get_milling_stage()
            for param in ms.pattern.advanced_attributes:
                label, control, _ = widget.parameters["pattern"].get(param, (None, None, None))
                if label:
                    label.setVisible(show_advanced)
                if control:
                    control.setVisible(show_advanced)
            for param in ms.strategy.config.advanced_attributes:
                label, control, _ = widget.parameters["strategy.config"].get(param, (None, None, None))
                if label:
                    label.setVisible(show_advanced)
                if control:
                    control.setVisible(show_advanced)
            for param in ms.milling.advanced_attributes:
                label, control, _ = widget.parameters["milling"].get(param, (None, None, None))
                if label:
                    label.setVisible(show_advanced)
                if control:
                    control.setVisible(show_advanced)
            # QUERY: should we also hide imaging, alignment?
            # consider strategy as advanced, so hide it as well
            widget.strategy_widget.setVisible(show_advanced)

    def _toggle_pattern_visibility(self, state: int):
        """Toggle the visibility of milling patterns in the viewer."""
        visible = bool(state == Qt.Checked)
        if self.milling_pattern_layers:
            for layer in self.milling_pattern_layers:
                if layer in self.viewer.layers:
                    self.viewer.layers[layer].visible = visible

    def _reorder_milling_stages(self, parent, start, end, destination, row):
        """Sync the object list when UI is reordered"""
        print(f"Syncing objects: moving {start}-{end} to {row}. ")
        
        # Same logic as before, but with objects
        dest_index = row if row < start else row - (end - start + 1)
        
        # Move objects in the list
        objects_to_move = self._milling_stages[start:end+1]
        del self._milling_stages[start:end+1]
        
        for i, obj in enumerate(objects_to_move):
            self._milling_stages.insert(dest_index + i, obj)
        
        print(f"Objects reordered: {[obj.name for obj in self._milling_stages]}")

        # when we re-order, we need to re-order the widgets as well
        dest_widgets = self._widgets[start:end+1]
        del self._widgets[start:end+1]
        for i, widget in enumerate(dest_widgets):
            self._widgets.insert(dest_index + i, widget)

        self._show_milling_stages()

    def _remove_selected_milling_stage(self):
        """Remove the selected milling stage from the list widget."""
        selected_items = self.listWidget_active_milling_stages.selectedItems()
        if not selected_items:
            print("No items selected to remove.")
            return

        for item in selected_items:
            index = self.listWidget_active_milling_stages.row(item)
            self.listWidget_active_milling_stages.takeItem(index)
            # also remove the corresponding widget
            if index < len(self._widgets):
                widget = self._widgets.pop(index)
                widget.deleteLater()
            
            self._milling_stages.pop(index)  # Remove from the milling stages list
            print(f"Removed item: {item.text()} at index {index}")

        self._on_selected_stage_changed()  # Update the display after removal
        self._show_milling_stages()  # Refresh the milling stages display

    def clear_milling_stages(self):
        """Clear all milling stages from the editor."""
        print("Clearing all milling stages.")
        self._milling_stages.clear()
        self.listWidget_active_milling_stages.clear()

        # clear previous widgets
        for widget in self._widgets:
            widget.deleteLater()
        self._widgets.clear()
        
        self._show_milling_stages()  # Refresh the milling stages display

    def set_milling_stages(self, milling_stages: List[FibsemMillingStage]):
        """Set the milling stages to be displayed in the editor."""

        self.clear_milling_stages()  # Clear existing milling stages
        self._milling_stages = copy.deepcopy(milling_stages)
        for milling_stage in self._milling_stages:
            self._add_milling_stage_widget(milling_stage)
        
        self._show_milling_stages()  # Refresh the milling stages display

    def _add_milling_stage(self, milling_stage: FibsemMillingStage = None):

        if milling_stage is None:
            num = len(self._milling_stages) + 1
            milling_stage = FibsemMillingStage(name=f"Milling Stage {num}", num=num)

        # Create a new widget for the milling stage
        logging.info(f"Added new milling stage: {milling_stage.name}")
        self._milling_stages.append(milling_stage)  # Add to the milling stages list

        self._add_milling_stage_widget(milling_stage)  # Add the widget to the scroll area
        self.listWidget_active_milling_stages.setCurrentRow(self.listWidget_active_milling_stages.count()-1)  # Select the new item
        
        # Update the display to show the new milling stage
        self._on_selected_stage_changed()  # Update the display after adding
        self._show_milling_stages()  # Refresh the milling stages display

    def _add_milling_stage_widget(self, milling_stage: FibsemMillingStage):
        milling_stage_widget = FibsemMillingStageWidget(self.microscope, milling_stage=milling_stage)
        self.scroll_layout.addWidget(milling_stage_widget)
        self._widgets.append(milling_stage_widget)

        # connect the signal to update the milling stages when a widget changes
        milling_stage_widget._milling_stage_changed.connect(
            lambda stage: self._show_milling_stages()  # Update the milling stages when a widget changes
        )

        # Create a list item for the active milling stage
        item = QListWidgetItem(f"{milling_stage.name} - {milling_stage.pattern.name}")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.listWidget_active_milling_stages.addItem(item)

    def _on_selected_stage_changed(self):
        print("Selected stage changed")
        selected_items = self.listWidget_active_milling_stages.selectedItems()
        if selected_items:
            index = self.listWidget_active_milling_stages.currentRow()
            print("Current selected index:", index)

            # hide all widgets, except selected
            for i, widget in enumerate(self._widgets):
                if i == index:
                    widget.show()
                else:
                    widget.hide()
        else:   
            print("No items selected")
            # hide all widgets
            for widget in self._widgets:
                widget.hide()

    def _on_milling_stage_check_changed(self, item: QListWidgetItem):
        print(item.text(), "checked state:", item.checkState())
        self._show_milling_stages()

    def _get_selected_milling_stages(self) -> List[FibsemMillingStage]:
        """Return the milling stages that are selected (checked) in the list widget."""
        checked_indexes = []
        for i in range(self.listWidget_active_milling_stages.count()):
            item = self.listWidget_active_milling_stages.item(i)
            if item.checkState() == Qt.Checked:
                checked_indexes.append(i)

        milling_stages = [widget.get_milling_stage() for i, widget in enumerate(self._widgets) if i in checked_indexes]
        return milling_stages

    def get_milling_stages(self) -> List[FibsemMillingStage]:
        """Public method to get the currently selected milling stages."""
        return self._get_selected_milling_stages()

    def _show_milling_stages(self):
        """Return the milling stages that are checked in the list widget."""
        if self.UPDATING_PATTERN:
            return # Prevent updates while moving patterns
        milling_stages = self._get_selected_milling_stages()

        if not milling_stages:
            print("No milling stages selected.")
            try:
                for layer in self.milling_pattern_layers:
                    if layer in self.viewer.layers:
                        self.viewer.layers.remove(layer)
            except Exception as e:
                print("Error removing milling pattern layers:", e)
            self.milling_pattern_layers = []
            return

        print("Checked milling stages:", [stage.name for stage in milling_stages])

        if self.image is None:
            image = FibsemImage.generate_blank_image(hfw=milling_stages[0].milling.hfw)
            self.set_image(image)

        self.milling_pattern_layers = draw_milling_patterns_in_napari(
            viewer=self.viewer,
            image_layer=self.image_layer,
            milling_stages=milling_stages,
            pixelsize=self.image.metadata.pixel_size.x,
            draw_crosshair=self.checkBox_show_milling_crosshair.isChecked(),
        )

    def set_image(self, image: FibsemImage) -> None:

        self.image = image  # Store the image in the widget

        try:
            self.image_layer.data = image.data
        except:
            self.image_layer = self.viewer.add_image(
                image.data,
                name="FIB Image",
            )

    def _on_single_click(self, viewer: napari.Viewer, event):
        
        if event.button != 1 or 'Shift' not in event.modifiers or self._milling_stages == []:
            return
        
    
        if not self.image_layer:
            logging.warning("No target layer found for the click event.")
            return


        if not is_position_inside_layer(event.position, self.image_layer):
            logging.warning("Click position is outside the image layer.")
            return

        coords = self.image_layer.world_to_data(event.position)
        current_stage_index = self.listWidget_active_milling_stages.currentRow()

        if current_stage_index < 0 or current_stage_index >= len(self._milling_stages):
            logging.warning("No milling stage selected or index out of range.")
            return

        # convert from image coordinates to microscope coordinates
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
        diff = point_clicked - self._milling_stages[current_stage_index].pattern.point

        # loop to check through all patterns to see if they are in bounds
        for idx, milling_stage in enumerate(self._milling_stages):
            if not is_moving_all_patterns:
                if idx != current_stage_index: 
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
            self.UPDATING_PATTERN = True
            if is_moving_all_patterns: # moving all patterns
                for idx, new_point in enumerate(new_points):
                    self._widgets[idx].set_point(new_point)
            else: # only moving selected pattern
                self._widgets[current_stage_index].set_point(point_clicked)

        self.UPDATING_PATTERN = False
        self._show_milling_stages()  # Refresh the milling stages display


if __name__ == "__main__":
    import sys

    # app = QApplication(sys.argv)
    import napari
    from PyQt5.QtWidgets import QApplication
    viewer = napari.Viewer()

    from fibsem import utils
    from fibsem.structures import BeamType
    microscope, settings = utils.setup_session()

    PROTOCOL_PATH = "/home/patrick/github/autolamella/autolamella/protocol/protocol-waffle.yaml"
    from autolamella.structures import AutoLamellaProtocol
    protocol = AutoLamellaProtocol.load(PROTOCOL_PATH)

    milling_stages = protocol.milling

    _milling_stages = []
    _milling_stages.extend(milling_stages["mill_rough"])
    _milling_stages.extend(milling_stages["mill_polishing"])

    from pprint import pprint
    pprint(_milling_stages)

    main_widget = FibsemMillingStageEditorWidget(viewer, microscope, _milling_stages)
    settings.image.hfw = 80e-6
    settings.image.beam_type = BeamType.ION
    image = microscope.acquire_image(settings.image)
    main_widget.set_image(image)

    viewer.window.add_dock_widget(main_widget, name="Milling Stages", area='right')
    # Show the widget

    # widget.show()
    # sys.exit(app.exec_())

    napari.run()
