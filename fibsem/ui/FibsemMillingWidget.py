
import logging
import time
from PIL import Image
import numpy as np
from copy import deepcopy
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets, QtCore

from fibsem import config as cfg
from fibsem import constants, conversions, milling, patterning, utils
from fibsem.microscope import (DemoMicroscope, FibsemMicroscope,
                               TescanMicroscope, ThermoMicroscope)
from fibsem.patterning import FibsemMillingStage
from fibsem.structures import (BeamType, FibsemMillingSettings,
                               FibsemPatternSettings, MicroscopeSettings,
                               Point, FibsemPattern)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMillingWidget
from fibsem.ui.utils import _draw_patterns_in_napari, _remove_all_layers, convert_pattern_to_napari_circle, convert_pattern_to_napari_rect, validate_pattern_placement,_get_directory_ui,_get_file_ui
from napari.qt.threading import thread_worker

_UNSCALED_VALUES  = ["rotation", "size_ratio", "scan_direction", "cleaning_cross_section", "number", "passes"]
_ANGLE_KEYS = ["rotation"]

def _scale_value(key, value, scale):
    if key not in _UNSCALED_VALUES:
        return value * scale    
    return value

def log_status_message(stage: FibsemMillingStage, step: str):
    logging.debug(
        f"STATUS | Milling Widget | {stage.name} | {step}"
    )

class FibsemMillingWidget(FibsemMillingWidget.Ui_Form, QtWidgets.QWidget):
    milling_position_changed = QtCore.pyqtSignal()
    _milling_finished = QtCore.pyqtSignal()

    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        image_widget: FibsemImageSettingsWidget = None,
        protocol: dict = None,
        milling_stages: list[FibsemMillingStage] = [], 
        parent=None,
    ):
        super(FibsemMillingWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget

        if protocol is None:
            protocol = utils.load_yaml(cfg.PROTOCOL_PATH)
        self.protocol = protocol
        
        self.milling_stages = milling_stages

        self.setup_connections()

        self.update_pattern_ui()

        self.good_copy_pattern = None

        self._UPDATING_PATTERN:bool = False
        self._PATTERN_IS_MOVEABLE: bool = True

    def setup_connections(self):

        self.image_widget.viewer_update_signal.connect(self.update_ui) # this happens after every time the viewer is updated

        # milling
        available_currents = self.microscope.get_available_values("current", BeamType.ION)
        self.comboBox_milling_current.addItems([str(current) for current in available_currents])

        _THERMO = isinstance(self.microscope, ThermoMicroscope)
        _TESCAN = isinstance(self.microscope, TescanMicroscope)

        if isinstance(self.microscope, DemoMicroscope):
            _THERMO, _TESCAN = True, False
        
        # THERMO 
        self.label_application_file.setVisible(_THERMO)
        self.comboBox_application_file.setVisible(_THERMO)
        available_application_files = self.microscope.get_available_values("application_file")
        self.comboBox_application_file.addItems(available_application_files)
        self.comboBox_preset.setVisible(_THERMO)
        self.label_preset.setVisible(_THERMO)
        self.comboBox_milling_current.setVisible(_THERMO)
        self.label_milling_current.setVisible(_THERMO)
        self.comboBox_application_file.currentIndexChanged.connect(self.update_settings)
        self.comboBox_milling_current.currentIndexChanged.connect(self.update_settings)
        self.doubleSpinBox_hfw.valueChanged.connect(self.update_settings)
        if self.comboBox_application_file.findText(self.protocol["milling"]["application_file"]) == -1:
                napari.utils.notifications.show_warning("Application file not available, setting to Si instead")
                self.protocol["milling"]["application_file"] = "Si"
        self.comboBox_application_file.setCurrentText(self.protocol["milling"]["application_file"])
        
        # TESCAN
        self.label_rate.setVisible(_TESCAN)
        self.label_spot_size.setVisible(_TESCAN)
        self.label_dwell_time.setVisible(_TESCAN)
        self.doubleSpinBox_rate.setVisible(_TESCAN)
        self.doubleSpinBox_spot_size.setVisible(_TESCAN)
        self.doubleSpinBox_dwell_time.setVisible(_TESCAN)   
        self.comboBox_preset.setVisible(_TESCAN)
        self.label_preset.setVisible(_TESCAN)
        self.label_spacing.setVisible(_TESCAN)
        self.doubleSpinBox_spacing.setVisible(_TESCAN)
        available_presets = self.microscope.get_available_values("presets")
        self.comboBox_preset.addItems(available_presets)   
        self.doubleSpinBox_rate.valueChanged.connect(self.update_settings)
        self.doubleSpinBox_spot_size.valueChanged.connect(self.update_settings)
        self.doubleSpinBox_dwell_time.valueChanged.connect(self.update_settings)  
        self.comboBox_preset.currentIndexChanged.connect(self.update_settings)
        self.doubleSpinBox_spacing.valueChanged.connect(self.update_settings)
        

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_drag_callbacks.append(self._single_click)
        self.image_widget.ib_layer.mouse_drag_callbacks.append(self._single_click)

        # new patterns
        self.comboBox_patterns.addItems([pattern.name for pattern in patterning.__PATTERNS__])
        if _TESCAN and not _THERMO:
            index = self.comboBox_patterns.findText("BitmapPattern")
            self.comboBox_patterns.removeItem(index)
        self.comboBox_patterns.currentIndexChanged.connect(lambda: self.update_pattern_ui(None))
    
        # milling stages
        self.pushButton_add_milling_stage.clicked.connect(self.add_milling_stage)
        self.pushButton_add_milling_stage.setStyleSheet("background-color: green; color: white;")
        self.pushButton_remove_milling_stage.clicked.connect(self.remove_milling_stage)
        self.pushButton_remove_milling_stage.setStyleSheet("background-color: red; color: white;")
        
        # update ui
        self.pushButton.clicked.connect(lambda: self.update_ui())
        self.pushButton.setStyleSheet("background-color: blue; color: white;")

        # run milling
        self.pushButton_run_milling.clicked.connect(self.run_milling)

        if self.milling_stages:
            self.comboBox_milling_stage.addItems([stage.name for stage in self.milling_stages])
            self.update_milling_stage_ui()
        self.comboBox_milling_stage.currentIndexChanged.connect(lambda: self.update_milling_stage_ui())

        # last
        self.doubleSpinBox_centre_x.setKeyboardTracking(False)
        self.doubleSpinBox_centre_y.setKeyboardTracking(False)
        self.doubleSpinBox_centre_x.valueChanged.connect(self.update_ui_pattern)
        self.doubleSpinBox_centre_y.valueChanged.connect(self.update_ui_pattern)
        self.checkBox_live_update.setChecked(True)


        self._AVAILABLE_SCAN_DIRECTIONS = self.microscope.get_available_values(key="scan_direction")

    def update_settings(self):
        settings = self.get_milling_settings_from_ui()
        index = self.comboBox_milling_stage.currentIndex()
        if index != -1:
            self.milling_stages[index].milling = settings

    def add_milling_stage(self):
        logging.info("Adding milling stage")

        num = len(self.milling_stages) + 1
        name = f"Milling Stage {num}"
        pattern = patterning.RectanglePattern()
        pattern.define(self.protocol["patterns"]["Rectangle"], Point(0,0))
        milling_stage = FibsemMillingStage(name=name, num=num, pattern=pattern)
        self.milling_stages.append(milling_stage)
        self.comboBox_milling_stage.addItem(name)
        self.comboBox_milling_stage.setCurrentText(name)
        napari.utils.notifications.show_info(f"Added {name}.")
        log_status_message(self.milling_stages[-1], "CREATED_STAGE")

    def remove_milling_stage(self):
        logging.info("Removing milling stage")

        current_index = self.comboBox_milling_stage.currentIndex()
        log_status_message(self.milling_stages[current_index], "REMOVED_STAGE")
        self.milling_stages.pop(current_index)
        self.comboBox_milling_stage.removeItem(current_index)
        napari.utils.notifications.show_info(f"Removed milling stage.")            

    def _remove_all_stages(self):

        self.milling_stages = []
        self.comboBox_milling_stage.currentIndexChanged.disconnect()
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.addItems([stage.name for stage in self.milling_stages])
        self.comboBox_milling_stage.currentIndexChanged.connect(lambda: self.update_milling_stage_ui())

        _remove_all_layers(self.viewer) # remove all shape layers

    def set_milling_stages(self, milling_stages: list[FibsemMillingStage]) -> None:
        logging.debug(f"Setting milling stages: {len(milling_stages)}")
        self.milling_stages = milling_stages
        
        # very explicitly set what is happening
        self.comboBox_milling_stage.currentIndexChanged.disconnect()
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.addItems([stage.name for stage in self.milling_stages])
        self.comboBox_milling_stage.currentIndexChanged.connect(lambda: self.update_milling_stage_ui())
        
        self.comboBox_patterns.currentIndexChanged.disconnect()
        self.comboBox_patterns.setCurrentText(self.milling_stages[0].pattern.name)
        self.comboBox_patterns.currentIndexChanged.connect(lambda: self.update_pattern_ui(None))
        
        logging.debug(f"Set milling stages: {len(milling_stages)}")
        self.update_milling_stage_ui()
        # self.update_ui()

    def get_milling_stages(self):
        return self.milling_stages

    def get_point_from_ui(self):

        point = Point(x=self.doubleSpinBox_centre_x.value() * constants.MICRO_TO_SI, 
                      y=self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI)

        return point

    def update_milling_stage_ui(self):

        # get the selected milling stage
        current_index = self.comboBox_milling_stage.currentIndex()
        if current_index == -1:
            return

        milling_stage: FibsemMillingStage = self.milling_stages[current_index]

        # set the milling settings
        self.set_milling_settings_ui(milling_stage.milling)

        # set the pattern protcol
        self.update_pattern_ui(milling_stage)


    def update_ui_pattern(self):

        if self._UPDATING_PATTERN:
            return

        if self.checkBox_live_update.isChecked():
            self.update_ui()

    def update_milling_stage_from_ui(self):
                
        # get current milling stage
        current_index = self.comboBox_milling_stage.currentIndex()

        if current_index == -1:
            msg = f"No milling stages defined, cannot draw patterns."
            logging.warning(msg)
            napari.utils.notifications.show_warning(msg)
            return

        milling_stage = self.milling_stages[current_index]

        # update milling settings
        milling_stage.milling = self.get_milling_settings_from_ui()

        # update pattern and define
        milling_stage.pattern = self.get_pattern_from_ui_v2()

        napari.utils.notifications.show_info(f"Updated {milling_stage.name}.")
        return 

    def open_path_dialog(self):

        file_path = _get_file_ui(msg="Select Bitmap File",_filter= "*bmp")
        
        self.path_edit.setText(file_path)

    def update_pattern_ui(self,milling_stage: FibsemMillingStage = None):

        self._UPDATING_PATTERN = True

        # get current pattern
        if milling_stage is None:
            current_pattern_text = self.comboBox_patterns.currentText()
            patterns_available = [pattern.name for pattern in patterning.__PATTERNS__]
            pattern_available_index = patterns_available.index(current_pattern_text)
            pattern = patterning.__PATTERNS__[pattern_available_index]
            pattern_protocol = self.protocol["patterns"][pattern.name]
            point = None
        else:
            pattern = milling_stage.pattern
            pattern_protocol = milling_stage.pattern.protocol
            point = milling_stage.pattern.point
            self.comboBox_patterns.currentIndexChanged.disconnect()
            self.comboBox_patterns.setCurrentText(milling_stage.pattern.name)
            self.comboBox_patterns.currentIndexChanged.connect(lambda: self.update_pattern_ui(None))

        logging.debug(f"PATTERN: {pattern.name}, PROTOCOL: {pattern_protocol}")

        # clear layout
        for i in reversed(range(self.gridLayout_patterns.count())):
            self.gridLayout_patterns.itemAt(i).widget().setParent(None)

        # set widgets for each required key / value
        for i, key in enumerate(pattern.required_keys):
            if key == "path":
                label = QtWidgets.QLabel(key)
                self.path_edit = QtWidgets.QLineEdit()
                self.gridLayout_patterns.addWidget(label, i, 0)
                self.gridLayout_patterns.addWidget(self.path_edit, i, 1)
                self.path_edit.setText(pattern_protocol[key])
                path_explorer = QtWidgets.QPushButton("...")
                self.gridLayout_patterns.addWidget(path_explorer, i, 2)
                path_explorer.clicked.connect(self.open_path_dialog)
                self.path_edit.editingFinished.connect(self.update_ui_pattern)
                continue
            
            if key == "scan_direction":
                label = QtWidgets.QLabel(key)
                self.comboBox_scan_direction = QtWidgets.QComboBox()
                self.gridLayout_patterns.addWidget(label, i, 0)
                self.gridLayout_patterns.addWidget(self.comboBox_scan_direction, i, 1)
                self.comboBox_scan_direction.addItems(self._AVAILABLE_SCAN_DIRECTIONS)
                scan_direction_to_use = pattern_protocol[key] if pattern_protocol[key] in self._AVAILABLE_SCAN_DIRECTIONS else self._AVAILABLE_SCAN_DIRECTIONS[0]
                # logging.info(f'Scan direction to use: {scan_direction_to_use}, available: {self._AVAILABLE_SCAN_DIRECTIONS}')
                self.comboBox_scan_direction.setCurrentText(scan_direction_to_use)
                self.comboBox_scan_direction.currentIndexChanged.connect(self.update_ui_pattern)
                continue

            if key == "cleaning_cross_section":
                if isinstance(self.microscope, ThermoMicroscope) and pattern.name == "Circle":
                    continue
                label = QtWidgets.QLabel(key)
                self.checkbox_cleaning_cross_section = QtWidgets.QCheckBox()
                self.gridLayout_patterns.addWidget(label, i, 0)
                self.gridLayout_patterns.addWidget(self.checkbox_cleaning_cross_section, i, 1)
                self.checkbox_cleaning_cross_section.setChecked(pattern_protocol[key])
                self.checkbox_cleaning_cross_section.stateChanged.connect(self.update_ui_pattern)
                continue

            if key == "passes":
                label = QtWidgets.QLabel(key)
                self.passes_comboBox = QtWidgets.QLineEdit()
                self.passes_comboBox.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
                self.gridLayout_patterns.addWidget(label, i, 0)
                self.gridLayout_patterns.addWidget(self.passes_comboBox, i, 1)
                self.passes_comboBox.setText("N/A")
                self.passes_comboBox.editingFinished.connect(self.update_ui_pattern)
                continue
            label = QtWidgets.QLabel(key)
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.001)
            spinbox.setRange(0, 1000)
            spinbox.setValue(0)
            self.gridLayout_patterns.addWidget(label, i, 0)
            self.gridLayout_patterns.addWidget(spinbox, i, 1)
            spinbox.setKeyboardTracking(False)

            # get default values from self.protocol and set values
            if key in pattern_protocol:
                value = _scale_value(key, pattern_protocol[key], constants.SI_TO_MICRO)
                spinbox.setValue(value)
            
            spinbox.valueChanged.connect(self.update_ui_pattern)

        if point is not None:
            self.doubleSpinBox_centre_x.setValue(point.x * constants.SI_TO_MICRO)
            self.doubleSpinBox_centre_y.setValue(point.y * constants.SI_TO_MICRO)

        if self.milling_stages:
            self.update_ui()

        self._UPDATING_PATTERN = False
    
    def get_pattern_settings_from_ui(self, pattern: patterning.BasePattern):
        # get pattern protocol from ui
        pattern_dict = {}
        for i, key in enumerate(pattern.required_keys):
            if key == "path":
                # add path in ui and get from there
                path = self.path_edit.text()
                pattern_dict[key] = path
                continue
            if key == "scan_direction":
                pattern_dict[key] = self.comboBox_scan_direction.currentText()
                continue
            if key == "cleaning_cross_section":
                if isinstance(self.microscope, ThermoMicroscope) and pattern.name == "Circle":
                    continue
                pattern_dict[key] = self.checkbox_cleaning_cross_section.isChecked()
                continue
            if key == "passes":
                pattern_dict[key] = self.passes_comboBox.text() if self.passes_comboBox.text() != "N/A" else None
                continue

            spinbox = self.gridLayout_patterns.itemAtPosition(i, 1).widget()
            value = _scale_value(key, spinbox.value(), constants.MICRO_TO_SI)
            # value = value * constants.DEGREES_TO_RADIANS if key in _ANGLE_KEYS else value
            pattern_dict[key] = value # TODO: not everythign is in microns
        return pattern_dict

    def get_pattern_from_ui_v2(self):

        # get current pattern
        pattern = patterning.get_pattern(self.comboBox_patterns.currentText())

        pattern_dict = self.get_pattern_settings_from_ui(pattern)
        # define pattern
        point = Point(x=self.doubleSpinBox_centre_x.value() * constants.MICRO_TO_SI, y=self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI)
        pattern.define(protocol=pattern_dict, point=point)
        
        return pattern

    def _single_click(self, layer, event):
        """Callback for single click on image layer."""
        if event.button != 1 or 'Shift' not in event.modifiers or self.milling_stages == []:
            return

        if not self._PATTERN_IS_MOVEABLE:
            msg = f"Pattern is not moveable."
            logging.info(msg)
            napari.utils.notifications.show_info(msg)
            return

        self._UPDATING_PATTERN = True

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

        # only move the pattern if milling widget is activate and beamtype is ion?
        if self.checkBox_move_all_patterns.isChecked():
            renewed_patterns = []
            for milling_stage in self.milling_stages:
                # loop to check through all patterns to see if they are in bounds
                pattern_dict_existing = milling_stage.pattern.protocol
                pattern_name = milling_stage.pattern.name
                pattern_renew = patterning.get_pattern(pattern_name)
                pattern_renew.define(protocol=pattern_dict_existing, point=point)

                pattern_is_valid = self.valid_pattern_location(pattern_renew)

                if not pattern_is_valid:
                    logging.info(f"Could not move Patterns, out of bounds at at {point}")
                    napari.utils.notifications.show_warning(f"Patterns is not within the image.")
                    break
                else:
                    renewed_patterns.append(pattern_renew)

            if len(renewed_patterns) == len(self.milling_stages):
                for milling_stage, pattern_renew in zip(self.milling_stages, renewed_patterns):

                    milling_stage.pattern = pattern_renew
                    milling_stage.pattern.point = point

                self.doubleSpinBox_centre_x.setValue(point.x * constants.SI_TO_MICRO)
                self.doubleSpinBox_centre_y.setValue(point.y * constants.SI_TO_MICRO) # THIS TRIGGERS AN UPDATE
                logging.info(f"Moved patterns to {point} ")
                self.update_ui(milling_stages=self.milling_stages)
                self.milling_position_changed.emit()

                
        else:
            # update pattern
            current_stage_index = self.comboBox_milling_stage.currentIndex()
            pattern = patterning.get_pattern(self.comboBox_patterns.currentText())
            pattern_dict = self.milling_stages[current_stage_index].pattern.protocol
            pattern.define(protocol=pattern_dict, point=point)
            is_valid = self.valid_pattern_location(pattern)
            if is_valid:

                logging.info(f"MILL | {pattern.name} | {point - self.milling_stages[current_stage_index].pattern.point} | {BeamType.ION}")

                # update ui
                self.doubleSpinBox_centre_x.setValue(point.x * constants.SI_TO_MICRO)
                self.doubleSpinBox_centre_y.setValue(point.y * constants.SI_TO_MICRO)
                logging.info(f"Moved pattern to {point}")
                log_status_message(self.milling_stages[current_stage_index], f"MOVED_PATTERN_TO_{point}")
                self.good_copy_pattern = deepcopy(pattern)
                self.update_ui()
                self.milling_position_changed.emit()

            else:
                napari.utils.notifications.show_warning("Pattern is not within the image.")
                self.milling_stages[current_stage_index].pattern = self.good_copy_pattern
        


        self._UPDATING_PATTERN = False

    def valid_pattern_location(self,stage_pattern):
        
        for pattern_settings in stage_pattern.patterns:
            if pattern_settings.pattern is FibsemPattern.Circle:
                shape = convert_pattern_to_napari_circle(pattern_settings=pattern_settings, image=self.image_widget.ib_image)
            else:
                shape = convert_pattern_to_napari_rect(pattern_settings=pattern_settings, image=self.image_widget.ib_image)
            resolution = [self.image_widget.ib_image.data.shape[1], self.image_widget.ib_image.data.shape[0]]
            output = validate_pattern_placement(patterns=shape, resolution=resolution,shape=shape)
            if not output:
                return False
        
        return True
   
    def set_milling_settings_ui(self, milling: FibsemMillingSettings) -> None:

        if self.comboBox_milling_current.findText(str(milling.milling_current)) == -1:
            napari.utils.notifications.show_warning(f"Could not find milling current {milling.milling_current} in list of available currents.")
        else:
            self.comboBox_milling_current.setCurrentText(str(milling.milling_current))
        self.comboBox_application_file.setCurrentText(milling.application_file)
        self.doubleSpinBox_rate.setValue(milling.rate*constants.SI_TO_NANO)
        self.doubleSpinBox_dwell_time.setValue(milling.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_spot_size.setValue(milling.spot_size * constants.SI_TO_MICRO)
        self.doubleSpinBox_hfw.setValue(milling.hfw * constants.SI_TO_MICRO)
        self.comboBox_preset.setCurrentText(str(milling.preset))

    def get_milling_settings_from_ui(self):

        milling_settings = FibsemMillingSettings(
            milling_current=float(self.comboBox_milling_current.currentText()),
            application_file=self.comboBox_application_file.currentText(),
            rate=self.doubleSpinBox_rate.value()*1e-9,
            dwell_time = self.doubleSpinBox_dwell_time.value() * constants.MICRO_TO_SI,
            spot_size=self.doubleSpinBox_spot_size.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_hfw.value() * constants.MICRO_TO_SI,
            preset= self.comboBox_preset.currentText(),
            spacing=self.doubleSpinBox_spacing.value(),

        )

        return milling_settings

    def update_ui(self, milling_stages: list[FibsemMillingStage] = None):
        self.doubleSpinBox_hfw.setValue(self.image_widget.doubleSpinBox_image_hfw.value())

        if milling_stages is None and len(self.milling_stages) < 1:
            return
        
        t0 = time.time()
        if milling_stages is None:
            self.update_milling_stage_from_ui() # update milling stage from ui
            milling_stages = self.get_milling_stages() # get the latest milling stages from the ui

        t1 = time.time() 

        if not milling_stages:
            _remove_all_layers(self.viewer)
            return

        # make milling stage a list if it is not
        if not isinstance(milling_stages, list):
            milling_stages = [milling_stages]

        # # check hfw threshold
        # for stage in milling_stages:
        #     if stage.pattern.name == "Trench":
        #         if stage.pattern.protocol["trench_height"] / stage.pattern.protocol["hfw"] < cfg.MILL_HFW_THRESHOLD:
        #             napari.utils.notifications.show_warning(f"Pattern dimensions are too small for milling. Please decrease the image hfw or increase the trench height.")
        #             return

        t2 = time.time()
        try:
            
            from fibsem.structures import FibsemImage
            if not isinstance(self.image_widget.ib_image, FibsemImage):
                raise Exception(f"No Ion Image, cannot draw patterns. Please take an image.")
            if not isinstance(self.image_widget.eb_image, FibsemImage):
                raise Exception(f"No Electron Image, cannot draw patterns. Please take an image.") # TODO: this is unintuitive why this is required -> ui issue only
            # clear patterns then draw new ones
            _draw_patterns_in_napari(self.viewer, 
                ib_image=self.image_widget.ib_image, 
                eb_image=self.image_widget.eb_image, 
                milling_stages = milling_stages) # TODO: add names and legend for this
        
        except Exception as e:
            napari.utils.notifications.show_error(f"Error drawing patterns: {e}")
            logging.error(e)
            return
        t2 = time.time()
        logging.debug(f"UPDATE_UI: GET: {t1-t0}, DRAW: {t2-t1}")
        self.viewer.layers.selection.active = self.image_widget.eb_layer

    def _toggle_interaction(self, enabled: bool = True):

        """Toggle microscope and pattern interactions."""

        self.pushButton.setEnabled(enabled)
        self.pushButton_add_milling_stage.setEnabled(enabled)
        self.pushButton_remove_milling_stage.setEnabled(enabled)
        # self.pushButton_save_milling_stage.setEnabled(enabled)
        self.pushButton_run_milling.setEnabled(enabled)

        # change run milling to Running... and turn orange
        if enabled:
            self.pushButton_run_milling.setText("Run Milling")
            self.pushButton_run_milling.setStyleSheet("")
        else:
            self.pushButton_run_milling.setText("Running...")
            self.pushButton_run_milling.setStyleSheet("background-color: orange")

    def run_milling(self):

        worker = self.run_milling_step()
        worker.finished.connect(self.run_milling_finished)
        worker.yielded.connect(self.update_milling_ui)
        worker.start()

    @thread_worker
    def run_milling_step(self):

        milling_stages = self.get_milling_stages()
        self._toggle_interaction(enabled=False)
        for stage in milling_stages:
            yield f"Preparing: {stage.name}"
            if stage.pattern is not None:
                log_status_message(stage, f"RUNNING_MILLING_STAGE_{stage.name}")
                log_status_message(stage, f"MILLING_PATTERN_{stage.pattern.name}: {stage.pattern.patterns}")
                log_status_message(stage, f"MILLING_SETTINGS_{stage.milling}")
                milling.setup_milling(self.microscope, mill_settings=stage.milling)

                milling.draw_patterns(self.microscope, stage.pattern.patterns)

                yield f"Running {stage.name}..."
                milling.run_milling(self.microscope, stage.milling.milling_current)
            
                milling.finish_milling(self.microscope, self.settings.system.ion.current)

                log_status_message(stage, "MILLING_COMPLETED_SUCCESSFULLY")

            yield f"Milling stage complete: {stage.name}"
        yield f"Milling complete. {len(self.milling_stages)} stages completed."

    def update_milling_ui(self, msg: str):
        logging.info(msg)
        napari.utils.notifications.notification_manager.records.clear()
        napari.utils.notifications.show_info(msg)
        # TODO: progress bar?

    def run_milling_finished(self):

        # take new images and update ui
        self._toggle_interaction(enabled=True)
        self.image_widget.take_reference_images()
        self.update_ui()
        self._milling_finished.emit()


def main():
    millings_stages = [
    FibsemMillingStage(
        name="Milling Stage X",
        num = 1,
        milling = FibsemMillingSettings(hfw=400e-6),
        pattern = patterning.get_pattern("Trench"),
    ),
        FibsemMillingStage(
        name="Milling Stage 2",
        num = 2,
        milling = FibsemMillingSettings(hfw=200e-6),
        pattern = patterning.get_pattern("Horseshoe"),
    )
    ]
    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMillingWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()