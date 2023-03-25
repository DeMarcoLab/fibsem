
import logging

import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem import config as cfg
from fibsem import constants, conversions, milling, patterning, utils
from fibsem.microscope import (DemoMicroscope, FibsemMicroscope,
                               TescanMicroscope, ThermoMicroscope)
from fibsem.patterning import FibsemMillingStage
from fibsem.structures import (BeamType, FibsemMillingSettings,
                               FibsemPatternSettings, MicroscopeSettings,
                               Point)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMillingWidget
from fibsem.ui.utils import _draw_patterns_in_napari
from napari.qt.threading import thread_worker

_UNSCALED_VALUES  = ["rotation", "size_ratio", "scan_direction", "cleaning_cross_section", "number"]
def _scale_value(key, value, scale):
    if key not in _UNSCALED_VALUES:
        return value * scale
    return value

class FibsemMillingWidget(FibsemMillingWidget.Ui_Form, QtWidgets.QWidget):
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

    def setup_connections(self):

        self.pushButton.clicked.connect(self.update_ui)
        self.pushButton_run_milling.clicked.connect(self.run_milling)

        # milling
        available_currents = self.microscope.get_available_values("current", BeamType.ION)
        self.comboBox_milling_current.addItems([str(current) for current in available_currents])

        _THERMO = isinstance(self.microscope, ThermoMicroscope)
        _TESCAN = isinstance(self.microscope, TescanMicroscope)

        if isinstance(self.microscope, DemoMicroscope):
            _THERMO, _TESCAN = True, True
        
        # THERMO 
        self.label_application_file.setVisible(_THERMO)
        self.comboBox_application_file.setVisible(_THERMO)
        available_application_files = self.microscope.get_available_values("application_file")
        self.comboBox_application_file.addItems(available_application_files)
        
        # TESCAN
        self.label_rate.setVisible(_TESCAN)
        self.label_spot_size.setVisible(_TESCAN)
        self.label_dwell_time.setVisible(_TESCAN)
        self.doubleSpinBox_rate.setVisible(_TESCAN)
        self.doubleSpinBox_spot_size.setVisible(_TESCAN)
        self.doubleSpinBox_dwell_time.setVisible(_TESCAN)       

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_drag_callbacks.append(self._single_click)
        self.image_widget.ib_layer.mouse_drag_callbacks.append(self._single_click)

        # new patterns
        self.comboBox_patterns.addItems([pattern.name for pattern in patterning.__PATTERNS__])
        self.comboBox_patterns.currentIndexChanged.connect(self.update_pattern_ui)
    
        # milling stages
        self.pushButton_add_milling_stage.clicked.connect(self.add_milling_stage)
        self.pushButton_add_milling_stage.setStyleSheet("background-color: green; color: white;")
        self.pushButton_remove_milling_stage.clicked.connect(self.remove_milling_stage)
        self.pushButton_remove_milling_stage.setStyleSheet("background-color: red; color: white;")
        
        self.pushButton_save_milling_stage.clicked.connect(self.update_milling_stage_from_ui)
        self.pushButton_save_milling_stage.setStyleSheet("background-color: blue; color: white;")

        self.pushButton_test_button.clicked.connect(self.test_function)

        # add one milling stage by default
        if len(self.milling_stages) == 0:
            self.add_milling_stage()
        else:
            self.comboBox_milling_stage.addItems([stage.name for stage in self.milling_stages])
        self.update_milling_stage_ui()
        self.comboBox_milling_stage.currentIndexChanged.connect(self.update_milling_stage_ui)


    def add_milling_stage(self):
        logging.info("Adding milling stage")

        num = len(self.milling_stages) + 1
        name = f"Milling Stage {num}"
        milling_stage = FibsemMillingStage(name=name, num=num)
        self.milling_stages.append(milling_stage)
        self.comboBox_milling_stage.addItem(name)
        napari.utils.notifications.show_info(f"Added {name}.")

    def remove_milling_stage(self):
        logging.info("Removing milling stage")

        current_index = self.comboBox_milling_stage.currentIndex()
        self.comboBox_milling_stage.removeItem(current_index)
        self.milling_stages.pop(current_index)
        napari.utils.notifications.show_info(f"Removed milling stage.")

    def test_function(self):
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
        self.set_milling_stages(millings_stages)

    def set_milling_stages(self, milling_stages: list[FibsemMillingStage]) -> None:

        self.milling_stages = milling_stages
        self.comboBox_milling_stage.clear()
        self.comboBox_milling_stage.addItems([stage.name for stage in self.milling_stages])
        self.update_milling_stage_ui()

    def get_milling_stages(self):
        return self.milling_stages

    def get_point_from_ui(self):

        point = Point(x=self.doubleSpinBox_centre_x.value() * constants.MICRO_TO_SI, 
                      y=self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI)

        return point


    def update_milling_stage_ui(self):

        # get the selected milling stage
        current_index = self.comboBox_milling_stage.currentIndex()
        milling_stage: FibsemMillingStage = self.milling_stages[current_index]

        # set the milling settings
        self.set_milling_settings_ui(milling_stage.milling)

        # set the pattern (and triggers the pattern settings)
        self.comboBox_patterns.setCurrentText(milling_stage.pattern.name)


    def update_milling_stage_from_ui(self):

        # get current milling stage
        current_index = self.comboBox_milling_stage.currentIndex()
        milling_stage = self.milling_stages[current_index]

        # update milling settings
        milling_stage.milling = self.get_milling_settings_from_ui()

        # update pattern and define
        milling_stage.pattern = self.get_pattern_from_ui_v2()

        # update point
        milling_stage.point = self.get_point_from_ui()

        napari.utils.notifications.show_info(f"Updated {milling_stage.name}.")
        return 

    def update_pattern_ui(self):

        # get current pattern
        pattern = patterning.__PATTERNS__[self.comboBox_patterns.currentIndex()]

        logging.info(f"Selected pattern: {pattern.name}")
        logging.info(f"Required parameters: {pattern.required_keys}")

        # create a label and double spinbox for each required keys and add it to the layout

        # clear layout
        for i in reversed(range(self.gridLayout_patterns.count())):
            self.gridLayout_patterns.itemAt(i).widget().setParent(None)

        pattern_protocol = self.protocol["patterns"][pattern.name]

        # add new widgets
        # TODO: smarter logic for which kinds of widgets to add
        for i, key in enumerate(pattern.required_keys):
            label = QtWidgets.QLabel(key)
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.001)
            spinbox.setRange(0, 1000)
            spinbox.setValue(0)
            self.gridLayout_patterns.addWidget(label, i, 0)
            self.gridLayout_patterns.addWidget(spinbox, i, 1)

            # get default values from self.protocol and set values
            if key in pattern_protocol:
                value = _scale_value(key, pattern_protocol[key], constants.SI_TO_MICRO)
                spinbox.setValue(value)

    def get_pattern_from_ui_v2(self):

        # get current pattern
        pattern = patterning.get_pattern(self.comboBox_patterns.currentText())

        # get pattern protocol from ui
        pattern_dict = {}
        for i, key in enumerate(pattern.required_keys):
            spinbox = self.gridLayout_patterns.itemAtPosition(i, 1).widget()
            value = _scale_value(key, spinbox.value(), constants.MICRO_TO_SI)
            pattern_dict[key] = value # TODO: not everythign is in microns

        # define pattern
        point = Point(x=self.doubleSpinBox_centre_x.value() * constants.MICRO_TO_SI, y=self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI)
        pattern.define(protocol=pattern_dict, point=point)
        
        return pattern


    def _single_click(self, layer, event):
        """Callback for single click on image layer."""
        if event.button != 2:
            return

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        coords, beam_type, image = self.image_widget.get_data_from_coord(coords)
        
        if beam_type is not BeamType.ION:
            napari.utils.notifications.show_info(
                f"Please right click on the {BeamType.ION.name} image to move pattern."
            )
            return

        # only move the pattern if milling widget is activate and beamtype is ion?

        # update pattern

        point = conversions.image_to_microscope_image_coordinates(
            Point(x=coords[1], y=coords[0]), image.data, image.metadata.pixel_size.x,
        )
        logging.info(f"Moved pattern to {point}")

        # update ui
        self.doubleSpinBox_centre_x.setValue(point.x * constants.SI_TO_MICRO)
        self.doubleSpinBox_centre_y.setValue(point.y * constants.SI_TO_MICRO)
        
        self.update_milling_stage_from_ui()
        
        self.update_ui()
   
    def set_milling_settings_ui(self, milling: FibsemMillingSettings) -> None:

        self.comboBox_milling_current.setCurrentText(str(milling.milling_current))
        self.comboBox_application_file.setCurrentText(milling.application_file)
        self.doubleSpinBox_rate.setValue(milling.rate)
        self.doubleSpinBox_dwell_time.setValue(milling.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_spot_size.setValue(milling.spot_size * constants.SI_TO_MICRO)
        self.doubleSpinBox_hfw.setValue(milling.hfw * constants.SI_TO_MICRO)

    def get_milling_settings_from_ui(self):

        milling_settings = FibsemMillingSettings(
            milling_current=float(self.comboBox_milling_current.currentText()),
            application_file=self.comboBox_application_file.currentText(),
            rate=self.doubleSpinBox_rate.value(),
            dwell_time = self.doubleSpinBox_dwell_time.value() * constants.MICRO_TO_SI,
            spot_size=self.doubleSpinBox_spot_size.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_hfw.value() * constants.MICRO_TO_SI,

        )

        return milling_settings

    def update_ui(self, milling_stages: list[FibsemMillingStage] = None):
        
        if self.sender() is self.pushButton or milling_stages is None:
            milling_stages = self.get_milling_stages()

        # make milling stage a list if it is not
        if not isinstance(milling_stages, list):
            milling_stages = [milling_stages]

        # get all patterns (2D list, one list of pattern settings per stage)
        patterns: list[list[FibsemPatternSettings]] = [stage.pattern.patterns for stage in milling_stages if stage.pattern is not None]
        
        try:
            # clear patterns then draw new ones
            _draw_patterns_in_napari(self.viewer, 
                ib_image=self.image_widget.ib_image, 
                eb_image=self.image_widget.eb_image, 
                all_patterns=patterns,) # TODO: add names and legend for this
            
        except Exception as e:
            napari.utils.notifications.show_error(f"Error drawing patterns: {e}")
            logging.error(e)
            return
        
        self.viewer.layers.selection.active = self.image_widget.eb_layer

    def _toggle_interaction(self, enabled: bool = True):

        """Toggle microscope and pattern interactions."""

        self.pushButton.setEnabled(enabled)
        self.pushButton_add_milling_stage.setEnabled(enabled)
        self.pushButton_remove_milling_stage.setEnabled(enabled)
        self.pushButton_save_milling_stage.setEnabled(enabled)
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

        self._toggle_interaction(enabled=False)
        for stage in self.milling_stages:
            yield f"Preparing: {stage.name}"
            if stage.pattern is not None:

                import time, random
                time.sleep(random.randint(1, 2))
                
                milling.setup_milling(self.microscope, mill_settings=stage.milling)

                milling.draw_patterns(self.microscope, stage.pattern.patterns)

                yield f"Running {stage.name}..."
                milling.run_milling(self.microscope, stage.milling.milling_current)
                
                time.sleep(random.randint(1, 5))

                milling.finish_milling(self.microscope, self.settings.system.ion.current)

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


def main():



    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMillingWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
