
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem import constants
from fibsem.microscope import (DemoMicroscope, FibsemMicroscope,
                               TescanMicroscope, ThermoMicroscope)
from fibsem.structures import (BeamType, FibsemMillingSettings, FibsemPattern,
                               FibsemPatternSettings, MicroscopeSettings)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMillingWidget
from fibsem.ui.utils import _draw_patterns_in_napari
from fibsem import milling, patterning
import logging
from fibsem.structures import Point


class FibsemMillingWidget(FibsemMillingWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        image_widget: FibsemImageSettingsWidget = None, 
        parent=None,
    ):
        super(FibsemMillingWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.image_widget = image_widget

        self.setup_connections()

        self.pattern_update()

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
        self.scan_direction.addItems(self.microscope.get_scan_directions())
        

        direction_list = self.microscope.get_scan_directions()
        for i in range(len(direction_list)-1):
            self.scan_direction.addItem(direction_list[i-1])
        
        # patterns
        self.comboBox_pattern.addItems([pattern.name for pattern in FibsemPattern])

        # register mouse callbacks
        self.image_widget.eb_layer.mouse_drag_callbacks.append(self._single_click)
        self.image_widget.ib_layer.mouse_drag_callbacks.append(self._single_click)


        # new patterns
        self.comboBox_patterns.addItems([pattern.name for pattern in patterning.__PATTERNS__])
        self.comboBox_patterns.currentIndexChanged.connect(self.pattern_update)
        

    def pattern_update(self):

        # get current pattern
        pattern = patterning.__PATTERNS__[self.comboBox_patterns.currentIndex()]

        logging.info(f"Selected pattern: {pattern.name}")
        logging.info(f"Required parameters: {pattern.required_keys}")

        # create a label and double spinbox for each required keys and add it to the layout

        # clear layout
        for i in reversed(range(self.gridLayout_patterns.count())):
            self.gridLayout_patterns.itemAt(i).widget().setParent(None)

        # add new widgets
        # TODO: smarter logic for which widgets to add
        for i, key in enumerate(pattern.required_keys):
            label = QtWidgets.QLabel(key)
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.001)
            spinbox.setRange(0, 1000)
            spinbox.setValue(0)
            self.gridLayout_patterns.addWidget(label, i, 0)
            self.gridLayout_patterns.addWidget(spinbox, i, 1)


    def get_pattern_from_ui(self):

        pattern_dict = {}

        # get current pattern
        pattern = patterning.__PATTERNS__[self.comboBox_patterns.currentIndex()]

        for i, key in enumerate(pattern.required_keys):
            spinbox = self.gridLayout_patterns.itemAtPosition(i, 1).widget()
            
            value = spinbox.value() * constants.MICRO_TO_SI if key not in ["rotation", "size_ratio", "scan_direction", "cleaning_cross_section", "number"] else spinbox.value()
            pattern_dict[key] = value # TODO: not everythign is in microns

        # define pattern
        pattern: patterning.BasePattern = pattern()
        point = Point(x=self.doubleSpinBox_centre_x.value() * constants.MICRO_TO_SI, y=self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI)
        pattern_settings_list = pattern.define(protocol=pattern_dict, point=point)
        
        return pattern_settings_list

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
        from fibsem import conversions
        from fibsem.structures import Point
        point = conversions.image_to_microscope_image_coordinates(
            Point(x=coords[1], y=coords[0]), image.data, image.metadata.pixel_size.x,
        )
        logging.info(f"Moved pattern to {point}")

        # update ui
        self.doubleSpinBox_centre_x.setValue(point.x * constants.SI_TO_MICRO)
        self.doubleSpinBox_centre_y.setValue(point.y * constants.SI_TO_MICRO)
        # pattern_settings = self.get_pattern_settings_from_ui()   
        self.update_ui()

    def get_pattern_settings_from_ui(self):
        
        pattern_settings = FibsemPatternSettings(
            pattern=FibsemPattern[self.comboBox_pattern.currentText()],
            centre_x=self.doubleSpinBox_centre_x.value() * constants.MICRO_TO_SI,
            centre_y=self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI,
            width=self.doubleSpinBox_width.value() * constants.MICRO_TO_SI,
            height=self.doubleSpinBox_height.value() * constants.MICRO_TO_SI,
            depth=self.doubleSpinBox_depth.value() * constants.MICRO_TO_SI,
            start_x = (self.doubleSpinBox_centre_x.value() - (self.doubleSpinBox_width.value() / 2)) * constants.MICRO_TO_SI,
            start_y = self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI,
            end_x = (self.doubleSpinBox_centre_x.value() + (self.doubleSpinBox_width.value() / 2)) * constants.MICRO_TO_SI,
            end_y = self.doubleSpinBox_centre_y.value() * constants.MICRO_TO_SI,
            scan_direction=self.scan_direction.currentText(),
        )
    
        return pattern_settings
    

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

    def update_ui(self, pattern_settings: list[FibsemPatternSettings] = None):
        
        if self.sender() == self.pushButton or pattern_settings is None:
            pattern_settings = self.get_pattern_from_ui()

        # should be a list for plotting
        if isinstance(pattern_settings, FibsemPatternSettings):
            pattern_settings = [pattern_settings]

        # clear patterns then draw new ones
        _draw_patterns_in_napari(self.viewer, 
            ib_image=self.image_widget.ib_image, 
            eb_image=self.image_widget.eb_image, 
            all_patterns=[pattern_settings])

        self.viewer.layers.selection.active = self.image_widget.eb_layer




        # if self.sender() == self.pushButton:
        #     pattern_settings = self.get_pattern_settings_from_ui()

        # # clear patterns then draw new ones
        # _draw_patterns_in_napari(self.viewer, 
        #     ib_image=self.image_widget.ib_image, 
        #     eb_image=self.image_widget.eb_image, 
        #     all_patterns=[[pattern_settings]])

        # self.viewer.layers.selection.active = self.image_widget.eb_layer

    def run_milling(self):
        
        # TODO: thread this
        pattern_settings = self.get_pattern_settings_from_ui()
        milling_settings = self.get_milling_settings_from_ui()

        milling.setup_milling(self.microscope, mill_settings=milling_settings)
        milling.draw_pattern(self.microscope, pattern_settings)

        milling.run_milling(self.microscope, milling_settings.milling_current)
        milling.finish_milling(self.microscope, self.settings.system.ion.current)

        napari.utils.notifications.show_info("Milling complete.")

        self.run_milling_finished()

    def run_milling_finished(self):

        # take new images and update ui
        self.image_widget.take_reference_images()
        pattern_settings = self.get_pattern_settings_from_ui()
        self.update_ui(pattern_settings)


def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMillingWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
