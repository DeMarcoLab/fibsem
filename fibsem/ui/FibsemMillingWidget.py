
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
from fibsem import milling


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

    def update_ui(self, pattern_settings: FibsemPatternSettings = None):
        
        if self.sender() == self.pushButton:
            pattern_settings = self.get_pattern_settings_from_ui()

        # clear patterns then draw new ones
        _draw_patterns_in_napari(self.viewer, 
            ib_image=self.image_widget.ib_image, 
            eb_image=self.image_widget.eb_image, 
            all_patterns=[[pattern_settings]])

        self.viewer.layers.selection.active = self.image_widget.eb_layer

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
