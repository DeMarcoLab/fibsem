
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (FibsemPattern, FibsemPatternSettings,
                               MicroscopeSettings)
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.qtdesigner_files import FibsemMillingWidget
from fibsem.ui.utils import _draw_patterns_in_napari


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

        print("setup_connections")

        self.pushButton.clicked.connect(self.update_ui)

        self.comboBox_pattern.addItems([pattern.name for pattern in FibsemPattern])
        self.comboBox_milling_current.addItems([current for current in ["2.0e-9", "7.4e-9" ]]) # TODO: fix

    def get_pattern_settings_from_ui(self):
        print("get pattern settings")
        
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
        )
    
        return pattern_settings

    def update_ui(self):

        print("update ui")

        pattern_settings = self.get_pattern_settings_from_ui()

        # clear patterns then draw new ones
        _draw_patterns_in_napari(self.viewer, 
            ib_image=self.image_widget.ib_image, 
            eb_image=self.image_widget.eb_image, 
            all_patterns=[[pattern_settings]])

        self.viewer.layers.selection.active = self.image_widget.eb_layer

def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemMillingWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
