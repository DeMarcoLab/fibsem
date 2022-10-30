import os


import fibsem
import napari
import napari.utils.notifications
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import calibration, constants, utils
from fibsem.structures import (
    BeamType,
    GammaSettings,
    ImageSettings,
)
from fibsem.ui.qtdesigner_files import ImageSettingsWidget
from PyQt5 import QtWidgets

BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")


class FibsemImageSettingsWidget(ImageSettingsWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
        image_settings: ImageSettings = None,
        parent=None,
    ):
        super(FibsemImageSettingsWidget, self).__init__(parent=parent)
        self.setupUi(self)


        self.microscope = microscope

        self.setup_connections()
    
        if image_settings is not None:
            self.set_ui_from_settings(image_settings)


    def setup_connections(self):

        # set ui elements
        self.comboBox_image_beam_type.addItems([beam.name for beam in BeamType])
        
        # resolutions = self.microscope.beams.electron_beam.scanning.resolution.available_values
        resolutions = ["1536x1024", "3072x2048", "6144x4096"]
        self.comboBox_image_resolution.addItems(resolutions)

        self.pushButton_save.clicked.connect(self.get_settings_from_ui)

    def set_ui_from_settings(self, image_settings: ImageSettings):

        self.comboBox_image_resolution.setCurrentText(image_settings.resolution)
        self.doubleSpinBox_image_dwell_time.setValue(image_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_image_hfw.setValue(image_settings.hfw * constants.SI_TO_MICRO)
        self.comboBox_image_beam_type.setCurrentText(image_settings.beam_type.name)
        self.checkBox_image_use_autocontrast.setChecked(image_settings.autocontrast)
        self.checkBox_image_use_autogamma.setChecked(image_settings.gamma.enabled)
        self.checkBox_image_save_image.setChecked(image_settings.save)
        self.lineEdit_image_path.setText(image_settings.save_path)
        self.lineEdit_image_label.setText(image_settings.label)

    def get_settings_from_ui(self):

        self.image_settings = ImageSettings(
            resolution=self.comboBox_image_resolution.currentText(),
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            beam_type=BeamType[self.comboBox_image_beam_type.currentText()],
            autocontrast=self.checkBox_image_use_autocontrast.isChecked(),
            gamma=GammaSettings(
                enabled=self.checkBox_image_use_autogamma.isChecked()
            ),
            save=self.checkBox_image_save_image.isChecked(),
            save_path=self.lineEdit_image_path.text(),
            label=self.lineEdit_image_label.text()
            
        )


        print(self.image_settings)


def main():

    image_settings = ImageSettings(resolution="1536x1024", dwell_time=1e-6, hfw=150e-6, 
    autocontrast=True, beam_type=BeamType.ION, 
    save=True, label="my_label", save_path="path/to/save", 
    gamma=GammaSettings(enabled=True))


    viewer = napari.Viewer(ndisplay=2)
    image_settings_ui = FibsemImageSettingsWidget(image_settings=image_settings)
    viewer.window.add_dock_widget(
        image_settings_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
