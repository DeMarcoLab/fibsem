import napari
import napari.utils.notifications
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from PyQt5 import QtWidgets

from fibsem import constants, acquire
from fibsem.structures import BeamType, GammaSettings, ImageSettings
from fibsem.ui.qtdesigner_files import ImageSettingsWidget

import numpy as np


class FibsemImageSettingsWidget(ImageSettingsWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
        image_settings: ImageSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemImageSettingsWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.viewer = viewer
        self.eb_layer, self.ib_layer = None, None

        self.setup_connections()
    
        if image_settings is not None:
            self.set_ui_from_settings(image_settings)

        # register initial images
        self.take_reference_images()

    def setup_connections(self):

        # set ui elements
        self.comboBox_image_beam_type.addItems([beam.name for beam in BeamType])
        
        resolutions = self.microscope.beams.electron_beam.scanning.resolution.available_values
        self.comboBox_image_resolution.addItems(resolutions)

        self.pushButton_take_image.clicked.connect(self.take_image)

        self.checkBox_image_save_image.toggled.connect(self.update_ui)



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

    def update_ui(self):

        self.label_image_save_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.label_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        
    def get_settings_from_ui(self):

        self.image_settings = ImageSettings(
            resolution=self.comboBox_image_resolution.currentText(),
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            beam_type=BeamType[self.comboBox_image_beam_type.currentText()],
            autocontrast=self.checkBox_image_use_autocontrast.isChecked(),
            gamma=GammaSettings(
                enabled=self.checkBox_image_use_autogamma.isChecked(),
                min_gamma = 0.15,
                max_gamma = 1.8,
                scale_factor = 0.01,
                threshold = 46 # px
            ),
            save=self.checkBox_image_save_image.isChecked(),
            save_path=self.lineEdit_image_path.text(),
            label=self.lineEdit_image_label.text()
            
        )

        return self.image_settings

    def take_image(self):
        self.image_settings = self.get_settings_from_ui()

        arr =  acquire.new_image(self.microscope, self.image_settings)
        # arr = np.random.random((1024, 1536))

        name = f"{self.image_settings.beam_type.name}"

        if self.image_settings.beam_type == BeamType.ELECTRON:
            self.eb_image = arr
        if self.image_settings.beam_type == BeamType.ION:
            self.ib_image = arr

        self.update_viewer(arr.data, name)

    def take_reference_images(self):

        self.image_settings = self.get_settings_from_ui()

        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_viewer(self.ib_image.data, BeamType.ION.name)
        self.update_viewer(self.eb_image.data, BeamType.ELECTRON.name)


    def update_viewer(self, arr: np.ndarray, name: str):
       
        try:
            self.viewer.layers[name].data = arr
        except:    
            layer = self.viewer.add_image(arr, name = name)
            
            if self.eb_layer is None and name == BeamType.ELECTRON.name:
                self.eb_layer = layer
            if self.ib_layer is None and name == BeamType.ION.name:
                self.ib_layer = layer



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
