import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem.microscope import FibsemMicroscope
from fibsem import constants, acquire
from fibsem.structures import BeamType, ImageSettings, FibsemImage
from fibsem.ui import utils as ui_utils 
from fibsem.ui.qtdesigner_files import ImageSettingsWidget

import numpy as np
from pathlib import Path


class FibsemImageSettingsWidget(ImageSettingsWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        image_settings: ImageSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemImageSettingsWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope
        self.viewer = viewer
        self.eb_layer, self.ib_layer = None, None

        if self.microscope is not None:
            self.ui_detector()

        self.setup_connections()
    
        if image_settings is not None:
            self.set_ui_from_settings(image_settings)

        # register initial images
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ION.name)
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ELECTRON.name)

    def setup_connections(self):

        # set ui elements
        self.comboBox_image_beam_type.addItems([beam.name for beam in BeamType])
        
        self.pushButton_take_image.clicked.connect(self.take_image)
        self.pushButton_take_all_images.clicked.connect(self.take_reference_images)
        self.checkBox_image_save_image.toggled.connect(self.update_ui)
        self.type.currentTextChanged.connect(self.select_detector_type)
        self.mode.currentTextChanged.connect(self.select_detector_mode)
        self.brightness.valueChanged.connect(self.update_brightness)
        self.contrast.valueChanged.connect(self.update_contrast)
        self.beam_detector.currentTextChanged.connect(self.ui_detector)

    def select_detector_type(self):
        beam = BeamType(self.beam_detector.currentIndex()+1)
        self.microscope.set("detector_type", self.type.currentText(), beam_type=beam)

    def select_detector_mode(self):
        beam = BeamType(self.beam_detector.currentIndex()+1)
        self.microscope.setI("detector_mode", self.mode.currentText(), beam_type=beam)
        
    def update_brightness(self):
        beam = BeamType(self.beam_detector.currentIndex()+1)
        self.microscope.set("detector_brightness", self.brightness.value()/100, beam_type=beam)

    def update_contrast(self):
        beam = BeamType(self.beam_detector.currentIndex()+1)
        self.microscope.set("detector_contrast", self.contrast.value()/100, beam_type=beam)

    def set_ui_from_settings(self, image_settings: ImageSettings):

        self.spinBox_resolution_x.setValue(image_settings.resolution[0])
        self.spinBox_resolution_y.setValue(image_settings.resolution[1])
        self.doubleSpinBox_image_dwell_time.setValue(image_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_image_hfw.setValue(image_settings.hfw * constants.SI_TO_MICRO)
        self.comboBox_image_beam_type.setCurrentText(image_settings.beam_type.name)
        self.checkBox_image_use_autocontrast.setChecked(image_settings.autocontrast)
        self.checkBox_image_use_autogamma.setChecked(image_settings.gamma_enabled)
        self.checkBox_image_save_image.setChecked(image_settings.save)
        self.lineEdit_image_path.setText(image_settings.save_path)
        self.lineEdit_image_label.setText(image_settings.label)

    def update_ui(self):

        self.label_image_save_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.label_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        
    def ui_detector(self):
        self.detector_type_ion = self.microscope.get_available_values("detector_type", beam_type=BeamType.ION)  
        for i in range(len(self.detector_type_ion)):
            self.type.addItem(self.detector_type_ion[i-1]) 
        self.detector_type_electron = self.microscope.get_available_values("detector_type", beam_type=BeamType.ELECTRON)
        for i in range(len(self.detector_type_electron)):
            self.type.addItem(self.detector_type_electron[i-1])
        self.detector_mode_ion = self.microscope.get_available_values("detector_mode", beam_type=BeamType.ION)
        if self.detector_mode_ion is not None:
            for i in range(len(self.detector_mode_ion)):
                self.mode.addItem(self.detector_mode_ion[i-1])
        else:
            self.mode.addItem("N/A")
        self.detector_mode_electron = self.microscope.get_available_values("detector_mode", beam_type=BeamType.ELECTRON)
        if self.detector_mode_electron is not None:
            for i in range(len(self.detector_mode_electron)):
                self.mode.addItem(self.detector_mode_electron[i-1])
        else:
            self.mode.addItem("N/A")

        self.microscope.set("detector_brightness", 0.5, beam_type=BeamType.ION)
        self.microscope.set("detector_contrast", 0.5, beam_type=BeamType.ION)
        
    def get_settings_from_ui(self):

        self.image_settings = ImageSettings(
            resolution=[self.spinBox_resolution_x.value(), self.spinBox_resolution_y.value()],
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            beam_type=BeamType[self.comboBox_image_beam_type.currentText()],
            autocontrast=self.checkBox_image_use_autocontrast.isChecked(),
            gamma_enabled=self.checkBox_image_use_autogamma.isChecked(),
            save=self.checkBox_image_save_image.isChecked(),
            save_path=Path(self.lineEdit_image_path.text()),
            label=self.lineEdit_image_label.text()
            
        )

        return self.image_settings

    def take_image(self):
        self.image_settings: ImageSettings = self.get_settings_from_ui()

        arr =  acquire.new_image(self.microscope, self.image_settings)
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
       
        arr = ui_utils._draw_crosshair(arr)

        try:
            self.viewer.layers[name].data = arr
        except:    
            layer = self.viewer.add_image(arr, name = name)
        

        layer = self.viewer.layers[name]
        if self.eb_layer is None and name == BeamType.ELECTRON.name:
            self.eb_layer = layer
        if self.ib_layer is None and name == BeamType.ION.name:
            self.ib_layer = layer
        

        # centre the camera
        if self.eb_layer:
            self.viewer.camera.center = [
                0.0,
                self.eb_layer.data.shape[0] / 2,
                self.eb_layer.data.shape[1],
            ]
            self.viewer.camera.zoom = 0.45

        if self.ib_layer:
            self.ib_layer.translate = [0.0, arr.shape[1]]        
        self.viewer.layers.selection.active = self.eb_layer

        if self.eb_layer:
            points = np.array([[-20, 200], [-20, self.eb_layer.data.shape[1] + 150]])
            string = ["ELECTRON BEAM", "ION BEAM"]
            text = {
                "string": string,
                "color": "white"
            }

            try:
                self.viewer.layers['label'].data = points
            except:    
                self.viewer.add_points(
                points,
                name="label",
                text=text,
                size=20,
                edge_width=7,
                edge_width_is_relative=False,
                edge_color='transparent',
                face_color='transparent',
                )   


        

    def get_data_from_coord(self, coords: tuple) -> tuple:
        # check inside image dimensions, (y, x)
        eb_shape = self.eb_image.data.shape[0], self.eb_image.data.shape[1]
        ib_shape = self.ib_image.data.shape[0], self.ib_image.data.shape[1] + self.eb_image.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (
            coords[1] > 0 and coords[1] < eb_shape[1]
        ):
            image = self.eb_image
            beam_type = BeamType.ELECTRON
            # print("electron")

        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (
            coords[1] > eb_shape[0] and coords[1] < ib_shape[1]
        ):
            image = self.ib_image
            coords = (coords[0], coords[1] - ib_shape[1] // 2)
            beam_type = BeamType.ION
            # print("ion")
        else:
            beam_type, image = None, None

        return coords, beam_type, image
    
    def closeEvent(self, event):
        self.viewer.layers.clear()
        event.accept()

def main():

    image_settings = ImageSettings(resolution=[1536, 1024], dwell_time=1e-6, hfw=150e-6, 
    autocontrast=True, beam_type=BeamType.ION, 
    save=True, label="my_label", save_path="path/to/save", 
    gamma_enabled=True)


    viewer = napari.Viewer(ndisplay=2)
    image_settings_ui = FibsemImageSettingsWidget(image_settings=image_settings)
    viewer.window.add_dock_widget(
        image_settings_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
