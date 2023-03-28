import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

from fibsem.microscope import FibsemMicroscope
from fibsem import constants, acquire

from fibsem.structures import BeamType, ImageSettings, FibsemImage, Point, FibsemDetectorSettings, BeamSettings
from fibsem.ui import utils as ui_utils 
import utils as ui_utils 

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

        self.setup_connections()

        if image_settings is not None:
            self.detector_settings = self.get_detector_settings()
            self.beam_settings = self.get_beam_settings()
            self.image_settings = image_settings
            # self.set_ui_from_settings(image_settings = image_settings, beam_settings= self.beam_settings, detector_settings= self.detector_settings, beam_type = BeamType.ELECTRON)
        self.update_detector_ui()

        # register initial images
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ION.name)
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ELECTRON.name)

    def setup_connections(self):

        # set ui elements

        self.selected_beam.addItems([beam.name for beam in BeamType])

        self.pushButton_take_image.clicked.connect(self.take_image)
        self.pushButton_take_all_images.clicked.connect(self.take_reference_images)
        self.checkBox_image_save_image.toggled.connect(self.update_ui)
        self.set_detector_button.clicked.connect(self.select_detector)
        self.selected_beam.currentTextChanged.connect(self.update_detector_ui)
        self.button_set_beam_settings.clicked.connect(self.update_beam_settings)
        self.detector_contrast_slider.valueChanged.connect(self.update_labels)
        self.detector_brightness_slider.valueChanged.connect(self.update_labels)
    
    def update_labels(self):
        self.detector_contrast_label.setText(f"{self.detector_contrast_slider.value()}%")
        self.detector_brightness_label.setText(f"{self.detector_brightness_slider.value()}%")

    def select_detector(self):
        beam =  BeamType[self.selected_beam.currentText()]
        self.microscope.set("detector_type", self.detector_type_combobox.currentText(), beam_type=beam)
        self.microscope.set("detector_mode", self.detector_mode_combobox.currentText(), beam_type=beam)
        self.microscope.set("detector_brightness", self.detector_brightness_slider.value()*constants.FROM_PERCENTAGES, beam_type=beam)
        self.detector_brightness_label.setText(f"{self.detector_brightness_slider.value()}%")
        self.microscope.set("detector_contrast", self.detector_contrast_slider.value()*constants.FROM_PERCENTAGES, beam_type=beam)
        self.detector_contrast_label.setText(f"{self.detector_contrast_slider.value()}%")

    def update_beam_settings(self):
        beam = BeamType[self.selected_beam.currentText()]
        self.microscope.set("working_distance", self.working_distance.value()*constants.MILLIMETRE_TO_METRE, beam_type=beam)
        self.microscope.set("current", self.beam_current.value()*constants.PICO_TO_SI, beam_type=beam)
        self.microscope.set("voltage", self.beam_voltage.value()*constants.KILO_TO_SI, beam_type=beam)
        self.microscope.set("stigmation", Point(self.stigmation_x.value(), self.stigmation_y.value()), beam_type=beam)
        self.microscope.set("shift", Point(self.shift_x.value(), self.shift_y.value()), beam_type=beam)

    def set_ui_from_settings(self, image_settings: ImageSettings, beam_settings: BeamSettings, detector_settings: FibsemDetectorSettings, beam_type: BeamType):

        self.spinBox_resolution_x.setValue(image_settings.resolution[0])
        self.spinBox_resolution_y.setValue(image_settings.resolution[1])


        self.doubleSpinBox_image_dwell_time.setValue(image_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_image_hfw.setValue(image_settings.hfw * constants.SI_TO_MICRO)
        self.selected_beam.setCurrentText(image_settings.beam_type.name)

        self.checkBox_image_use_autocontrast.setChecked(image_settings.autocontrast)
        self.checkBox_image_use_autogamma.setChecked(image_settings.gamma_enabled)
        self.checkBox_image_save_image.setChecked(image_settings.save)
        self.lineEdit_image_path.setText(str(image_settings.save_path))
        self.lineEdit_image_label.setText(image_settings.label)

        self.detector_contrast_slider.setValue(detector_settings.contrast*100)
        self.detector_brightness_slider.setValue(detector_settings.brightness*100)
        self.beam_current.setValue(beam_settings.beam_current*constants.SI_TO_PICO)
        self.beam_voltage.setValue(beam_settings.voltage*constants.SI_TO_KILO)
        self.working_distance.setValue(beam_settings.working_distance*constants.METRE_TO_MILLIMETRE if beam_settings.working_distance is not None else None)
        if beam_settings.shift is not None:
            self.shift_x.setValue(beam_settings.shift.x)
            self.shift_y.setValue(beam_settings.shift.y)
        if beam_settings.stigmation is not None:
            self.stigmation_x.setValue(beam_settings.stigmation.x)
            self.stigmation_y.setValue(beam_settings.stigmation.y)

    def update_ui(self):

        self.label_image_save_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.label_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        
  
    def update_detector_ui(self):
        beam_type = BeamType[self.selected_beam.currentText()]
        self.detector_type = self.microscope.get_available_values("detector_type", beam_type=beam_type)
        self.detector_type_combobox.clear()
        self.detector_type_combobox.addItems(self.detector_type)
        self.detector_type_combobox.setCurrentText(self.microscope.get("detector_type", beam_type=beam_type))
        
        self.detector_mode = self.microscope.get_available_values("detector_mode", beam_type=beam_type)
        self.detector_mode_combobox.clear()
        self.detector_mode_combobox.addItems(self.detector_mode if self.detector_mode is not None else ["N/A"])# if self.detector_mode is not None  else self.mode.addItem("N/A")
        self.detector_mode_combobox.setCurrentText(self.microscope.get("detector_mode", beam_type=beam_type))

        self.microscope.set("detector_brightness", self.detector_brightness_slider.value()*constants.FROM_PERCENTAGES, beam_type=beam_type)
        self.microscope.set("detector_contrast", self.detector_contrast_slider.value()*constants.FROM_PERCENTAGES, beam_type=beam_type)


    def get_settings_from_ui(self):

        self.image_settings = ImageSettings(
            resolution=[self.spinBox_resolution_x.value(), self.spinBox_resolution_y.value()],
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            beam_type=BeamType[self.selected_beam.currentText()],
            autocontrast=self.checkBox_image_use_autocontrast.isChecked(),
            gamma_enabled=self.checkBox_image_use_autogamma.isChecked(),
            save=self.checkBox_image_save_image.isChecked(),
            save_path=Path(self.lineEdit_image_path.text()),
            label=self.lineEdit_image_label.text()
            
        )

        self.detector_settings = FibsemDetectorSettings(
            type=self.detector_type_combobox.currentText(),
            mode=self.detector_mode_combobox.currentText(),
            brightness=self.detector_brightness_slider.value(),
            contrast=self.detector_contrast_slider.value()
        )

        self.beam_settings = BeamSettings(
            beam_type=BeamType[self.selected_beam.currentText()],
            working_distance=self.working_distance.value(),
            beam_current=self.beam_current.value(),
            voltage=self.beam_voltage.value(),
            hfw = self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            resolution=[self.spinBox_resolution_x.value(), self.spinBox_resolution_y.value()],
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            stigmation = Point(self.stigmation_x.value(), self.stigmation_y.value()),
            shift = Point(self.shift_x.value(), self.shift_y.value()),
        )

        return self.image_settings, self.detector_settings, self.beam_settings

    def take_image(self):
        self.image_settings = self.get_settings_from_ui()[0]

        arr =  acquire.new_image(self.microscope, self.image_settings)
        name = f"{self.image_settings.beam_type.name}"

        if self.image_settings.beam_type == BeamType.ELECTRON:
            self.eb_image = arr
        if self.image_settings.beam_type == BeamType.ION:
            self.ib_image = arr

        self.update_viewer(arr.data, name)

    def take_reference_images(self):

        self.image_settings = self.get_settings_from_ui()[0]

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

        beam_settings = self.get_beam_settings(BeamType[self.selected_beam.currentText()])
        detector_settings = self.get_detector_settings(BeamType[self.selected_beam.currentText()])
        self.set_ui_from_settings(image_settings = self.image_settings, detector_settings  = detector_settings, beam_settings = beam_settings, beam_type= BeamType[self.selected_beam.currentText()])
        
        # set the active layer to the electron beam (for movement)
        if self.eb_layer:
            self.viewer.layers.selection.active = self.eb_layer

    def get_detector_settings(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemDetectorSettings:
        contrast = self.microscope.get("detector_contrast", beam_type=beam_type)
        brightness = self.microscope.get("detector_brightness", beam_type=beam_type)
        type = self.microscope.get("detector_type", beam_type=beam_type)
        mode = self.microscope.get("detector_mode", beam_type=beam_type)
        return FibsemDetectorSettings(type, mode, brightness, contrast)
    
    def get_beam_settings(self, beam_type: BeamType= BeamType.ELECTRON) -> BeamSettings:
        beam_settings = BeamSettings(
            beam_type = beam_type,
            working_distance=self.microscope.get("working_distance", beam_type=beam_type),
            beam_current = self.microscope.get("current", beam_type=beam_type),
            voltage = self.microscope.get("voltage", beam_type=beam_type),
            shift=self.microscope.get("shift", beam_type=beam_type),
            stigmation=self.microscope.get("stigmation", beam_type=beam_type),
        )
        return beam_settings


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


    def clear_viewer(self):
        self.viewer.layers.clear()
        self.eb_layer = None
        self.ib_layer = None


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
