import napari
import napari.utils.notifications
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from fibsem.microscope import FibsemMicroscope, TescanMicroscope
from fibsem import constants, acquire

from fibsem.structures import BeamType, ImageSettings, FibsemImage, Point, FibsemDetectorSettings, BeamSettings
from fibsem.ui import utils as ui_utils 

from fibsem.ui.qtdesigner_files import ImageSettingsWidget

from scipy.ndimage import median_filter
from PIL import Image
from scipy.ndimage import median_filter
from PIL import Image
import numpy as np
from pathlib import Path
import logging

def log_status_message(step: str):
    logging.debug(
        f"STATUS | Image Widget | {step}"
    )

class FibsemImageSettingsWidget(ImageSettingsWidget.Ui_Form, QtWidgets.QWidget):
    picture_signal = pyqtSignal()
    viewer_update_signal = pyqtSignal()
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
        self.eb_image, self.ib_image = None, None

        self.eb_last = np.zeros(shape=(1024, 1536), dtype=np.uint8)
        self.ib_last = np.zeros(shape=(1024, 1536), dtype=np.uint8)

        self._features_layer = None

        self._TESCAN = isinstance(self.microscope, TescanMicroscope)

        self.setup_connections()

        if image_settings is not None:
            # self.detector_settings = self.get_detector_settings()
            # self.beam_settings = self.get_beam_settings()
            self.image_settings = image_settings
            self.set_ui_from_settings(image_settings = image_settings, beam_type = BeamType.ELECTRON)
        self.update_detector_ui()

        # register initial images
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ION.name)
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ELECTRON.name)

    def setup_connections(self):

        # set ui elements

        self.selected_beam.addItems([beam.name for beam in BeamType])

        self.pushButton_take_image.clicked.connect(lambda: self.take_image(None))
        self.pushButton_take_all_images.clicked.connect(self.take_reference_images)
        self.checkBox_image_save_image.toggled.connect(self.update_ui_saving_settings)
        self.set_detector_button.clicked.connect(self.apply_detector_settings)
        self.selected_beam.currentIndexChanged.connect(self.update_detector_ui)
        self.button_set_beam_settings.clicked.connect(self.apply_beam_settings)
        self.detector_contrast_slider.valueChanged.connect(self.update_labels)
        self.detector_brightness_slider.valueChanged.connect(self.update_labels)
        self.ion_ruler_checkBox.toggled.connect(self.update_ruler)
        self.scalebar_checkbox.toggled.connect(self.update_ui_tools)
        self.crosshair_checkbox.toggled.connect(self.update_ui_tools)

        if self._TESCAN:

            self.label_11.hide()
            self.stigmation_x.hide()
            self.stigmation_y.hide()
            self.stigmation_x.setEnabled(False)
            self.stigmation_y.setEnabled(False)
            available_presets = self.microscope.get_available_values("presets")
            self.comboBox_presets.addItems(available_presets)   
            self.comboBox_presets.currentTextChanged.connect(self.update_presets)
        else:
            self.comboBox_presets.hide()
            self.label_presets.hide()
  
    def update_presets(self):
        beam_type = BeamType[self.selected_beam.currentText()]
        self.microscope.set("preset", self.comboBox_presets.currentText(), beam_type)
    def check_point_image(self,point):
            
            if point[1] >= 0 and point[1] <= self.eb_layer.data.shape[1]:
                return True
            else:
                return False

    def update_ruler(self):

        if self.ion_ruler_checkBox.isChecked():
            self.ion_ruler_label.setText("Ruler: is on")

            # create initial ruler

            data = [[500,500],[500,1000]]
            p1,p2 = data[0],data[1]


            hfw_scale = self.eb_image.metadata.pixel_size.x if self.check_point_image(p1) else self.ib_image.metadata.pixel_size.x

            midpoint = [np.mean([p1[0],p2[0]]),np.mean([p1[1],p2[1]])]
            dist_um = 500 * hfw_scale*constants.SI_TO_MICRO
            text = {
                "string": [f"{dist_um:.2f} um"],
                "color": "white"
            }

            # creating initial layers 

            self._features_layer = self.viewer.add_points(data, size=20, face_color='lime', edge_color='white', name='ruler')
            self.viewer.add_shapes(data, shape_type='line', edge_color='lime', name='ruler_line',edge_width=5)
            self.viewer.add_points(midpoint,text=text, size=20, face_color='transparent', edge_color='transparent', name='ruler_value')
            self._features_layer.mode = 'select'


            self.viewer.layers.selection.active = self._features_layer
            self._features_layer.mouse_drag_callbacks.append(self.update_ruler_points)


        else:
            self.ion_ruler_label.setText("Ruler: is off")
            self.viewer.layers.remove(self._features_layer)
            self.viewer.layers.remove('ruler_line')
            self.viewer.layers.remove('ruler_value')
            self._features_layer = None
            self.viewer.layers.selection.active = self.eb_layer




    def update_ruler_points(self,layer, event):
        
        dragged = False
        yield

        while event.type == 'mouse_move':


            if self._features_layer.selected_data is not None:
                data = self._features_layer.data


                p1 = data[0]
                p2 = data[1]

                dist_pix = np.linalg.norm(p1-p2)
                
                midpoint = [(np.mean([p1[0],p2[0]])),(np.mean([p1[1],p2[1]]))]
                
                self.viewer.layers['ruler_line'].data = [p1,p2]
                self.viewer.layers['ruler_value'].data = midpoint
                
                hfw_scale = self.eb_image.metadata.pixel_size.x if self.check_point_image(p1) else self.ib_image.metadata.pixel_size.x

                dist_um = dist_pix * hfw_scale*constants.SI_TO_MICRO

                text = {
                "string": [f"{dist_um:.2f} um"],
                "color": "white"
                }

                self.viewer.layers['ruler_value'].text = text
                dist_dx = abs(p2[1]-p1[1]) * self.image_settings.hfw/self.image_settings.resolution[0]*constants.SI_TO_MICRO
                dist_dy = abs(p2[0]-p1[0]) * self.image_settings.hfw/self.image_settings.resolution[0]*constants.SI_TO_MICRO


                self.ion_ruler_label.setText(f"Ruler: {dist_um:.2f} um  dx: {dist_dx:.2f} um  dy: {dist_dy:.2f} um")
                self.viewer.layers.selection.active = self._features_layer
                self.viewer.layers['ruler_line'].refresh()

                
                dragged = True
                yield
            



    def update_labels(self):
        self.detector_contrast_label.setText(f"{self.detector_contrast_slider.value()}%")
        self.detector_brightness_label.setText(f"{self.detector_brightness_slider.value()}%")

    def apply_detector_settings(self):
        beam =  BeamType[self.selected_beam.currentText()]
        self.get_settings_from_ui()
        self.microscope.set("detector_type", self.detector_settings.type, beam_type=beam)
        self.microscope.set("detector_mode", self.detector_settings.mode, beam_type=beam)
        self.microscope.set("detector_brightness", self.detector_settings.brightness, beam_type=beam)
        self.microscope.set("detector_contrast", self.detector_settings.contrast, beam_type=beam)
        log_status_message("SET_DETECTOR_PARAMETERS")
        log_status_message(f"Detector Type: {self.detector_settings.type}, Mode: {self.detector_settings.mode}, Brightness: {self.detector_settings.brightness}, Contrast: {self.detector_settings.contrast}")
        

    def apply_beam_settings(self):
        beam = BeamType[self.selected_beam.currentText()]
        self.get_settings_from_ui()
        self.microscope.set("working_distance", self.beam_settings.working_distance, beam_type=beam)
        self.microscope.set("current", self.beam_settings.beam_current, beam_type=beam)
        self.microscope.set("voltage", self.beam_settings.voltage, beam_type=beam)
        self.microscope.set("stigmation", self.beam_settings.stigmation, beam_type=beam)
        self.microscope.set("shift", self.beam_settings.shift, beam_type=beam)
        log_status_message("SET_BEAM_PARAMETERS")
        log_status_message(f"Working Distance: {self.beam_settings.working_distance}, Current: {self.beam_settings.beam_current}, Voltage: {self.beam_settings.voltage}, Stigmation: {self.beam_settings.stigmation}, Shift: {self.beam_settings.shift}")
        self.set_ui_from_settings(self.image_settings,beam)

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
            brightness=self.detector_brightness_slider.value()*constants.FROM_PERCENTAGES,
            contrast=self.detector_contrast_slider.value()*constants.FROM_PERCENTAGES,
        )

        self.beam_settings = BeamSettings(
            beam_type=BeamType[self.selected_beam.currentText()],
            working_distance=self.working_distance.value()*constants.MILLI_TO_SI,
            beam_current=self.beam_current.value()*constants.PICO_TO_SI,
            voltage=self.beam_voltage.value()*constants.KILO_TO_SI,
            hfw = self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            resolution=[self.spinBox_resolution_x.value(), self.spinBox_resolution_y.value()],
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            stigmation = Point(self.stigmation_x.value(), self.stigmation_y.value()),
            shift = Point(self.shift_x.value() * constants.MICRO_TO_SI, self.shift_y.value()*constants.MICRO_TO_SI),
        )
        return self.image_settings, self.detector_settings, self.beam_settings

    def set_ui_from_settings(self, image_settings: ImageSettings, beam_type: BeamType):

        # disconnect beam type combobox
        self.selected_beam.currentIndexChanged.disconnect()
        self.selected_beam.setCurrentText(beam_type.name)
        self.selected_beam.currentIndexChanged.connect(self.update_detector_ui)
        
        beam_settings = self.get_beam_settings(beam_type)
        detector_settings = self.get_detector_settings(beam_type)

        self.spinBox_resolution_x.setValue(image_settings.resolution[0])
        self.spinBox_resolution_y.setValue(image_settings.resolution[1])

        self.doubleSpinBox_image_dwell_time.setValue(image_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_image_hfw.setValue(image_settings.hfw * constants.SI_TO_MICRO)

        self.checkBox_image_use_autocontrast.setChecked(image_settings.autocontrast)
        self.checkBox_image_use_autogamma.setChecked(image_settings.gamma_enabled)
        self.checkBox_image_save_image.setChecked(image_settings.save)
        self.lineEdit_image_path.setText(str(image_settings.save_path))
        self.lineEdit_image_label.setText(image_settings.label)

        self.detector_type_combobox.setCurrentText(detector_settings.type)
        self.detector_mode_combobox.setCurrentText(detector_settings.mode)
        self.detector_contrast_slider.setValue(int(detector_settings.contrast*100))
        self.detector_brightness_slider.setValue(int(detector_settings.brightness*100))
        self.beam_current.setValue(beam_settings.beam_current*constants.SI_TO_PICO)
        self.beam_voltage.setValue(beam_settings.voltage*constants.SI_TO_KILO)
        if beam_settings.working_distance is not None:
            self.working_distance.setValue(beam_settings.working_distance*constants.METRE_TO_MILLIMETRE)
        if beam_settings.shift is not None:
            self.shift_x.setValue(beam_settings.shift.x * constants.SI_TO_MICRO)
            self.shift_y.setValue(beam_settings.shift.y * constants.SI_TO_MICRO)
        if beam_settings.stigmation is not None:
            self.stigmation_x.setValue(beam_settings.stigmation.x)
            self.stigmation_y.setValue(beam_settings.stigmation.y)
        
        self.update_ui_saving_settings()

    def update_ui_saving_settings(self):

        self.label_image_save_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.label_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        
  
    def update_detector_ui(self):
        beam_type = BeamType[self.selected_beam.currentText()]
        # if beam_type is BeamType.ELECTRON:
        #     self.comboBox_presets.hide()
        #     self.label_presets.hide()
        # else:
        #     self.comboBox_presets.show()
        #     self.label_presets.show()

        _is_ion = bool(beam_type is BeamType.ION)
        _is_tescan = isinstance(self.microscope, TescanMicroscope)
        
        self.comboBox_presets.setVisible(_is_ion and _is_tescan)
        self.label_presets.setVisible(_is_ion and _is_tescan)


        self.detector_type = self.microscope.get_available_values("detector_type", beam_type=beam_type)
        self.detector_type_combobox.clear()
        self.detector_type_combobox.addItems(self.detector_type)
        self.detector_type_combobox.setCurrentText(self.microscope.get("detector_type", beam_type=beam_type))
        
        self.detector_mode = self.microscope.get_available_values("detector_mode", beam_type=beam_type)
        self.detector_mode_combobox.clear()
        self.detector_mode_combobox.addItems(self.detector_mode if self.detector_mode is not None else ["N/A"])# if self.detector_mode is not None  else self.mode.addItem("N/A")
        self.detector_mode_combobox.setCurrentText(self.microscope.get("detector_mode", beam_type=beam_type))

        self.set_ui_from_settings(self.image_settings, beam_type)

    def take_image(self, beam_type: BeamType = None):
        self.image_settings = self.get_settings_from_ui()[0]
        
        if beam_type is not None:
            self.image_settings.beam_type = beam_type

        arr =  acquire.new_image(self.microscope, self.image_settings)
        name = f"{self.image_settings.beam_type.name}"

        if self.image_settings.beam_type == BeamType.ELECTRON:
            self.eb_image = arr
        if self.image_settings.beam_type == BeamType.ION:
            self.ib_image = arr
        
        self.picture_signal.emit()

        self.update_viewer(arr.data, name)

        log_status_message("IMAGE_TAKEN_{beam_type}".format(beam_type=self.image_settings.beam_type.name))
        log_status_message("Settings used: {}".format(self.image_settings))

    def take_reference_images(self):

        self.image_settings = self.get_settings_from_ui()[0]

        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.update_viewer(self.ib_image.data, BeamType.ION.name)
        self.update_viewer(self.eb_image.data, BeamType.ELECTRON.name)
        self.picture_signal.emit()
        log_status_message("REFERENCE_IMAGES_TAKEN")
        log_status_message("Settings used: {}".format(self.image_settings))

    def update_ui_tools(self):

        self.update_viewer(self.eb_last, BeamType.ELECTRON.name)
        self.update_viewer(self.ib_last, BeamType.ION.name)



    def update_viewer(self, arr: np.ndarray, name: str):


        if name == BeamType.ELECTRON.name:
            self.eb_last = arr
        if name == BeamType.ION.name:
            self.ib_last = arr

        # median filter for display
        arr = median_filter(arr, size=3)
       
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
            translation = (
                self.viewer.layers["ELECTRON"].data.shape[1]
                if self.eb_layer
                else arr.shape[1]
            )
            self.ib_layer.translate = [0.0, translation]       

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


        # draw scalebar and crosshair
        if self.eb_image is not None and self.ib_image is not None:
            ui_utils._draw_scalebar(viewer=self.viewer,eb_image= self.eb_image,ib_image= self.ib_image,is_checked=self.scalebar_checkbox.isChecked())
            ui_utils._draw_crosshair(viewer=self.viewer,eb_image= self.eb_image,ib_image= self.ib_image,is_checked=self.crosshair_checkbox.isChecked()) 
            
        self.set_ui_from_settings(image_settings = self.image_settings, beam_type= BeamType[self.selected_beam.currentText()])      
        
        # set the active layer to the electron beam (for movement)
        if self.eb_layer:
            self.viewer.layers.selection.active = self.eb_layer

        self.viewer_update_signal.emit()

    def get_data_from_coord(self, coords: tuple) -> tuple:
        # check inside image dimensions, (y, x)
        eb_shape = self.eb_image.data.shape[0], self.eb_image.data.shape[1]
        ib_shape = self.ib_image.data.shape[0], self.ib_image.data.shape[1] + self.eb_image.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (
            coords[1] > 0 and coords[1] < eb_shape[1]
        ):
            image = self.eb_image
            beam_type = BeamType.ELECTRON

        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (
            coords[1] > eb_shape[0] and coords[1] < ib_shape[1]
        ):
            image = self.ib_image
            coords = (coords[0], coords[1] - ib_shape[1] // 2)
            beam_type = BeamType.ION
        else:
            beam_type, image = None, None
        log_status_message(f"COORDS: {coords}, BEAM_TYPE: {beam_type}")
        return coords, beam_type, image
    
    def closeEvent(self, event):
        self.viewer.layers.clear()
        event.accept()

    def clear_viewer(self):
        self.viewer.layers.clear()
        self.eb_layer = None
        self.ib_layer = None

    def _set_active_layer(self):
        if self.eb_layer:
            self.viewer.layers.selection.active = self.eb_layer

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
