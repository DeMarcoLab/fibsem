import napari
import logging
from pathlib import Path

import napari.utils.notifications

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

from fibsem import acquire, constants
from fibsem.microscope import FibsemMicroscope, TescanMicroscope
from fibsem.structures import (BeamSettings, BeamType, FibsemDetectorSettings,
                               FibsemImage, ImageSettings, Point)
from fibsem.ui import utils as ui_utils

from fibsem.microscope import FibsemMicroscope, TescanMicroscope
from fibsem import constants, acquire

from fibsem.structures import BeamType, ImageSettings, FibsemImage, Point, FibsemDetectorSettings, BeamSettings
from fibsem.ui import utils as ui_utils 
from fibsem.ui import _stylesheets

from fibsem.ui.qtdesigner_files import ImageSettingsWidget
from fibsem.ui import _stylesheets
# from napari.qt.threading import thread_worker
   
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
import threading
from queue import Queue

import numpy as np
from fibsem import config as cfg
        

class FibsemImageSettingsWidget(ImageSettingsWidget.Ui_Form, QtWidgets.QWidget):
    picture_signal = pyqtSignal()
    viewer_update_signal = pyqtSignal()
    image_notification_signal = pyqtSignal(str)
    live_imaging_signal = pyqtSignal(dict)

    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        image_settings: ImageSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemImageSettingsWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent
        self.microscope = microscope
        self.viewer = viewer
        self.eb_layer, self.ib_layer = None, None
        self.eb_image, self.ib_image = None, None

        self.eb_last = np.zeros(shape=(1024, 1536), dtype=np.uint8)
        self.ib_last = np.zeros(shape=(1024, 1536), dtype=np.uint8)

        self._features_layer = None
        self.stop_event = threading.Event()
        self.stop_event.set()
        self.image_queue = Queue()

        self._TESCAN = isinstance(self.microscope, TescanMicroscope)
        self.TAKING_IMAGES = False
        self._LIVE_IMAGING = False

        self.setup_connections()

        if image_settings is not None:
            # self.detector_settings = self.get_detector_settings()
            # self.beam_settings = self.get_beam_settings()
            self.image_settings = image_settings
            self.set_ui_from_settings(image_settings = image_settings, beam_type = BeamType.ELECTRON)
        self.update_detector_ui() # TODO: can this be removed?

        # register initial images
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ION.name)
        self.update_viewer(np.zeros(shape=(1024, 1536), dtype=np.uint8), BeamType.ELECTRON.name)
        self.live_imaging_signal.connect(self.live_update)
    
    def setup_connections(self):

        # set ui elements
        self.selected_beam.addItems([beam.name for beam in BeamType])

        # buttons
        self.pushButton_take_image.clicked.connect(lambda: self.take_image(None))
        self.pushButton_take_all_images.clicked.connect(self.take_reference_images)
        self.pushButton_live_imaging.clicked.connect(self.live_imaging)
                
        # feature flags
        self.pushButton_live_imaging.setVisible(cfg._LIVE_IMAGING_ENABLED)
        
        # image / beam settings
        self.selected_beam.currentIndexChanged.connect(self.update_detector_ui)
        self.checkBox_image_save_image.toggled.connect(self.update_ui_saving_settings)
        self.button_set_beam_settings.clicked.connect(self.apply_beam_settings)
        
        # detector
        self.set_detector_button.clicked.connect(self.apply_detector_settings)
        self.detector_contrast_slider.valueChanged.connect(self.update_labels)
        self.detector_brightness_slider.valueChanged.connect(self.update_labels)
        
        # util
        self.ion_ruler_checkBox.toggled.connect(self.update_ruler)
        self.scalebar_checkbox.toggled.connect(self.update_ui_tools)
        self.crosshair_checkbox.toggled.connect(self.update_ui_tools)
        
        # signals
        self.image_notification_signal.connect(self.update_imaging_ui)
        
        # advanced settings
        self.checkBox_advanced_settings.stateChanged.connect(self.toggle_mode)
        self.toggle_mode()


        # set ui stylesheets
        self.pushButton_take_image.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_take_all_images.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_live_imaging.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.set_detector_button.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
        self.button_set_beam_settings.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)

        if self._TESCAN:

            self.label_stigmation.hide()
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
  
    def toggle_mode(self):
        """Toggle the visibility of the advanced settings"""
        advanced_mode = self.checkBox_advanced_settings.isChecked()

        self.label_detector_type.setVisible(advanced_mode)
        self.detector_type_combobox.setVisible(advanced_mode)
        self.label_detector_mode.setVisible(advanced_mode)
        self.detector_mode_combobox.setVisible(advanced_mode)
        self.label_stigmation.setVisible(advanced_mode)
        self.stigmation_x.setVisible(advanced_mode)
        self.stigmation_y.setVisible(advanced_mode)
        self.shift_x.setVisible(advanced_mode)
        self.shift_y.setVisible(advanced_mode)
        self.label_shift.setVisible(advanced_mode)
        self.beam_voltage.setVisible(advanced_mode)
        self.label_beam_voltage.setVisible(advanced_mode)
        self.label_beam_scan_rotation.setVisible(advanced_mode)
        self.spinBox_beam_scan_rotation.setVisible(advanced_mode)
        self.checkBox_image_use_autocontrast.setVisible(advanced_mode)
        self.checkBox_image_use_autogamma.setVisible(advanced_mode)
            


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
        
        # read settings from ui
        beam =  BeamType[self.selected_beam.currentText()]
        self.get_settings_from_ui()

        # set detector settings
        self.microscope.set_detector_settings(self.detector_settings, beam_type=beam)
        
        # logging
        logging.debug({"msg": "apply_detector_settings", "detector_settings": self.detector_settings.to_dict()})

        # notifications
        napari.utils.notifications.show_info("Detector Settings Updated")

    def apply_beam_settings(self):
        beam = BeamType[self.selected_beam.currentText()]
        self.get_settings_from_ui()

        # set beam settings
        self.microscope.set_beam_settings(self.beam_settings)

        # logging 
        logging.debug({"msg": "apply_beam_settings", "beam_settings": self.beam_settings.to_dict()})

        # QUERY: why is this here?
        self.set_ui_from_settings(self.image_settings,beam)
        
        # notifications
        napari.utils.notifications.show_info("Beam Settings Updated")

    def get_settings_from_ui(self):

        self.image_settings = ImageSettings(
            resolution=[self.spinBox_resolution_x.value(), self.spinBox_resolution_y.value()],
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            beam_type=BeamType[self.selected_beam.currentText()],
            autocontrast=self.checkBox_image_use_autocontrast.isChecked(),
            autogamma=self.checkBox_image_use_autogamma.isChecked(),
            save=self.checkBox_image_save_image.isChecked(),
            path=Path(self.lineEdit_image_path.text()),
            filename=self.lineEdit_image_label.text()
            
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
            scan_rotation = np.deg2rad(self.spinBox_beam_scan_rotation.value())
        )
        return self.image_settings, self.detector_settings, self.beam_settings

    def set_ui_from_settings(self, image_settings: ImageSettings, 
                             beam_type: BeamType, 
                             beam_settings: BeamSettings=None, 
                             detector_settings: FibsemDetectorSettings=None ):
        """Update the ui from the image, beam and detector settings"""

        # disconnect beam type combobox
        self.selected_beam.currentIndexChanged.disconnect()
        self.selected_beam.setCurrentText(beam_type.name)
        self.selected_beam.currentIndexChanged.connect(self.update_detector_ui)
        
        if beam_settings is None:
            beam_settings = self.microscope.get_beam_settings(beam_type)
        if detector_settings is None:
            detector_settings = self.microscope.get_detector_settings(beam_type)

        # imaging settings
        self.spinBox_resolution_x.setValue(image_settings.resolution[0])
        self.spinBox_resolution_y.setValue(image_settings.resolution[1])

        self.doubleSpinBox_image_dwell_time.setValue(image_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_image_hfw.setValue(image_settings.hfw * constants.SI_TO_MICRO)

        self.checkBox_image_use_autocontrast.setChecked(image_settings.autocontrast)
        self.checkBox_image_use_autogamma.setChecked(image_settings.autogamma)
        self.checkBox_image_save_image.setChecked(image_settings.save)
        self.lineEdit_image_path.setText(str(image_settings.path))
        self.lineEdit_image_label.setText(image_settings.filename)

        # detector settings
        self.detector_type_combobox.setCurrentText(detector_settings.type)
        self.detector_mode_combobox.setCurrentText(detector_settings.mode)
        self.detector_contrast_slider.setValue(int(detector_settings.contrast*100))
        self.detector_brightness_slider.setValue(int(detector_settings.brightness*100))
        
        # beam settings
        self.beam_current.setValue(beam_settings.beam_current*constants.SI_TO_PICO)
        self.beam_voltage.setValue(beam_settings.voltage*constants.SI_TO_KILO)
        self.spinBox_beam_scan_rotation.setValue(np.rad2deg(beam_settings.scan_rotation))
        
        self.working_distance.setValue(beam_settings.working_distance*constants.METRE_TO_MILLIMETRE)
        self.shift_x.setValue(beam_settings.shift.x * constants.SI_TO_MICRO)
        self.shift_y.setValue(beam_settings.shift.y * constants.SI_TO_MICRO)
        self.stigmation_x.setValue(beam_settings.stigmation.x)
        self.stigmation_y.setValue(beam_settings.stigmation.y)
        
        self.update_ui_saving_settings()

    def update_ui_saving_settings(self):
        """Toggle the visibility of the imaging saving settings"""
        self.label_image_save_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_path.setVisible(self.checkBox_image_save_image.isChecked())
        self.label_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        self.lineEdit_image_label.setVisible(self.checkBox_image_save_image.isChecked())
        
  
    def update_detector_ui(self):
        """Update the detector ui based on currently selected beam"""
        beam_type = BeamType[self.selected_beam.currentText()]

        _is_ion = bool(beam_type is BeamType.ION)
        _is_tescan = isinstance(self.microscope, TescanMicroscope)
        
        self.comboBox_presets.setVisible(_is_ion and _is_tescan)
        self.label_presets.setVisible(_is_ion and _is_tescan)


        available_detector_types = self.microscope.get_available_values("detector_type", beam_type=beam_type)
        self.detector_type_combobox.clear()
        self.detector_type_combobox.addItems(available_detector_types)
        self.detector_type_combobox.setCurrentText(self.microscope.get("detector_type", beam_type=beam_type))
        
        available_detector_modes = self.microscope.get_available_values("detector_mode", beam_type=beam_type)
        if available_detector_modes is None: 
            available_detector_modes = ["N/A"]
        self.detector_mode_combobox.clear()
        self.detector_mode_combobox.addItems(available_detector_modes)
        self.detector_mode_combobox.setCurrentText(self.microscope.get("detector_mode", beam_type=beam_type))

        self.set_ui_from_settings(self.image_settings, beam_type)

    def live_imaging(self):
        if self.stop_event.is_set():
            self._toggle_interactions(False)
            self.pushButton_live_imaging.setEnabled(True)
            self.pushButton_live_imaging.setText("Stop live imaging")
            self.parent.movement_widget.checkBox_movement_acquire_electron.setChecked(False)
            self.parent.movement_widget.checkBox_movement_acquire_ion.setChecked(False)
            self.pushButton_live_imaging.setStyleSheet("background-color: orange")

            self.stop_event.clear()
            self.image_queue.queue.clear()
            image_settings = self.get_settings_from_ui()[0]
            image_settings.autocontrast = False
            from copy import deepcopy
            _thread = threading.Thread(
                target=self.microscope.live_imaging,
                args=(deepcopy(image_settings), self.image_queue, self.stop_event),
            )
            _thread.start()
            sleep_time = image_settings.dwell_time*image_settings.resolution[0]*image_settings.resolution[1]
            worker = self.microscope.consume_image_queue(parent_ui = self, sleep = sleep_time)
            worker.returned.connect(self.update_live_finished)
            import time
            time.sleep(1)
            worker.start()  
            self._LIVE_IMAGING = True

        else:
            self._LIVE_IMAGING = False
            self.stop_event.set()
            self.pushButton_live_imaging.setStyleSheet("""
                    QPushButton {
                        background-color: green;
                        color: white;
                    }
                    QPushButton:hover {
                        background-color: gray;
                    }
                    QPushButton:pressed { """
                )
            self.parent.movement_widget.checkBox_movement_acquire_electron.setChecked(False)
            self.parent.movement_widget.checkBox_movement_acquire_ion.setChecked(False)


    def update_live_finished(self):
        self.pushButton_live_imaging.setText("Live imaging")
        self.pushButton_take_all_images.setEnabled(True)
        self.pushButton_take_image.setEnabled(True)
        self._toggle_interactions(True)

    def live_update(self, dict):
        arr = dict["image"].data
        name = BeamType[self.selected_beam.currentText()].name

        try:
            self.viewer.layers[name].data = arr
        except:    
            layer = self.viewer.add_image(arr, name = name)

        if name == BeamType.ELECTRON.name:
            self.eb_image = dict["image"]
        if name == BeamType.ION.name:
            self.ib_image = dict["image"]    
        
    def take_image(self, beam_type: BeamType = None):
        self.TAKING_IMAGES = True
        worker = self.take_image_worker(beam_type)
        worker.finished.connect(self.imaging_finished)
        worker.start()

    def update_imaging_ui(self, msg: str):
        logging.info(msg)
        napari.utils.notifications.notification_manager.records.clear()
        napari.utils.notifications.show_info(msg)

    def imaging_finished(self):
        if self.ib_image is not None:
            self.update_viewer(self.ib_image.data, BeamType.ION.name)
        if self.eb_image is not None:
            self.update_viewer(self.eb_image.data, BeamType.ELECTRON.name)
        self._toggle_interactions(True)
        self.TAKING_IMAGES = False

    def _toggle_interactions(self, enable: bool, caller: str = None, imaging: bool = False):
        self.pushButton_take_image.setEnabled(enable)
        self.pushButton_live_imaging.setEnabled(enable)
        self.pushButton_take_all_images.setEnabled(enable)
        self.set_detector_button.setEnabled(enable)
        self.button_set_beam_settings.setEnabled(enable)
        self.parent.movement_widget._toggle_interactions(enable, caller="ui")
        if caller != "milling":
            self.parent.milling_widget._toggle_interactions(enable, caller="ui")
        if enable:
            self.pushButton_take_all_images.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.pushButton_live_imaging.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.set_detector_button.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
            self.button_set_beam_settings.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setText("Acquire Image")
            self.pushButton_take_all_images.setText("Acquire All Images")
        elif imaging:
            self.pushButton_take_all_images.setStyleSheet(_stylesheets._ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setStyleSheet(_stylesheets._ORANGE_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setText("Acquiring Images...")
            self.pushButton_take_all_images.setText("Acquiring Images...")
            self.set_detector_button.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.button_set_beam_settings.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_live_imaging.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
        else:
            self.pushButton_take_all_images.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.set_detector_button.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.button_set_beam_settings.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setText("Acquire Image")
            self.pushButton_take_all_images.setText("Acquire All Images")

    from napari.qt.threading import thread_worker
                
    @thread_worker
    def take_image_worker(self, beam_type: BeamType = None):
        self._toggle_interactions(enable=False, imaging=True)
        self.image_settings = self.get_settings_from_ui()[0]
        self.image_notification_signal.emit("Acquiring Image...")
        if beam_type is not None:
            self.image_settings.beam_type = beam_type

        arr =  acquire.new_image(self.microscope, self.image_settings)
        name = f"{self.image_settings.beam_type.name}"

        if self.image_settings.beam_type == BeamType.ELECTRON:
            self.eb_image = arr
        if self.image_settings.beam_type == BeamType.ION:
            self.ib_image = arr
        
        self.picture_signal.emit()

        logging.debug({"msg": "take_image_worker", "image_settings": self.image_settings.to_dict()})


    def take_reference_images(self):
        self.TAKING_IMAGES = True
        worker = self.take_reference_images_worker()
        worker.finished.connect(self.imaging_finished)
        worker.start()

    @thread_worker
    def take_reference_images_worker(self):
        self._toggle_interactions(enable=False,imaging=True)
        self.image_settings = self.get_settings_from_ui()[0]
        self.image_notification_signal.emit("Acquiring Images...")
        self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.picture_signal.emit()

        logging.debug({"msg": "take_reference_images_worker", "image_settings": self.image_settings.to_dict()})

    def update_ui_tools(self):
        """Redraw the ui tools (scalebar, crosshair)"""
        self.update_viewer(self.eb_last, BeamType.ELECTRON.name)
        self.update_viewer(self.ib_last, BeamType.ION.name)

    def update_viewer(self, arr: np.ndarray, name: str, _set_ui: bool = False):
        """Update the viewer with the given image array"""

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
            
        
        # set ui from image metadata
        if _set_ui:
            if name == BeamType.ELECTRON.name:
                self.image_settings = self.eb_image.metadata.image_settings
                beam_settings = self.eb_image.metadata.microscope_state.electron_beam
                detector_settings = self.eb_image.metadata.microscope_state.electron_detector
                beam_type = BeamType.ELECTRON
            if name == BeamType.ION.name:
                self.image_settings = self.ib_image.metadata.image_settings
                beam_settings = self.ib_image.metadata.microscope_state.ion_beam
                detector_settings = self.ib_image.metadata.microscope_state.ion_detector
                beam_type = BeamType.ION            
        else:
            beam_type = BeamType[self.selected_beam.currentText()]
            beam_settings, detector_settings = None, None    
    
        self.set_ui_from_settings(image_settings = self.image_settings, beam_type= beam_type, 
            beam_settings=beam_settings, detector_settings=detector_settings)  
            
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

        # logging
        logging.debug( {"msg": "get_data_from_coord", "coords": coords, "beam_type": beam_type})

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
    save=True, filename="my_label", path="path/to/save", 
    autogamma=True)

    viewer = napari.Viewer(ndisplay=2)
    image_settings_ui = FibsemImageSettingsWidget(image_settings=image_settings)
    viewer.window.add_dock_widget(
        image_settings_ui, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
