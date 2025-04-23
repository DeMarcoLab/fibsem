import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Shapes as NapariShapesLayer
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from scipy.ndimage import median_filter

from fibsem import acquire, constants
from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope, TescanMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    Point,
)
from fibsem.ui import stylesheets as stylesheets
from fibsem.ui.napari.patterns import (
    convert_reduced_area_to_napari_shape,
    convert_shape_to_image_area,
)
from fibsem.ui.napari.properties import (
    ALIGNMENT_LAYER_PROPERTIES,
    IMAGE_TEXT_LAYER_PROPERTIES,
    RULER_LAYER_NAME,
    RULER_LINE_LAYER_NAME,
    RULER_VALUE_LAYER_NAME,
)
from fibsem.ui.napari.utilities import draw_crosshair_in_napari, draw_scalebar_in_napari
from fibsem.ui.qtdesigner_files import ImageSettingsWidget as ImageSettingsWidgetUI

# feature flags
ENABLE_ADVANCED_IMAGING_SETTINGS = True

class FibsemImageSettingsWidget(ImageSettingsWidgetUI.Ui_Form, QtWidgets.QWidget):
    viewer_update_signal = pyqtSignal()             # when the viewer is updated
    acquisition_progress_signal = pyqtSignal(dict)  # TODO: add progress indicator

    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        image_settings: ImageSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent
        self.microscope = microscope
        self.viewer = viewer
        self.eb_layer: NapariImageLayer = None
        self.ib_layer: NapariImageLayer = None

        # TODO: migrate to this structure
        self.imaging_layers: Dict[str, NapariImageLayer] = {}
        self.imaging_layers[BeamType.ELECTRON] = None
        self.imaging_layers[BeamType.ION] = None

        # generate initial blank images 
        self.eb_image = FibsemImage.generate_blank_image(resolution=image_settings.resolution, hfw=image_settings.hfw)
        self.ib_image = FibsemImage.generate_blank_image(resolution=image_settings.resolution, hfw=image_settings.hfw)

        # overlay layers
        self.ruler_layer: NapariShapesLayer = None
        self.alignment_layer: NapariShapesLayer = None

        self.is_acquiring: bool = False

        self.setup_connections()

        if image_settings is not None:
            self.image_settings = image_settings
            self.set_ui_from_settings(image_settings = image_settings, beam_type = BeamType.ELECTRON)
        self.update_detector_ui() # TODO: can this be removed?

        # register initial images
        self.update_viewer(self.ib_image.data, BeamType.ION)
        self.update_viewer(self.eb_image.data, BeamType.ELECTRON)
    
    def setup_connections(self):

        # set ui elements
        AVAILABLE_BEAMS = [beam.name for beam in BeamType] # TODO: convert to microscope method
        self.selected_beam.addItems(AVAILABLE_BEAMS)
        self.comboBox_image_resolution.addItems(cfg.STANDARD_RESOLUTIONS)
        self.comboBox_image_resolution.setCurrentText(cfg.DEFAULT_STANDARD_RESOLUTION)
        self.doubleSpinBox_beam_current.setRange(0.1, 10000.0) # TODO: convert to combobox

        # buttons
        self.pushButton_take_image.clicked.connect(self.acquire_image)
        self.pushButton_acquire_sem_image.clicked.connect(self.acquire_sem_image)
        self.pushButton_acquire_fib_image.clicked.connect(self.acquire_fib_image)
        self.pushButton_take_all_images.clicked.connect(self.acquire_reference_images)
                        
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
        self.acquisition_progress_signal.connect(self.handle_acquisition_progress_update)
        self.viewer_update_signal.connect(self.update_ui_tools)
                
        # advanced settings
        self.checkBox_advanced_settings.stateChanged.connect(self.toggle_mode)
        self.toggle_mode()

        # set ui stylesheets
        self.pushButton_take_image.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_acquire_sem_image.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_acquire_fib_image.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.pushButton_take_all_images.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
        self.set_detector_button.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
        self.button_set_beam_settings.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)

        self.IS_TESCAN = isinstance(self.microscope, TescanMicroscope)
        if self.IS_TESCAN:
            self.label_stigmation.hide()
            self.stigmation_x.hide()
            self.stigmation_y.hide()
            self.stigmation_x.setEnabled(False)
            self.stigmation_y.setEnabled(False)
            available_presets = self.microscope.get_available_values("presets")
            self.comboBox_presets.addItems(available_presets)   
            self.comboBox_presets.currentTextChanged.connect(self.update_presets)
            self.checkBox_image_line_integration.setVisible(False)
            self.checkBox_image_scan_interlacing.setVisible(False)
            self.checkBox_image_frame_integration.setVisible(False)
            self.checkBox_image_drift_correction.setVisible(False)
            self.spinBox_image_line_integration.setVisible(False)
            self.spinBox_image_scan_interlacing.setVisible(False)
            self.spinBox_image_frame_integration.setVisible(False)
        else:
            self.comboBox_presets.hide()
            self.label_presets.hide()

        # advanced imaging settings
        self.spinBox_image_line_integration.setRange(2, 255)
        self.spinBox_image_frame_integration.setRange(2, 512)
        self.spinBox_image_scan_interlacing.setRange(2, 8)
        self.spinBox_image_line_integration.setEnabled(False)
        self.spinBox_image_frame_integration.setEnabled(False)
        self.spinBox_image_scan_interlacing.setEnabled(False)
        self.checkBox_image_line_integration.toggled.connect(self.spinBox_image_line_integration.setEnabled)
        self.checkBox_image_scan_interlacing.toggled.connect(self.spinBox_image_scan_interlacing.setEnabled)
        self.checkBox_image_frame_integration.toggled.connect(self.spinBox_image_frame_integration.setEnabled)
        self.checkBox_image_frame_integration.toggled.connect(self.checkBox_image_drift_correction.setEnabled)

        # feature flags
        self.pushButton_show_alignment_area.setVisible(False)
        self.pushButton_show_alignment_area.clicked.connect(lambda: self.toggle_alignment_area(None))

        self.pushButton_take_image.setVisible(False)
        self.acquisition_buttons: list =[self.pushButton_take_all_images, 
                                         self.pushButton_take_image, 
                                         self.pushButton_acquire_sem_image, 
                                         self.pushButton_acquire_fib_image]

    def toggle_mode(self):
        """Toggle the visibility of the advanced settings"""
        advanced_mode = self.checkBox_advanced_settings.isChecked()

        self.label_detector_type.setVisible(advanced_mode)
        self.detector_type_combobox.setVisible(advanced_mode)
        self.label_detector_mode.setVisible(advanced_mode)
        self.detector_mode_combobox.setVisible(advanced_mode)
        self.label_stigmation.setVisible(advanced_mode)
        self.doubleSpinBox_stigmation_x.setVisible(advanced_mode)
        self.doubleSpinBox_stigmation_y.setVisible(advanced_mode)
        self.doubleSpinBox_shift_x.setVisible(advanced_mode)
        self.doubleSpinBox_shift_y.setVisible(advanced_mode)
        self.label_shift.setVisible(advanced_mode)
        self.doubleSpinBox_beam_voltage.setVisible(advanced_mode)
        self.label_beam_voltage.setVisible(advanced_mode)
        self.label_beam_scan_rotation.setVisible(advanced_mode)
        self.spinBox_beam_scan_rotation.setVisible(advanced_mode)
        self.checkBox_image_use_autocontrast.setVisible(advanced_mode)
        self.checkBox_image_use_autogamma.setVisible(advanced_mode)

        # advanced imaging
        self.checkBox_image_line_integration.setVisible(advanced_mode and ENABLE_ADVANCED_IMAGING_SETTINGS)
        self.checkBox_image_scan_interlacing.setVisible(advanced_mode and ENABLE_ADVANCED_IMAGING_SETTINGS)
        self.checkBox_image_frame_integration.setVisible(advanced_mode and ENABLE_ADVANCED_IMAGING_SETTINGS)
        self.checkBox_image_drift_correction.setVisible(advanced_mode and ENABLE_ADVANCED_IMAGING_SETTINGS)
        self.spinBox_image_line_integration.setVisible(advanced_mode and ENABLE_ADVANCED_IMAGING_SETTINGS)
        self.spinBox_image_scan_interlacing.setVisible(advanced_mode and ENABLE_ADVANCED_IMAGING_SETTINGS)
        self.spinBox_image_frame_integration.setVisible(advanced_mode and ENABLE_ADVANCED_IMAGING_SETTINGS)
            
    def update_presets(self):
        beam_type = BeamType[self.selected_beam.currentText()]
        self.microscope.set("preset", self.comboBox_presets.currentText(), beam_type)
    
    def update_ruler(self):

        def check_point_image_in_eb(point: Tuple[int,int]) -> bool:
            
            if point[1] >= 0 and point[1] <= self.eb_layer.data.shape[1]:
                return True
            else:
                return False

        # TODO: re-do this and consolidate into napari.utilities
        # TODO: add properties config
        if self.ion_ruler_checkBox.isChecked():
            self.ion_ruler_label.setText("Ruler: is on")
            self.ion_ruler_label.setVisible(True)

            # create initial ruler

            data = [[500,500],[500,1000]]
            p1,p2 = data[0],data[1]


            hfw_scale = self.eb_image.metadata.pixel_size.x if check_point_image_in_eb(p1) else self.ib_image.metadata.pixel_size.x

            midpoint = [np.mean([p1[0],p2[0]]),np.mean([p1[1],p2[1]])]
            dist_um = 500 * hfw_scale*constants.SI_TO_MICRO
            text = {
                "string": [f"{dist_um:.2f} um"],
                "color": "white"
            }

            # creating initial layers 

            self.ruler_layer = self.viewer.add_points(data, size=20, face_color='lime', edge_color='white', name=RULER_LAYER_NAME)
            self.viewer.add_shapes(data, shape_type='line', edge_color='lime', name=RULER_LINE_LAYER_NAME,edge_width=5)
            self.viewer.add_points(midpoint,text=text, size=20, face_color='transparent', edge_color='transparent', name=RULER_VALUE_LAYER_NAME)
            self.ruler_layer.mode = 'select'


            self.viewer.layers.selection.active = self.ruler_layer
            self.ruler_layer.mouse_drag_callbacks.append(self.update_ruler_points)


        else:
            self.ion_ruler_label.setText("Ruler: is off")
            self.ion_ruler_label.setVisible(False)
            self.viewer.layers.remove(self.ruler_layer)
            self.viewer.layers.remove(RULER_LINE_LAYER_NAME)
            self.viewer.layers.remove(RULER_VALUE_LAYER_NAME)
            self.ruler_layer = None
            self.restore_active_layer_for_movement()

    def update_ruler_points(self,layer, event):

        dragged = False
        yield

        def check_point_image_in_eb(point: Tuple[int,int]) -> bool:
            if point[1] >= 0 and point[1] <= self.eb_layer.data.shape[1]:
                return True
            else:
                return False

        while event.type == 'mouse_move':


            if self.ruler_layer.selected_data is not None:
                data = self.ruler_layer.data


                p1 = data[0]
                p2 = data[1]

                dist_pix = np.linalg.norm(p1-p2)
                
                midpoint = [(np.mean([p1[0],p2[0]])),(np.mean([p1[1],p2[1]]))]
                
                self.viewer.layers['ruler_line'].data = [p1,p2]
                self.viewer.layers['ruler_value'].data = midpoint
                
                hfw_scale = self.eb_image.metadata.pixel_size.x if check_point_image_in_eb(p1) else self.ib_image.metadata.pixel_size.x

                dist_um = dist_pix * hfw_scale*constants.SI_TO_MICRO

                text = {
                "string": [f"{dist_um:.2f} um"],
                "color": "white"
                }

                self.viewer.layers['ruler_value'].text = text
                dist_dx = abs(p2[1]-p1[1]) * self.image_settings.hfw/self.image_settings.resolution[0]*constants.SI_TO_MICRO
                dist_dy = abs(p2[0]-p1[0]) * self.image_settings.hfw/self.image_settings.resolution[0]*constants.SI_TO_MICRO


                self.ion_ruler_label.setText(f"Ruler: {dist_um:.2f} um  dx: {dist_dx:.2f} um  dy: {dist_dy:.2f} um")
                self.viewer.layers.selection.active = self.ruler_layer
                self.viewer.layers['ruler_line'].refresh()

                
                dragged = True
                yield

    def update_labels(self):
        self.detector_contrast_label.setText(f"{self.detector_contrast_slider.value()}%")
        self.detector_brightness_label.setText(f"{self.detector_brightness_slider.value()}%")

    def apply_detector_settings(self):
        """Apply the detector settings from the UI to the microscope."""
        
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
        """Apply the beam settings from the UI to the microscope."""
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
        """Get the imaging, detector and beam settings from the UI"""

        resolution = list(map(int, self.comboBox_image_resolution.currentText().split("x")))

        # advanced imaging settings
        line_integration, scan_interlacing, frame_integration = None, None, None
        image_drift_correction = False

        if ENABLE_ADVANCED_IMAGING_SETTINGS:
            if self.checkBox_image_line_integration.isChecked():
                line_integration = self.spinBox_image_line_integration.value()
            if self.checkBox_image_scan_interlacing.isChecked():
                scan_interlacing = self.spinBox_image_scan_interlacing.value()
            if self.checkBox_image_frame_integration.isChecked():
                frame_integration = self.spinBox_image_frame_integration.value()
                image_drift_correction = self.checkBox_image_drift_correction.isChecked()

        # imaging settings
        self.image_settings = ImageSettings(
            resolution=resolution,
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            hfw=self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            beam_type=BeamType[self.selected_beam.currentText()],
            autocontrast=self.checkBox_image_use_autocontrast.isChecked(),
            autogamma=self.checkBox_image_use_autogamma.isChecked(),
            save=self.checkBox_image_save_image.isChecked(),
            path=Path(self.lineEdit_image_path.text()),
            filename=self.lineEdit_image_label.text(),
            line_integration=line_integration,
            scan_interlacing=scan_interlacing,
            frame_integration=frame_integration,
            drift_correction=image_drift_correction,
        )

        # detector settings
        self.detector_settings = FibsemDetectorSettings(
            type=self.detector_type_combobox.currentText(),
            mode=self.detector_mode_combobox.currentText(),
            brightness=self.detector_brightness_slider.value()*constants.FROM_PERCENTAGES,
            contrast=self.detector_contrast_slider.value()*constants.FROM_PERCENTAGES,
        )

        # beam settings
        self.beam_settings = BeamSettings(
            beam_type=BeamType[self.selected_beam.currentText()],
            working_distance=self.doubleSpinBox_working_distance.value()*constants.MILLI_TO_SI,
            beam_current=self.doubleSpinBox_beam_current.value()*constants.PICO_TO_SI,
            voltage=self.doubleSpinBox_beam_voltage.value()*constants.KILO_TO_SI,
            hfw = self.doubleSpinBox_image_hfw.value() * constants.MICRO_TO_SI,
            resolution=resolution,
            dwell_time=self.doubleSpinBox_image_dwell_time.value() * constants.MICRO_TO_SI,
            stigmation = Point(x = self.doubleSpinBox_stigmation_x.value(), 
                               y = self.doubleSpinBox_stigmation_y.value()),
            shift = Point(self.doubleSpinBox_shift_x.value() * constants.MICRO_TO_SI, 
                          self.doubleSpinBox_shift_y.value() * constants.MICRO_TO_SI),
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
        res = image_settings.resolution
        res_str = f"{res[0]}x{res[1]}"
        self.comboBox_image_resolution.setCurrentText(res_str) # TODO: handle when it doesn't match exactly?

        self.doubleSpinBox_image_dwell_time.setValue(image_settings.dwell_time * constants.SI_TO_MICRO)
        self.doubleSpinBox_image_hfw.setValue(image_settings.hfw * constants.SI_TO_MICRO)

        self.checkBox_image_use_autocontrast.setChecked(image_settings.autocontrast)
        self.checkBox_image_use_autogamma.setChecked(image_settings.autogamma)
        self.checkBox_image_save_image.setChecked(image_settings.save)
        self.lineEdit_image_path.setText(str(image_settings.path))
        self.lineEdit_image_label.setText(image_settings.filename)

        if image_settings.line_integration is not None:
            self.checkBox_image_line_integration.setChecked(True)
            self.spinBox_image_line_integration.setValue(image_settings.line_integration)
        if image_settings.scan_interlacing is not None:
            self.checkBox_image_scan_interlacing.setChecked(True)
            self.spinBox_image_scan_interlacing.setValue(image_settings.scan_interlacing)
        if image_settings.frame_integration is not None:
            self.checkBox_image_frame_integration.setChecked(True)
            self.spinBox_image_frame_integration.setValue(image_settings.frame_integration)
            self.checkBox_image_drift_correction.setChecked(image_settings.drift_correction)

        # detector settings
        self.detector_type_combobox.setCurrentText(detector_settings.type)
        self.detector_mode_combobox.setCurrentText(detector_settings.mode)
        self.detector_contrast_slider.setValue(int(detector_settings.contrast*100))
        self.detector_brightness_slider.setValue(int(detector_settings.brightness*100))
        
        # beam settings
        self.doubleSpinBox_beam_current.setValue(beam_settings.beam_current*constants.SI_TO_PICO)
        self.doubleSpinBox_beam_voltage.setValue(beam_settings.voltage*constants.SI_TO_KILO)
        self.spinBox_beam_scan_rotation.setValue(np.rad2deg(beam_settings.scan_rotation))
        
        self.doubleSpinBox_working_distance.setValue(beam_settings.working_distance*constants.METRE_TO_MILLIMETRE)
        self.doubleSpinBox_shift_x.setValue(beam_settings.shift.x * constants.SI_TO_MICRO)
        self.doubleSpinBox_shift_y.setValue(beam_settings.shift.y * constants.SI_TO_MICRO)
        self.doubleSpinBox_stigmation_x.setValue(beam_settings.stigmation.x)
        self.doubleSpinBox_stigmation_y.setValue(beam_settings.stigmation.y)
        
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

        IS_FIB = bool(beam_type is BeamType.ION)
        self.comboBox_presets.setVisible(IS_FIB and self.IS_TESCAN)
        self.label_presets.setVisible(IS_FIB and self.IS_TESCAN)

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

    def _toggle_interactions(self, enable: bool, caller: str = None, imaging: bool = False):
        for btn in self.acquisition_buttons:
            btn.setEnabled(enable)
        self.set_detector_button.setEnabled(enable)
        self.button_set_beam_settings.setEnabled(enable)
        self.parent.movement_widget._toggle_interactions(enable, caller="ui")
        if caller != "milling":
            self.parent.milling_widget._toggle_interactions(enable, caller="ui")
        if enable:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.GREEN_PUSHBUTTON_STYLE)
            self.set_detector_button.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
            self.button_set_beam_settings.setStyleSheet(stylesheets.BLUE_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setText("Acquire Image")
            self.pushButton_take_all_images.setText("Acquire All Images")
            self.pushButton_acquire_sem_image.setText("Acquire SEM Image")
            self.pushButton_acquire_fib_image.setText("Acquire FIB Image")
        elif imaging:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.ORANGE_PUSHBUTTON_STYLE)
                btn.setText("Acquiring...")
            self.set_detector_button.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.button_set_beam_settings.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
        else:
            for btn in self.acquisition_buttons:
                btn.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.set_detector_button.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.button_set_beam_settings.setStyleSheet(stylesheets.DISABLED_PUSHBUTTON_STYLE)
            self.pushButton_take_image.setText("Acquire Image")
            self.pushButton_take_all_images.setText("Acquire All Images")

    def handle_acquisition_progress_update(self, ddict: dict):
        """Handle the acquisition progress update"""
        logging.debug(f"Acquisition Progress Update: {ddict}")
        
        msg = ddict.get("msg", None)
        if msg is not None:
            logging.debug(msg)
            napari.utils.notifications.notification_manager.records.clear()
            napari.utils.notifications.show_info(msg)

        # TODO: implement progress bar for acquisition

    def acquisition_finished(self):
        """Imaging has finished, update the viewer and re-enable interactions"""
        if self.ib_image is not None:
            self.update_viewer(self.ib_image.data, BeamType.ION)
        if self.eb_image is not None:
            self.update_viewer(self.eb_image.data, BeamType.ELECTRON)
        self._toggle_interactions(True)
        self.is_acquiring = False

    def acquire_sem_image(self):
        """Acquire an SEM image with the current settings"""
        self.start_acquisition(both=False, beam_type=BeamType.ELECTRON)
    
    def acquire_fib_image(self):
        """Acquire an FIB image with the current settings"""
        self.start_acquisition(both=False, beam_type=BeamType.ION)

    def acquire_image(self):
        """Acquire a single image with the current settings"""
        self.start_acquisition(both=False)

    def acquire_reference_images(self):
        """Acquire both SEM and FIB images with the current settings."""
        self.start_acquisition(both=True)

    def start_acquisition(self, both: bool = False, beam_type: Optional[BeamType] = None):
        """Start the image acquisition process"""
        # disable interactions
        self.is_acquiring = True
        self._toggle_interactions(enable=False, imaging=True)
        
        # get imaging settings from ui
        self.image_settings = self.get_settings_from_ui()[0] # TODO: QUERY why assigniong to image_settings?
        if beam_type is not None:
            self.image_settings.beam_type = beam_type

        # start the acquisition worker
        worker = self.acquisition_worker(self.image_settings, both=both)
        worker.finished.connect(self.acquisition_finished)
        worker.start()

    @thread_worker
    def acquisition_worker(self, image_settings: ImageSettings, both: bool = False):
        """Threaded image acquisition worker"""

        msg = "Acquiring Both Images..." if both else "Acquiring Image..."
        self.acquisition_progress_signal.emit({"msg": msg})

        if both:
            self.eb_image, self.ib_image = acquire.take_reference_images(self.microscope, image_settings)
        else:
            image = acquire.new_image(self.microscope, image_settings)
            if image_settings.beam_type is BeamType.ELECTRON:
                self.eb_image = image
            if image_settings.beam_type is BeamType.ION:
                self.ib_image = image
        
        self.acquisition_progress_signal.emit({"finished": True})

        logging.debug({"msg": "acquisition_worker", "image_settings": self.image_settings.to_dict()})

    def update_ui_tools(self):
        """Redraw the ui tools (scalebar, crosshair)"""

        # draw scalebar and crosshair
        if self.eb_image is not None and self.ib_image is not None:
            draw_scalebar_in_napari(
                viewer=self.viewer,
                eb_image=self.eb_image,
                ib_image=self.ib_image,
                is_checked=self.scalebar_checkbox.isChecked(),
            )
            draw_crosshair_in_napari(
                viewer=self.viewer,
                sem_shape=self.eb_image.data.shape,
                fib_shape=self.ib_image.data.shape,
                is_checked=self.crosshair_checkbox.isChecked(),
            ) 

    def update_viewer(self, arr: np.ndarray, beam_type: BeamType, set_ui_from_image: bool = False):
        """Update the viewer with the given image array"""

        # median filter for display
        arr = median_filter(arr, size=3)

        try:
            self.viewer.layers[beam_type.name].data = arr
        except KeyError:    
            layer = self.viewer.add_image(arr, name = beam_type.name, blending='additive')

        layer = self.viewer.layers[beam_type.name]
        if self.eb_layer is None and beam_type is BeamType.ELECTRON:
            self.eb_layer = layer
        if self.ib_layer is None and beam_type is BeamType.ION:
            self.ib_layer = layer

        # centre the camera
        if self.eb_layer:
            self.viewer.reset_view()

        if self.ib_layer:
            translation = (
                self.viewer.layers[BeamType.ELECTRON.name].data.shape[1]
                if self.eb_layer
                else arr.shape[1]
            )
            self.ib_layer.translate = [0.0, translation]       


        if self.eb_layer:
            points = np.array([[-20, 200], [-20, self.ib_layer.translate[1] + 150]])

            try:
                self.viewer.layers['label'].data = points
            except KeyError:    
                self.viewer.add_points(
                    data=points,
                    name=IMAGE_TEXT_LAYER_PROPERTIES["name"],
                    size=IMAGE_TEXT_LAYER_PROPERTIES["size"],
                    text=IMAGE_TEXT_LAYER_PROPERTIES["text"],
                    edge_width=IMAGE_TEXT_LAYER_PROPERTIES["edge_width"],
                    edge_width_is_relative=IMAGE_TEXT_LAYER_PROPERTIES[
                        "edge_width_is_relative"
                    ],
                    edge_color=IMAGE_TEXT_LAYER_PROPERTIES["edge_color"],
                    face_color=IMAGE_TEXT_LAYER_PROPERTIES["face_color"],
                )   

        # set ui from image metadata
        if set_ui_from_image:
            if beam_type is BeamType.ELECTRON:
                self.image_settings = self.eb_image.metadata.image_settings
                beam_settings = self.eb_image.metadata.microscope_state.electron_beam
                detector_settings = (
                    self.eb_image.metadata.microscope_state.electron_detector
                )
                beam_type = BeamType.ELECTRON
            if beam_type is BeamType.ION:
                self.image_settings = self.ib_image.metadata.image_settings
                beam_settings = self.ib_image.metadata.microscope_state.ion_beam
                detector_settings = self.ib_image.metadata.microscope_state.ion_detector
                beam_type = BeamType.ION
        else:
            beam_type = BeamType[self.selected_beam.currentText()]
            beam_settings, detector_settings = None, None
            # QUERY: what is this doing?

        self.set_ui_from_settings(
            image_settings=self.image_settings,
            beam_type=beam_type,
            beam_settings=beam_settings,
            detector_settings=detector_settings,
        )

        # set the active layer to the electron beam (for movement)
        self.restore_active_layer_for_movement()

        self.viewer_update_signal.emit()

    def get_data_from_coord(self, coords: Tuple[float, float]) -> Tuple[Tuple[float, float], BeamType, FibsemImage]:
        
        # TODO: change this to use the image layers, and extent

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

    def restore_active_layer_for_movement(self):
        """Restore the active layer to the electron beam for movement"""
        if self.eb_layer in self.viewer.layers:
            self.viewer.layers.selection.active = self.eb_layer

    def clear_alignment_area(self):
        """Hide the alignment area layer"""
        self.alignment_layer.visible = False
        self.restore_active_layer_for_movement()

    def toggle_alignment_area(self, reduced_area: FibsemRectangle = None, editable: bool = True):
        """Toggle the alignment area layer to selection mode, and display the alignment area."""
        if self.viewer.layers.selection.active == self.eb_layer:
            logging.debug(f"Alignment area being set to: {reduced_area}")
            self.set_alignment_layer(reduced_area, editable=editable)
            self.alignment_layer.visible = True
        else:
            self.clear_alignment_area()

    def set_alignment_layer(self, reduced_area: FibsemRectangle = FibsemRectangle(0.25, 0.25, 0.5, 0.5), 
                            editable: bool = True):
        """Set the alignment area layer in napari."""

        # add alignment area to napari
        data = convert_reduced_area_to_napari_shape(reduced_area=reduced_area, 
                                                    image_shape=self.ib_image.data.shape, 
                                                   )
        if self.alignment_layer is None:
            self.alignment_layer = self.viewer.add_shapes(data=data, 
                                                          name=ALIGNMENT_LAYER_PROPERTIES["name"], 
                        shape_type=ALIGNMENT_LAYER_PROPERTIES["shape_type"], 
                        edge_color=ALIGNMENT_LAYER_PROPERTIES["edge_color"], 
                        edge_width=ALIGNMENT_LAYER_PROPERTIES["edge_width"], 
                        face_color=ALIGNMENT_LAYER_PROPERTIES["face_color"], 
                        opacity=ALIGNMENT_LAYER_PROPERTIES["opacity"], 
                        translate=self.ib_layer.translate) # match the ib layer translation
            self.alignment_layer.metadata = ALIGNMENT_LAYER_PROPERTIES["metadata"]
        else:
            self.alignment_layer.data = data

        if editable:
            self.viewer.layers.selection.active = self.alignment_layer
            self.alignment_layer.mode = "select"
        # TODO: prevent rotation of rectangles?  
        self.alignment_layer.events.data.connect(self.update_alignment)

    def update_alignment(self, event):
        """Validate the alignment area, and update the parent ui."""
        reduced_area = self.get_alignment_area()
        is_valid = reduced_area.is_valid_reduced_area

        logging.info(f"Updated alignment area: {reduced_area}, valid: {is_valid}")

        # show valid/invalid message in parent # TODO: make this more generic
        if self.parent.__class__.__name__ == "AutoLamellaUI":
            logging.info("Parent is AutoLamellaUI")
            try:
                msg = "Edit Alignment Area. Press Continue when done." 
                if not is_valid:
                    msg = "Invalid Alignment Area. Please adjust inside FIB Image."
                self.parent.label_instructions.setText(msg)
                self.parent.pushButton_yes.setEnabled(is_valid)
            except Exception as e:
                logging.info(f"Error updating alignment area: {e}")

        return reduced_area, is_valid

    def get_alignment_area(self) -> FibsemRectangle:
        """Get the alignment area from the alignment layer."""
        data = self.alignment_layer.data
        if data is None:
            return None
        data = data[0]
        return convert_shape_to_image_area(data, self.ib_image.data.shape)
    

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
