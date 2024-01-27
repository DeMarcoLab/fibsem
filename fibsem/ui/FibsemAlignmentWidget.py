
import logging
from copy import deepcopy
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets

import numpy as np
from fibsem import alignment
from fibsem.microscope import FibsemMicroscope, ThermoMicroscope, DemoMicroscope, TescanMicroscope
from fibsem.structures import (BeamType, MicroscopeSettings)
from fibsem.ui.qtdesigner_files import CurrentAlignmentWidget
from fibsem.ui import utils as ui_utils 
from fibsem.ui import _stylesheets



_UNSCALED_VALUES  = ["rotation", "size_ratio", "scan_direction", "cleaning_cross_section", "number"]
_ANGLE_KEYS = ["rotation"]
def _scale_value(key, value, scale):
    if key not in _UNSCALED_VALUES:
        return value * scale    
    return value


class FibsemAlignmentWidget(CurrentAlignmentWidget.Ui_BeamAlignment, QtWidgets.QWidget):
    # milling_param_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemAlignmentWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer
        self.parent = parent

        if isinstance(self.microscope, ThermoMicroscope) or isinstance(self.microscope, DemoMicroscope):
            currents = self.microscope.get_available_values("current", BeamType.ION)
            string_list = [str(num) for num in currents]
            self.comboBox_aligned_current.addItems(string_list)
            self.comboBox_ref_current.addItems(string_list)
        elif isinstance(self.microscope, TescanMicroscope):
            presets = self.microscope.get_available_values("presets", BeamType.ION)
            self.comboBox_aligned_current.addItems(presets)
            self.comboBox_ref_current.addItems(presets)
            self.label_2.setText("Reference Preset")
            self.label_3.setText("Aligned Preset")

        self.setup_connections()
        self.ref_layer = None
        self.aligned_layer = None   

    def setup_connections(self):
        self.checkBox_overlay.stateChanged.connect(self.update_overlay)
        self.pushButton_align_beam.clicked.connect(self.align_beam)
        self.pushButton_take_images.clicked.connect(self.take_images)
        self.pushButton_take_images.setStyleSheet(_stylesheets._GRAY_PUSHBUTTON_STYLE)
        self.pushButton_align_beam.setEnabled(False)
        self.pushButton_align_beam.setStyleSheet(_stylesheets._DISABLED_PUSHBUTTON_STYLE)

    def update_overlay(self):
        if self.checkBox_overlay.isChecked():
            self.aligned_layer.translate = [0.0, 0.0] 
            self.aligned_layer.opacity = 0.5
        else:
            self.aligned_layer.opacity = 1.0
            self.aligned_layer.translate = [0.0, self.ref_layer.data.shape[1] if self.ref_layer
                else self.aligned_layer.data.shape[1]]

    def align_beam(self):
        self.pushButton_align_beam.setEnabled(False)
        self.pushButton_align_beam.setText("Aligning...")
        self.pushButton_align_beam.setStyleSheet("color: orange")
        if isinstance(self.microscope, ThermoMicroscope) or isinstance(self.microscope, DemoMicroscope):
            self.microscope.set("current", float(self.comboBox_aligned_current.currentText()), beam_type=BeamType.ION)
        elif isinstance(self.microscope, TescanMicroscope):
            self.microscope.set("preset", self.comboBox_aligned_current.currentText(), beam_type=BeamType.ION)
        alignment.beam_shift_alignment(
            microscope=self.microscope,
            image_settings=self.settings.image,
            ref_image=self.ref_image,
            reduced_area=None,
        )
        shift = self.microscope.get("shift", BeamType.ION)
        ui_utils.message_box_ui(
            title="Beam Shift Alignment Complete",
            text=f"Beam Shifted by {shift.x}m, {shift.y}m",
            buttons=QtWidgets.QMessageBox.Ok,
        )
        string = self.listdone.toPlainText()
        string += (f"Aligned reference {self.comboBox_ref_current.currentText()} with {self.comboBox_aligned_current.currentText()}. Beam Shifted by {shift.x}m, {shift.y}m \n")
        self.listdone.setPlainText(string)
        self.take_images()
        self.parent.ref_current = self.comboBox_ref_current.currentText()
        self.pushButton_align_beam.setEnabled(True)
        self.pushButton_align_beam.setText("Align Beam")
        self.pushButton_align_beam.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)

    def take_images(self):
        if isinstance(self.microscope, ThermoMicroscope) or isinstance(self.microscope, DemoMicroscope):
            self.microscope.set("current", float(self.comboBox_ref_current.currentText()), beam_type=BeamType.ION)
        elif isinstance(self.microscope, TescanMicroscope):
            self.microscope.set("preset", self.comboBox_ref_current.currentText(), beam_type=BeamType.ION)        
        self.settings.image.beam_type = BeamType.ION
        self.ref_image = self.microscope.acquire_image(self.settings.image)
        self.update_viewer(self.ref_image.data, "Reference")
        if isinstance(self.microscope, ThermoMicroscope) or isinstance(self.microscope, DemoMicroscope):
            self.microscope.set("current", float(self.comboBox_aligned_current.currentText()), beam_type=BeamType.ION)
        elif isinstance(self.microscope, TescanMicroscope):
            self.microscope.set("preset", self.comboBox_aligned_current.currentText(), beam_type=BeamType.ION)
        self.settings.image.beam_type = BeamType.ION
        self.aligned_image = self.microscope.acquire_image(self.settings.image)
        self.update_viewer(self.aligned_image.data, "Aligned")
        self.pushButton_align_beam.setEnabled(True)
        self.pushButton_align_beam.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)


    def update_viewer(self, arr: np.ndarray, name: str = None):
        # arr = ui_utils._draw_crosshair(arr)

        try:
            self.viewer.layers[name].data = arr
        except:    
            layer = self.viewer.add_image(arr, name = name, colormap= "red" if name == "Reference" else "green")
        

        layer = self.viewer.layers[name]
        if self.ref_layer is None and name == "Reference":
            self.ref_layer = layer
        if self.aligned_layer is None and name == "Aligned":
            self.aligned_layer = layer
        
        # centre the camera
        if self.ref_layer:
            self.viewer.camera.center = [
                0.0,
                self.ref_layer.data.shape[0] / 2,
                self.ref_layer.data.shape[1],
            ]
            self.viewer.camera.zoom = 0.45

        if self.aligned_layer:
            translation = (
                self.viewer.layers["Reference"].data.shape[1]
                if self.ref_layer
                else arr.shape[1]
            )
            self.aligned_layer.translate = [0.0, translation]       

        if self.ref_layer:
            points = np.array([[-20, 200], [-20, self.ref_layer.data.shape[1] + 150]])
            string = ["REFERENCE", "ALIGNED"]
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

def main():
    pass


if __name__ == "__main__":
    main()