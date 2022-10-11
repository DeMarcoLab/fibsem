import logging
import sys
from enum import Enum
import traceback

import numpy as np
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.structures import (MoveSettings,
                                                         StagePosition)
from fibsem import acquire, conversions, movement, constants, alignment, utils, milling, calibration
from fibsem.structures import BeamSystemSettings, BeamType, MicroscopeSettings, MillingSettings, Point
from fibsem.ui.qtdesigner_files import FibsemSettings
from PyQt5 import QtCore, QtWidgets
from fibsem.detection.detection import DetectionType, DetectionFeature, DetectionResult
from fibsem.detection import detection
from fibsem.detection import utils as det_utils

import napari.utils.notifications
import napari

from pprint import pprint


class MovementMode(Enum):
    Stable = 1
    Eucentric = 2

class FibsemSettings(FibsemSettings.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self, viewer: napari.Viewer, microscope: SdbMicroscopeClient = None, 
        settings: MicroscopeSettings = None, parent = None
    ):
        super(FibsemSettings, self).__init__(parent = parent)
        self.setupUi(self)

        # connect to microscope, if required
        if microscope is None:
            self.microscope, self.settings = utils.setup_session()
    
        self.setup_connections()
        self.update_ui_from_settings()


    def setup_connections(self):

        # TODO: set limits based on microscope

        # buttons   
        self.pushButton_save_settings.clicked.connect(self.save_settings)
        self.pushButton_get_electron_settings.clicked.connect(self.get_beam_settings)
        self.pushButton_get_ion_settings.clicked.connect(self.get_beam_settings)


        # system
        app_files = sorted(self.microscope.patterning.list_all_application_files())
        self.comboBox_system_application_file.addItems(app_files)

        # stage


        # electron
        self.microscope.imaging.set_active_view(BeamType.ELECTRON.value)
        self.microscope.imaging.set_active_device(BeamType.ELECTRON.value)
        eb_detector_modes = self.microscope.detector.mode.available_values
        eb_detector_types = self.microscope.detector.type.available_values

        self.comboBox_electron_detector_mode.addItems(eb_detector_modes)
        self.comboBox_electron_detector_type.addItems(eb_detector_types)

        # ion
        self.microscope.imaging.set_active_view(BeamType.ION.value)
        self.microscope.imaging.set_active_device(BeamType.ION.value)
        ib_detector_modes = self.microscope.detector.mode.available_values
        ib_detector_types = self.microscope.detector.type.available_values
        ib_plasma_gases = self.microscope.beams.ion_beam.source.plasma_gas.available_values
        self.comboBox_ion_detector_mode.addItems(ib_detector_modes)
        self.comboBox_ion_detector_type.addItems(ib_detector_types)
        self.comboBox_ion_plasma_gas.addItems(ib_plasma_gases)

    def save_settings(self):


        logging.info(f"save settings...")



    def update_ui_from_settings(self):


        settings = self.settings.system


        # system
        self.lineEdit_system_ip_address.setText(settings.ip_address)
        self.comboBox_system_application_file.setCurrentText(settings.application_file)

        # stage
        self.spinBox_stage_rotation_flat_to_electron.setValue(settings.stage.rotation_flat_to_electron)
        self.spinBox_stage_rotation_flat_to_ion.setValue(settings.stage.rotation_flat_to_ion)
        self.spinBox_stage_pre_tilt.setValue(0) # TODO: fix when pre-titl implemented
        self.spinBox_stage_tilt_flat_to_electron.setValue(settings.stage.tilt_flat_to_electron)
        self.spinBox_stage_tilt_flat_to_ion.setValue(settings.stage.tilt_flat_to_ion)
        self.doubleSpinBox_stage_needle_height_limit.setValue(settings.stage.needle_stage_height_limit * constants.METRE_TO_MILLIMETRE)

        # electron
        self.doubleSpinBox_electron_voltage.setValue(settings.electron.voltage * constants.SI_TO_KILO)
        self.doubleSpinBox_electron_current.setValue(settings.electron.current * constants.SI_TO_PICO)
        self.comboBox_electron_detector_mode.setCurrentText(settings.electron.detector_mode)
        self.comboBox_electron_detector_type.setCurrentText(settings.electron.detector_type)        
        self.doubleSpinBox_electron_eucentric_height.setValue(settings.electron.eucentric_height * constants.METRE_TO_MILLIMETRE)

        # ion
        self.doubleSpinBox_ion_voltage.setValue(settings.ion.voltage * constants.SI_TO_KILO)
        self.doubleSpinBox_ion_current.setValue(settings.ion.current * constants.SI_TO_PICO)
        self.comboBox_ion_detector_mode.setCurrentText(settings.ion.detector_mode)
        self.comboBox_ion_detector_type.setCurrentText(settings.ion.detector_type)        
        self.doubleSpinBox_ion_eucentric_height.setValue(settings.ion.eucentric_height * constants.METRE_TO_MILLIMETRE)
        self.comboBox_ion_plasma_gas.setCurrentText(settings.ion.plasma_gas)


    

    def get_beam_settings(self):

        if self.sender() == self.pushButton_get_electron_settings:
            beam_type = BeamType.ELECTRON
        
        if self.sender() == self.pushButton_get_ion_settings:
            beam_type = BeamType.ION

        beam_settings = calibration.get_current_beam_system_state(self.microscope, beam_type)

        logging.info(f"getting settings: {beam_type}")
        logging.info(f"settings: {beam_settings}")

        # electron
        if beam_type is BeamType.ELECTRON:
            self.doubleSpinBox_electron_voltage.setValue(beam_settings.voltage * constants.SI_TO_KILO)
            self.doubleSpinBox_electron_current.setValue(beam_settings.current * constants.SI_TO_PICO)
            self.comboBox_electron_detector_mode.setCurrentText(beam_settings.detector_mode)
            self.comboBox_electron_detector_type.setCurrentText(beam_settings.detector_type)        
            self.doubleSpinBox_electron_eucentric_height.setValue(beam_settings.eucentric_height * constants.METRE_TO_MILLIMETRE)

        # ion
        if beam_type is BeamType.ION:
            self.doubleSpinBox_ion_voltage.setValue(beam_settings.voltage * constants.SI_TO_KILO)
            self.doubleSpinBox_ion_current.setValue(beam_settings.current * constants.SI_TO_PICO)
            self.comboBox_ion_detector_mode.setCurrentText(beam_settings.detector_mode)
            self.comboBox_ion_detector_type.setCurrentText(beam_settings.detector_type)        
            self.doubleSpinBox_ion_eucentric_height.setValue(beam_settings.eucentric_height * constants.METRE_TO_MILLIMETRE)
            self.comboBox_ion_plasma_gas.setCurrentText(beam_settings.plasma_gas)


# TODO: change ion beam current to comboBox

def main():
    
    app = QtWidgets.QApplication([])
    viewer = napari.Viewer(ndisplay=2)
    fibsem_settings = FibsemSettings(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_settings, area='right', add_vertical_stretch=False)  

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
