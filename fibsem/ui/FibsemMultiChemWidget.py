from ast import Mult
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
    MultiChemSettings,
)
from fibsem.ui.qtdesigner_files import FibsemMultiChemWidget
from PyQt5 import QtWidgets

BASE_PATH = os.path.join(os.path.dirname(fibsem.__file__), "config")


class FibsemMultiChemWidget(FibsemMultiChemWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
        image_settings: ImageSettings = None,
        parent=None,
    ):
        super(FibsemMultiChemWidget, self).__init__(parent=parent)
        self.setupUi(self)


        self.microscope = microscope

        self.setup_connections()
        self.update_ui_from_settings()

    def setup_connections(self):

        print("setup connections")

        self.comboBox_beam_type.addItems([beam.name for beam in BeamType])
        self.comboBox_application_file.addItems(["cryo_Pt_dep"])
        self.comboBox_gas.addItems(["Pt cryo"])
        self.comboBox_position.addItems(["cryo"])
        
        self.pushButton_run_sputtering.clicked.connect(self.run_sputtering)


    def update_ui_from_settings(self):

        self.multichem_settings = MultiChemSettings()
        self.comboBox_beam_type.setCurrentText(self.multichem_settings.beam_type.name)
        self.comboBox_application_file.setCurrentText(self.multichem_settings.application_file)
        self.comboBox_gas.setCurrentText(self.multichem_settings.gas)
        self.comboBox_position.setCurrentText(self.multichem_settings.position)
        self.doubleSpinBox_hfw.setValue(self.multichem_settings.hfw * constants.SI_TO_MICRO)
        self.doubleSpinBox_length.setValue(self.multichem_settings.length * constants.SI_TO_MICRO)
        self.doubleSpinBox_time.setValue(self.multichem_settings.time)


    def get_settings_from_ui(self):

        print("get settings")

        mutltichem_settings = MultiChemSettings(
            application_file=self.comboBox_application_file.currentText(),
            gas = self.comboBox_gas.currentText(),
            position = self.comboBox_position.currentText(),
            beam_type= BeamType[self.comboBox_beam_type.currentText()],
            hfw = self.doubleSpinBox_hfw.value() * constants.MICRO_TO_SI,
            length = self.doubleSpinBox_length.value() * constants.MICRO_TO_SI,
            time = self.doubleSpinBox_time.value()
        )


        print(mutltichem_settings)

        return mutltichem_settings

    def run_sputtering(self):

        mutltichem_settings = self.get_settings_from_ui()

        print("run sputtering") 

        from fibsem import utils

        # utils.sputter_multichem_v2(self.microscope, multichem_settings)


        

def main():

    viewer = napari.Viewer(ndisplay=2)
    multichem_widget = FibsemMultiChemWidget()
    viewer.window.add_dock_widget(
        multichem_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
