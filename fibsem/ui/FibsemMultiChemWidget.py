import os

import fibsem
import napari
import napari.utils.notifications
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import calibration, constants, utils
from fibsem.structures import BeamType, GammaSettings, ImageSettings, MultiChemSettings
from fibsem.ui.qtdesigner_files import FibsemMultiChemWidget
from PyQt5 import QtWidgets


class FibsemMultiChemWidget(FibsemMultiChemWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
        mc_settings: MultiChemSettings = None,
        parent=None,
    ):
        super(FibsemMultiChemWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.microscope = microscope

        self.setup_connections()
        if mc_settings is not None:
            self.update_ui_from_settings(mc_settings)

    def setup_connections(self):

        print("setup connections")

        self.comboBox_beam_type.addItems([beam.name for beam in BeamType])
        self.comboBox_application_file.addItems(["cryo_Pt_dep"])
        self.comboBox_gas.addItems(["Pt cryo"])
        self.comboBox_position.addItems(["cryo"])

        self.pushButton_run_sputtering.clicked.connect(self.run_sputtering)

    def update_ui_from_settings(self, mc_settings: MultiChemSettings):

        self.comboBox_beam_type.setCurrentText(mc_settings.beam_type.name)
        self.comboBox_application_file.setCurrentText(mc_settings.application_file)
        self.comboBox_gas.setCurrentText(mc_settings.gas)
        self.comboBox_position.setCurrentText(mc_settings.position)
        self.doubleSpinBox_hfw.setValue(mc_settings.hfw * constants.SI_TO_MICRO)
        self.doubleSpinBox_length.setValue(mc_settings.length * constants.SI_TO_MICRO)
        self.doubleSpinBox_time.setValue(mc_settings.time)

    def get_settings_from_ui(self):

        mc_settings = MultiChemSettings(
            application_file=self.comboBox_application_file.currentText(),
            gas=self.comboBox_gas.currentText(),
            position=self.comboBox_position.currentText(),
            beam_type=BeamType[self.comboBox_beam_type.currentText()],
            hfw=self.doubleSpinBox_hfw.value() * constants.MICRO_TO_SI,
            length=self.doubleSpinBox_length.value() * constants.MICRO_TO_SI,
            time=self.doubleSpinBox_time.value(),
        )

        print(f"get settings: {mc_settings}")

        return mc_settings

    def run_sputtering(self):

        mc_settings = self.get_settings_from_ui()

        print("run sputtering")

        from fibsem import utils

        # utils.sputter_multichem_v2(self.microscope, multichem_settings)


def main():

    viewer = napari.Viewer(ndisplay=2)
    multichem_widget = FibsemMultiChemWidget(mc_settings=MultiChemSettings())
    viewer.window.add_dock_widget(
        multichem_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
