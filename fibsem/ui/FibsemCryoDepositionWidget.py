import logging
import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.microscope import FibsemMicroscope, TescanMicroscope, ThermoMicroscope, DemoMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.ui import FibsemCryoDepositionWidget_qt
from fibsem import gis


class FibsemCryoDepositionWidget(FibsemCryoDepositionWidget_qt.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        parent=None,
    ):
        super(FibsemCryoDepositionWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle("Cryo Deposition")

        self.microscope = microscope
        self.settings = settings

        self.setup_connections()

    def setup_connections(self):

        self.pushButton_run_sputter.clicked.connect(self.run_sputter)

        positions = utils.load_yaml(cfg.POSITION_PATH)
        self.comboBox_sputter_stage_position.addItems([p["name"] for p in positions])


    def _get_protocol_from_ui(self):
        
        protocol = {
            "application_file": self.lineEdit_sputter_application_file.text(),
            "gas": self.lineEdit_sputter_gas.text(),
            "position": self.lineEdit_sputter_gis_position.text(),
            "name": self.comboBox_sputter_stage_position.currentText(),
            "time": self.doubleSpinBox_sputter_time.value(),
            "hfw": self.doubleSpinBox_sputter_hfw.value() * constants.MICRO_TO_SI,
            "beam_current": self.doubleSpinBox_sputter_beam_current.value() * constants.NANO_TO_SI,
            "length": self.doubleSpinBox_sputter_length.value() * constants.MICRO_TO_SI,

        }

        return protocol

    # TODO: thread this
    def run_sputter(self):
        
        protocol = self._get_protocol_from_ui()

        gis.cryo_deposition(self.microscope, protocol, name=protocol["name"])




def main():

    viewer = napari.Viewer(ndisplay=2)
    microscope, settings = utils.setup_session()
    cryo_sputter_widget = FibsemCryoDepositionWidget(microscope, settings)
    viewer.window.add_dock_widget(
        cryo_sputter_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()