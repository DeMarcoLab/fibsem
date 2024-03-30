import logging
import napari
import napari.utils.notifications
import numpy as np
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.microscope import FibsemMicroscope, TescanMicroscope, ThermoMicroscope, DemoMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.ui.qtdesigner_files import FibsemCryoDepositionWidget
from fibsem import gis


class FibsemCryoDepositionWidget(FibsemCryoDepositionWidget.Ui_Dialog, QtWidgets.QDialog):
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
        self.comboBox_stage_position.addItems([p["name"] for p in positions])
        available_ports = self.microscope.get_available_values("gis_ports")
        self.comboBox_port.addItems([str(p) for p in available_ports])


        # TODO: show / hide based on gis / multichem available
        
        multichem_available = self.microscope.is_available("gis_multichem")
        self.lineEdit_insert_position.setVisible(multichem_available)
        self.label_insert_position.setVisible(multichem_available)

    def _get_protocol_from_ui(self):
        
        protocol = {
            "port": self.comboBox_port.currentText(),
            "gas": self.lineEdit_gas.text(),
            "insert_position": self.lineEdit_insert_position.text(),
            "duration": self.doubleSpinBox_duration.value(),
            "name": self.comboBox_stage_position.currentText(),

        }

        return protocol

    # TODO: thread this
    def run_sputter(self):
        
        from fibsem.structures import FibsemGasInjectionSettings
        gdict = self._get_protocol_from_ui()
        gis_settings = FibsemGasInjectionSettings.from_dict(gdict)
        gis.cryo_deposition_v2(self.microscope, gis_settings, name=gdict["name"])


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