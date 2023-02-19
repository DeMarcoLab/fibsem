import os

import fibsem
import napari
import napari.utils.notifications
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import calibration, constants, utils
from fibsem.structures import BeamType, GammaSettings, ImageSettings, MultiChemSettings
from fibsem.ui.qtdesigner_files import FibsemToolsWidget
from PyQt5 import QtWidgets

import logging

class FibsemToolsWidget(FibsemToolsWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
        parent=None,
    ):
        super(FibsemToolsWidget, self).__init__(parent=parent)
        self.setupUi(self)

        # self.microscope = microscope

        # if microscope is None:
        #     self.microscope, self.settings = utils.setup_session()
        # else:
        #     self.microscope, self.settings = microscope, None
        self.setup_connections()


    def setup_connections(self):

        print("setup connections")
        self.pushButton_sputter_multichem.clicked.connect(self.run_tools)
        self.pushButton_move_stage_out.clicked.connect(self.run_tools)
        self.pushButton_auto_discharge_beam.clicked.connect(self.run_tools)
        self.pushButton_auto_focus_beam.clicked.connect(self.run_tools)
        self.pushButton_auto_home_stage.clicked.connect(self.run_tools)
        self.pushButton_auto_link_stage.clicked.connect(self.run_tools)
        self.pushButton_auto_needle_calibration.clicked.connect(self.run_tools)
        self.pushButton_validate_microscope_settings.clicked.connect(self.run_tools)


    ##### TOOLS
    def run_tools(self):

        sender = self.sender()
        logging.info(f"Sender: {sender}")

        if sender == self.pushButton_sputter_multichem:
            logging.info(f"Sputtering MultiChem")

        if sender == self.pushButton_move_stage_out:
            logging.info(f"Moving Stage Out")

        if sender == self.pushButton_auto_focus_beam:
            logging.info(f"Auto Focus")

        if sender == self.pushButton_auto_discharge_beam:
            logging.info(f"Auto Discharge Beam")
            # calibration.auto_discharge_beam(self.microscope, self.settings.image)

        if sender == self.pushButton_auto_home_stage:
            logging.info(f"Auto Home Stage")

        if sender == self.pushButton_auto_link_stage:
            logging.info(f"Auto Link Stage")
        
        if sender == self.pushButton_auto_needle_calibration:
            logging.info(f"Auto Needle Calibration")

        if sender == self.pushButton_validate_microscope_settings:
            logging.info(f"Validating Microscope Settings")
        

def main():

    viewer = napari.Viewer(ndisplay=2)
    tools_widget = FibsemToolsWidget()
    viewer.window.add_dock_widget(
        tools_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
