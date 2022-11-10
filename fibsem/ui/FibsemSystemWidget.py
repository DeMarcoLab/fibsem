import os

import fibsem
import napari
import napari.utils.notifications
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem.structures import  MicroscopeSettings
from fibsem.ui.qtdesigner_files import FibsemSystemWidget
from PyQt5 import QtWidgets

import logging

class FibsemSystemWidget(FibsemSystemWidget.Ui_Form, QtWidgets.QWidget):
    def __init__(
        self,
        settings: MicroscopeSettings = None,
        parent=None,
    ):
        super(FibsemSystemWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.setup_connections()


    def setup_connections(self):

        print("setup connections")
        self.pushButton_connect.clicked.connect(self.connect_to_microscope)


    def connect_to_microscope(self):
        print("connect to microscope")
        # self.microscope, self.settings = utils.setup_session()

    # connect to microscope

    # apply settings / validate


def main():

    viewer = napari.Viewer(ndisplay=2)
    system_widget = FibsemSystemWidget()
    viewer.window.add_dock_widget(
        system_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
