import logging

import napari
import napari.utils.notifications
from fibsem import utils
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets

from fibsem.ui.qtdesigner_files import FibsemUI

class FibsemUI(FibsemUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super(FibsemUI, self).__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.setup_connections()

        self.microscope = None
        self.settings = None

        self.update_ui()


    def setup_connections(self):

        self.pushButton.clicked.connect(self.connect_to_microscope)

        self.comboBox_manufacturer.addItems(["Thermo", "Tescan", "Demo"])


    def update_ui(self):

        _microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(1, _microscope_connected)
        self.tabWidget.setTabVisible(2, _microscope_connected)
        self.tabWidget.setTabVisible(3, _microscope_connected)

        if _microscope_connected:
            self.pushButton.setStyleSheet("background-color: green")
            self.pushButton.setText("Microscope Connected")
            self.pushButton.setEnabled(False)
        else:
            self.pushButton.setStyleSheet("background-color: gray")
            self.pushButton.setText("Connect to Microscope")

    def connect_to_microscope(self):

        ip_address = self.lineEdit_ip_address.text()
        manufacturer = self.comboBox_manufacturer.currentText()

        try:
            self.microscope, self.settings = utils.setup_session(ip_address=ip_address, 
                                                                 manufacturer=manufacturer)  # type: ignore
        except Exception as e:
            msg = f"Could not connect to microscope: {e}"
            logging.exception(msg)
            napari.utils.notifications.show_info(msg)
            return
        
        self.update_microscope_ui()
        self.update_ui()

    def update_microscope_ui(self):

        if self.microscope is not None:
            # reusable components
            self.image_widget = FibsemImageSettingsWidget(
                microscope=self.microscope,
                image_settings=self.settings.image,
                viewer=self.viewer,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                image_widget=self.image_widget,
            )
            self.milling_widget = FibsemMillingWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                image_widget=self.image_widget,
            )

            self.gridLayout_imaging_tab.addWidget(self.image_widget, 0, 0)
            self.gridLayout_movement_tab.addWidget(self.movement_widget, 0, 0)
            self.gridLayout_milling_tab.addWidget(self.milling_widget, 0, 0)






def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, area="right", add_vertical_stretch=False)
    napari.run()


if __name__ == "__main__":
    main()



