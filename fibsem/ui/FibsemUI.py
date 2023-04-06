import logging

import napari
import napari.utils.notifications
from fibsem import utils
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemGISWidget import FibsemGISWidget
from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets


from fibsem.microscope import FibsemMicroscope, MicroscopeSettings
from fibsem.ui.qtdesigner_files import FibsemUI
from fibsem import config as cfg


class FibsemUI(FibsemUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super(FibsemUI, self).__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.setup_connections()


        self.microscope: FibsemMicroscope = None
        self.settings: MicroscopeSettings = None

        self.image_widget: FibsemImageSettingsWidget = None
        self.movement_widget: FibsemMovementWidget = None
        self.milling_widget: FibsemMillingWidget = None
        self.manipulator_widget: FibsemManipulatorWidget = None
        self.GIS_widget: FibsemGISWidget = None


        self.update_ui()


    def setup_connections(self):

        self.pushButton.clicked.connect(self.connect_to_microscope)
        self.comboBox_manufacturer.addItems(cfg.__SUPPORTED_MANUFACTURERS__)

    def update_ui(self):

        _microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(1, _microscope_connected)
        self.tabWidget.setTabVisible(2, _microscope_connected)
        self.tabWidget.setTabVisible(3, _microscope_connected)
        self.tabWidget.setTabVisible(4, _microscope_connected)
        self.tabWidget.setTabVisible(5, _microscope_connected)

        if _microscope_connected:
            self.pushButton.setStyleSheet("background-color: green")
            self.pushButton.setText("Microscope Connected")

        else:
            self.pushButton.setStyleSheet("background-color: gray")
            self.pushButton.setText("Connect to Microscope")

    def connect_to_microscope(self):

        
        # TODO: add toggle for connect / disconnect
        
        _microscope_connected = bool(self.microscope is not None)

        if _microscope_connected:
            self.microscope.disconnect()
            self.microscope, self.settings = None, None
            if self.image_widget is not None:
                self.image_widget.clear_viewer()
            
        else:
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
            # self.image_widget.setMinimumWidth(500)
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
            self.manipulator_widget = FibsemManipulatorWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                image_widget=self.image_widget,
            )
            self.GIS_widget = FibsemGISWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                image_widget=self.image_widget,
            )

            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.tabWidget.addTab(self.milling_widget, "Milling")
            self.tabWidget.addTab(self.manipulator_widget, "Manipulator")
            self.tabWidget.addTab(self.GIS_widget, "GIS")


        else:
            if self.image_widget is None:
                return
            
            # remove tabs
            self.tabWidget.removeTab(5)
            self.tabWidget.removeTab(4)
            self.tabWidget.removeTab(3)
            self.tabWidget.removeTab(2)
            self.tabWidget.removeTab(1)
            

            self.image_widget.deleteLater()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()
            self.manipulator_widget.deleteLater()
            self.GIS_widget.deleteLater()


def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, area="right", add_vertical_stretch=False)
    napari.run()


if __name__ == "__main__":
    main()



