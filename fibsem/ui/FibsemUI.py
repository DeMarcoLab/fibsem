import logging
import os
import napari
import fibsem
import napari.utils.notifications
from fibsem import utils
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget

from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemGISWidget import FibsemGISWidget

from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget

from napari.qt.threading import thread_worker
from PyQt5 import QtWidgets
from fibsem import config as cfg


from fibsem.microscope import FibsemMicroscope, MicroscopeSettings
from fibsem.ui.qtdesigner_files import FibsemUI


class FibsemUI(FibsemUI.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self, viewer: napari.Viewer):
        super(FibsemUI, self).__init__()
        self.setupUi(self)

        self.viewer = viewer
        self.viewer.window._qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window._qt_viewer.dockLayerControls.setVisible(False)

        self.microscope: FibsemMicroscope = None
        self.settings: MicroscopeSettings = None

        self.image_widget: FibsemImageSettingsWidget = None
        self.movement_widget: FibsemMovementWidget = None
        self.milling_widget: FibsemMillingWidget = None
        self.manipulator_widget: FibsemManipulatorWidget = None
        self.GIS_widget: FibsemGISWidget = None

        CONFIG_PATH = os.path.join(cfg.CONFIG_PATH)
        self.system_widget = FibsemSystemSetupWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                config_path = CONFIG_PATH,
            )
        
        self.setup_connections()
        self.tabWidget.addTab(self.system_widget, "System")

        self.update_ui()


    def setup_connections(self):

        self.system_widget.set_stage_signal.connect(self.set_stage_parameters)
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)

    def set_stage_parameters(self):
        if self.microscope is None:
            return
        self.settings.system.stage = self.system_widget.settings.system.stage   # TODO: this doesnt actually update the movement widget
        logging.info("Stage parameters set")  

    def update_ui(self):

        _microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(1, _microscope_connected)
        self.tabWidget.setTabVisible(2, _microscope_connected)
        self.tabWidget.setTabVisible(3, _microscope_connected)
        self.tabWidget.setTabVisible(4, _microscope_connected)
        self.tabWidget.setTabVisible(5, _microscope_connected)

    def connect_to_microscope(self):
        self.microscope = self.system_widget.microscope
        self.settings = self.system_widget.settings
        self.update_microscope_ui()
        self.update_ui()

    def disconnect_from_microscope(self):
        
        self.microscope = None
        self.settings = None
        self.update_microscope_ui()
        self.update_ui()
        self.image_widget = None
        self.movement_widget = None
        self.milling_widget = None

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
            

            self.image_widget.clear_viewer()
            self.image_widget.deleteLater()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()
            self.manipulator_widget.deleteLater()
            self.GIS_widget.deleteLater()

def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, 
                                  area="right", 
                                  add_vertical_stretch=True, 
                                  name="OpenFIBSEM")
    napari.run()


if __name__ == "__main__":
    main()



