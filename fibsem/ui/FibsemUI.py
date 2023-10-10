import napari

import logging

import fibsem
from fibsem import config as cfg
from fibsem.microscope import (DemoMicroscope, FibsemMicroscope,
                               MicroscopeSettings, TescanMicroscope,
                               ThermoMicroscope)
from fibsem.structures import BeamType
from fibsem.ui.FibsemAlignmentWidget import FibsemAlignmentWidget
from fibsem.ui.FibsemImageViewer import FibsemImageViewer
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.qtdesigner_files import FibsemUI
from fibsem.ui.utils import message_box_ui

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal


class FibsemUI(FibsemUI.Ui_MainWindow, QtWidgets.QMainWindow):
    _minimap_signal = pyqtSignal(object)

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
        self.alignment_widget: FibsemAlignmentWidget = None
        self.manipulator_widget: FibsemManipulatorWidget = None
        self.image_viewer: FibsemImageViewer = None

        self.minimap_widget: FibsemMinimapWidget = None

        self.system_widget = FibsemSystemSetupWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                config_path = cfg.SYSTEM_PATH,
                parent=self,
            )
        
        self.tabWidget.addTab(self.system_widget, "System")
        self.setup_connections()
        self.ref_current = None
        self.update_ui()


    def setup_connections(self):
        from fibsem import utils
        self.system_widget.set_stage_signal.connect(self.set_stage_parameters)
        self.system_widget.connected_signal.connect(self.connect_to_microscope)
        self.system_widget.disconnected_signal.connect(self.disconnect_from_microscope)
        self.actionCurrent_alignment.triggered.connect(self.align_currents)
        self.actionImage_Viewer.triggered.connect(self._open_image_viewer)
        self.actionManipulator_Positions_Calibration.triggered.connect(self.calibrate_manipulator_positions)
        if self.system_widget.microscope is not None:
            self.connect_to_microscope()
            settings_dict = utils.load_yaml(cfg.SYSTEM_PATH)
            if bool(settings_dict["apply_settings_on_startup"]):
                self.system_widget.apply_settings = True
                self.system_widget.apply_defaults_settings() 
        self.actionOpen_Minimap.triggered.connect(self._open_minimap)

        

    def minimap_connection(self,positions=None):

        if self.minimap_widget is None:
            return
        else:
            self._minimap_signal.emit(positions)


    def _open_image_viewer(self):
        viewer2 = napari.Viewer(ndisplay=2)
        self.image_viewer = FibsemImageViewer(viewer=viewer2)
        viewer2.window.add_dock_widget(self.image_viewer, 
                                        area="right", 
                                        add_vertical_stretch=True, 
                                        name=f"OpenFIBSEM Image Viewer")

        napari.run(max_loop_level=2)

    def _open_minimap(self):
        if self.microscope is None:
            napari.utils.notifications.show_warning(f"Please connect to a microscope first... [No Microscope Connected]")
            return

        if self.movement_widget is None:
            napari.utils.notifications.show_warning(f"Please connect to a microscope first... [No Movement Widget]")
            return

        # TODO: need to register this with the main ui somehow
        self.viewer2 = napari.Viewer(ndisplay=2)
        self.minimap_widget = FibsemMinimapWidget(self.microscope, self.settings, viewer=self.viewer2, parent=self)
        self.viewer2.window.add_dock_widget(
            self.minimap_widget, area="right", add_vertical_stretch=False, name="OpenFIBSEM Minimap"
        )
        self.minimap_widget._stage_position_moved.connect(self.movement_widget._stage_position_moved)

        self.minimap_widget._minimap_positions.connect(self.movement_widget.minimap_window_positions)

        self.minimap_widget.positions = self.movement_widget.positions
        self.minimap_widget._update_position_info()
        self.minimap_widget._update_viewer()

        napari.run(max_loop_level=2)


    def calibrate_manipulator_positions(self):

        if not isinstance(self.microscope,TescanMicroscope):
            message_box_ui(title="Not Available", text="Manipulator Position Calibration is only available for Tescan Microscopes", buttons=QMessageBox.Ok)
            return

        response = self.manipulator_widget._check_manipulator_positions_setup()

        if not response:
            
            ok_to_cal =message_box_ui(title="Manipulator Position calibration",text="This tool calibrates the positions of the manipulator, it will switch between the parking, standby and working positions rapidly, please ensure it is safe to do so. If not please click no, otherwise press yes to continue")
                                      
            if ok_to_cal:

                positions = self.settings.hardware.manipulator_positions

                for position in positions:
                    if position == "calibrated":
                        continue
                    logging.info(f"Calibrating Manipulator {position} position")
                    self.microscope.insert_manipulator(position)
                    manipulator_loc = self.microscope.get_manipulator_position()
                    positions[position]["x"] = manipulator_loc.x
                    positions[position]["y"] = manipulator_loc.y
                    positions[position]["z"] = manipulator_loc.z
                
                positions["calibrated"] = True
                self.settings.hardware.manipulator_positions = positions
                self.system_widget.settings = self.settings
                self.system_widget.save_defaults(path = cfg.SYSTEM_PATH)

                message_box_ui(title="Manipulator Position calibration",text="Manipulator Positions calibrated successfully", buttons=QMessageBox.Ok)

                
                self.manipulator_widget.insertManipulator_button.setEnabled(True)
                self.manipulator_widget.moveRelative_button.setEnabled(True)
                self.manipulator_widget.addSavedPosition_button.setEnabled(True)
                self.manipulator_widget.goToPosition_button.setEnabled(True)
                self.manipulator_widget.manipulator_inserted = self.manipulator_widget.microscope.get_manipulator_state(settings=self.manipulator_widget.settings)
                self.manipulator_widget._hide_show_buttons(show=self.manipulator_widget.manipulator_inserted)
                self.manipulator_widget.manipulatorStatus_label.setText("Manipulator Status: Inserted" if self.manipulator_widget.manipulator_inserted else "Manipulator Status: Retracted")
                self.manipulator_widget.insertManipulator_button.setText("Insert" if not self.manipulator_widget.manipulator_inserted else "Retract")
                self.manipulator_widget.calibrated_status_label.setText("Calibrated")
                self.manipulator_widget.settings = self.settings

    def align_currents(self):
        second_viewer = napari.Viewer()
        self.alignment_widget = FibsemAlignmentWidget(settings=self.settings, microscope=self.microscope, viewer=second_viewer, parent = self)
        second_viewer.window.add_dock_widget(self.alignment_widget, name='Beam Alignment', area='right')
        self.alignment_widget.destroyed.connect(self.reset_currents)

    def reset_currents(self):
        if isinstance(self.microscope, ThermoMicroscope)or isinstance(self.microscope, DemoMicroscope):
            self.microscope.set("current", float(self.ref_current), BeamType.ION)
        if isinstance(self.microscope, TescanMicroscope):
            self.microscope.set("preset", self.ref_current, BeamType.ION)

    def set_stage_parameters(self):
        if self.microscope is None:
            return
        self.settings.system.stage = self.system_widget.settings.system.stage  
        self.movement_widget.settings = self.settings
        self.microscope.stage_settings = self.settings.system.stage
        logging.debug(f"Stage parameters set to {self.settings.system.stage}")
        logging.info("Stage parameters set")  

    def update_ui(self):

        _microscope_connected = bool(self.microscope is not None)
        self.tabWidget.setTabVisible(1, _microscope_connected)
        self.tabWidget.setTabVisible(2, _microscope_connected)
        self.tabWidget.setTabVisible(3, _microscope_connected)
        self.tabWidget.setTabVisible(4, _microscope_connected)
        self.actionOpen_Minimap.setVisible(_microscope_connected)
        self.actionCurrent_alignment.setVisible(_microscope_connected)
        self.actionManipulator_Positions_Calibration.setVisible(_microscope_connected)


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
                parent=self,
            )
            self.movement_widget = FibsemMovementWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                image_widget=self.image_widget,
                parent=self,
            )
            self.milling_widget = FibsemMillingWidget(
                microscope=self.microscope,
                settings=self.settings,
                viewer=self.viewer,
                image_widget=self.image_widget,
                parent=self,
            )
            if self.microscope.hardware_settings.manipulator_enabled:
                self.manipulator_widget = FibsemManipulatorWidget(
                    microscope=self.microscope,
                    settings=self.settings,
                    viewer=self.viewer,
                    image_widget=self.image_widget,
                    parent=self,
                )
            else:
                self.manipulator_widget = None
  


            # add widgets to tabs
            self.tabWidget.addTab(self.image_widget, "Image")
            self.tabWidget.addTab(self.movement_widget, "Movement")
            self.tabWidget.addTab(self.milling_widget, "Milling")

            if self.microscope.hardware_settings.manipulator_enabled:
                self.tabWidget.addTab(self.manipulator_widget, "Manipulator")

            self.system_widget.image_widget = self.image_widget
            self.system_widget.milling_widget = self.milling_widget
        
            # connect movement widget signal
            self.movement_widget.positions_signal.connect(self.minimap_connection)
            self.movement_widget.move_signal.connect(self.minimap_connection)

        else:
            if self.image_widget is None:
                return
            
            # remove tabs
            self.tabWidget.removeTab(4)
            self.tabWidget.removeTab(3)
            self.tabWidget.removeTab(2)
            self.tabWidget.removeTab(1)
            self.image_widget.clear_viewer()
            self.image_widget.deleteLater()
            self.movement_widget.deleteLater()
            self.milling_widget.deleteLater()
            if self.manipulator_widget is not None:
                self.manipulator_widget.deleteLater() 



def main():

    viewer = napari.Viewer(ndisplay=2)
    fibsem_ui = FibsemUI(viewer=viewer)
    viewer.window.add_dock_widget(fibsem_ui, 
                                  area="right", 
                                  add_vertical_stretch=True, 
                                  name=f"OpenFIBSEM v{fibsem.__version__}")
    napari.run()


if __name__ == "__main__":
    main()



