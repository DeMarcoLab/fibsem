import logging
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem import constants, conversions, utils
from fibsem.ui.qtdesigner_files import FibsemMicroscopeConfigurationWidget
from fibsem.ui import _stylesheets


from pprint import pprint


class FibsemMicroscopeConfigurationWidget(FibsemMicroscopeConfigurationWidget.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        path: str,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemMicroscopeConfigurationWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle("Microscope Configuration")

        self.viewer = viewer

        self.setup_connections()

        self.configuration = self.load_configuration(path)
        self.load_configuration_to_ui()

    def setup_connections(self):

        print("setup connections")

        # buttons
        self.pushButton_save_configuration.clicked.connect(self.save_configuration)
        self.pushButton_exit_configuration.clicked.connect(self.exit_configuration)
        self.pushButton_save_configuration.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
        self.pushButton_exit_configuration.setStyleSheet(_stylesheets._RED_PUSHBUTTON_STYLE)

    def save_configuration(self):

        print("save configuration")

    def exit_configuration(self):
            
        print("exit configuration")
        self.close()
        
        # close the window
        # detroy the docked widget
        # remove the docked widget from the viewer
        # remove the docked widget from the napari window

        self.viewer.window.remove_dock_widget(self)

    def load_configuration(self, path: str):
        
        self.configuration = utils.load_yaml(path)

        return self.configuration
    
    def load_configuration_to_ui(self):

        
        # core


        # stage


        # electron


        # ion


        # imaging

        # milling

        # subsystems
        subsystem_configuration = self.configuration["subsystems"]
        # electron
        self.checkBox_subsystems_electron_enabled.setChecked(subsystem_configuration["electron"]["enabled"])
        # ion
        self.checkBox_subsystems_ion_enabled.setChecked(subsystem_configuration["ion"]["enabled"])
        self.checkBox_subsystems_ion_plasma.setChecked(subsystem_configuration["ion"]["plasma"])
        # stage
        self.checkBox_subsystems_stage_enabled.setChecked(subsystem_configuration["stage"]["enabled"])
        self.checkBox_subsystems_stage_rotation.setChecked(subsystem_configuration["stage"]["rotation"])
        self.checkBox_subsystems_stage_tilt.setChecked(subsystem_configuration["stage"]["tilt"])
        # manipulator
        self.checkBox_subsystems_manipulator_enabled.setChecked(subsystem_configuration["manipulator"]["enabled"])
        self.checkBox_subsystems_manipulator_rotation.setChecked(subsystem_configuration["manipulator"]["rotation"])
        self.checkBox_subsystems_manipulator_tilt.setChecked(subsystem_configuration["manipulator"]["tilt"])
        # gis
        self.checkBox_subsystems_gis_enabled.setChecked(subsystem_configuration["gis"]["enabled"])
        self.checkBox_subsystems_gis_multichem_enabled.setChecked(subsystem_configuration["gis"]["multichem"])
        self.checkBox_subsystems_gis_sputter_coater_enabled.setChecked(subsystem_configuration["gis"]["sputter_coater"])


def main():

    viewer = napari.Viewer(ndisplay=2)
    microscope, settings = utils.setup_session()
    microscope_configuration = FibsemMicroscopeConfigurationWidget(path=cfg.DEFAULT_CONFIGURATION_PATH, viewer=viewer)
    viewer.window.add_dock_widget(
        microscope_configuration, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()