import logging
import traceback
import yaml
import os 
import napari
import napari.utils.notifications
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

import fibsem
from fibsem import config as cfg
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings, SystemSettings
from fibsem.ui.qtdesigner_files import FibsemSystemSetupWidget
from fibsem.ui.utils import _get_file_ui, _get_save_file_ui
from fibsem.ui import _stylesheets

def log_status_message(step: str):
    logging.debug(
        f"STATUS | System Widget | {step}"
    )


class FibsemSystemSetupWidget(FibsemSystemSetupWidget.Ui_Form, QtWidgets.QWidget):
    connected_signal = pyqtSignal()
    disconnected_signal = pyqtSignal()

    def __init__(
        self,
        microscope: FibsemMicroscope = None,
        settings: MicroscopeSettings = None,
        viewer: napari.Viewer = None,
        parent=None,
        config_path: str = None,
    ):
        super(FibsemSystemSetupWidget, self).__init__(parent=parent)
        self.setupUi(self)

        self.parent = parent
        self.microscope = microscope
        self.settings = settings
        self.viewer = viewer

        self.setup_connections()
        
        self.update_ui()

    def setup_connections(self):

        # connection
        self.pushButton_connect_to_microscope.clicked.connect(self.connect_to_microscope)
        
        # configuration   
        self.comboBox_configuration.addItems(cfg.USER_CONFIGURATIONS.keys())
        self.comboBox_configuration.setCurrentText(cfg.DEFAULT_CONFIGURATION_NAME) 
        self.comboBox_configuration.currentTextChanged.connect(lambda: self.load_configuration(None))
        self.toolButton_import_configuration.clicked.connect(self.import_configuration_from_file)
    
        self.pushButton_apply_configuration.clicked.connect(lambda: self.apply_microscope_configuration(None))
        self.pushButton_apply_configuration.setToolTip(f"Apply configuration can take some time. Please make sure the microscope beams are both on.")

    def load_configuration(self, configuration_name: str):
        if configuration_name is None:
            configuration_name = self.comboBox_configuration.currentText()
        
        configuration_path = cfg.USER_CONFIGURATIONS[configuration_name]["path"]

        if configuration_path is None:  
            napari.utils.notifications.show_error(f"Configuration {configuration_name} not found.")
            return
    
        # load the configuration
        self.settings = utils.load_microscope_configuration(configuration_path)

        from pprint import pprint 

        pprint(self.settings.to_dict()["info"])

        return configuration_path
    
    def import_configuration_from_file(self):
    
        path = _get_file_ui(msg="Select microscope configuration file", 
            path=cfg.CONFIG_PATH, _filter="YAML (*.yaml *.yml)")

        if path == "":
            napari.utils.notifications.show_error(f"No file selected. Configuration not loaded.")
            return
        
        # TODO: validate configuration  

        # ask user to add to user configurations
        configuration_name = os.path.basename(path).removesuffix(".yaml")

        if configuration_name not in cfg.USER_CONFIGURATIONS: 
            from fibsem.ui.utils import message_box_ui
            msg = f"Would you like to add this configuration to the user configurations?"
            ret = message_box_ui(text=msg, title="Add to user configurations?")

            # add to user configurations
            if ret:
                cfg.add_configuration(configuration_name=configuration_name, path=path)
                
                # set default configuration
                msg = f"Would you like to make this the default configuration?"
                ret = message_box_ui(text=msg, title="Set default configuration?")
                
                if ret:
                    cfg.set_default_configuration(configuration_name=configuration_name)

        # add configuration to combobox
        self.comboBox_configuration.addItem(configuration_name)
        self.comboBox_configuration.setCurrentText(configuration_name)

    def connect_to_microscope(self):
                
        _microscope_connected = bool(self.microscope)

        if _microscope_connected:
            self.microscope.disconnect()
            self.microscope, self.settings = None, None
        else:

            napari.utils.notifications.show_info(f"Connecting to microscope...")

            configuration_path = self.load_configuration(None)

            if configuration_path is None:
                napari.utils.notifications.show_error(f"Configuration not selected.")
                return

            # connect
            self.microscope, self.settings = utils.setup_session(
                config_path=configuration_path,
            )

            # user notification
            msg = f"Connected to microscope at {self.microscope.system.info.ip_address}"
            logging.info(msg)
            napari.utils.notifications.show_info(msg)

        self.update_ui()
            

    def apply_microscope_configuration(self, system_settings: SystemSettings = None):
        """Apply the microscope configuration to the microscope."""

        if self.microscope is None:
            napari.utils.notifications.show_error(f"Microscope not connected.")
            return
        
        # apply the configuration
        self.microscope.apply_configuration(system_settings=system_settings)
    



    # def apply_defaults_settings(self):
    #     microscope_settings = self.get_default_settings_from_ui()

    #     self.microscope.set_beam_system_settings(microscope_settings.system.ion)
    #     self.microscope.set_beam_system_settings(microscope_settings.system.electron)
        
    #     # TODO: complete this system setting
    #     if self.parent:
    #         if self.parent.image_widget:
    #             self.parent.image_widget.set_ui_from_settings(microscope_settings.image, beam_type=microscope_settings.image.beam_type)
    #         if self.parent.milling_widget:
    #             self.parent.milling_widget.set_milling_settings_ui(microscope_settings.milling)

    #     self.get_stage_settings_from_ui()
    #     self.get_model_from_ui()

    def update_ui(self):

        _microscope_connected = bool(self.microscope)
        self.pushButton_apply_configuration.setVisible(_microscope_connected)
        self.pushButton_apply_configuration.setEnabled(_microscope_connected and cfg._APPLY_CONFIGURATION_ENABLED)

        if _microscope_connected:
            self.pushButton_connect_to_microscope.setText("Microscope Connected")
            self.pushButton_connect_to_microscope.setStyleSheet(_stylesheets._GREEN_PUSHBUTTON_STYLE)
            self.pushButton_apply_configuration.setStyleSheet(_stylesheets._BLUE_PUSHBUTTON_STYLE)
            self.connected_signal.emit()
            
            info = self.microscope.system.info
            self.label_connection_information.setText(f"Connected to {info.manufacturer}-{info.model} at {info.ip_address}")

        else:
            self.pushButton_connect_to_microscope.setText("Connect To Microscope")
            self.pushButton_connect_to_microscope.setStyleSheet(_stylesheets._GRAY_PUSHBUTTON_STYLE)
            self.pushButton_apply_configuration.setStyleSheet(_stylesheets._GRAY_PUSHBUTTON_STYLE)
            self.disconnected_signal.emit()
            self.label_connection_information.setText("Not connected to microscope")


def main():

    viewer = napari.Viewer(ndisplay=2)
    movement_widget = FibsemSystemSetupWidget()
    viewer.window.add_dock_widget(
        movement_widget, area="right", add_vertical_stretch=False
    )
    napari.run()


if __name__ == "__main__":
    main()
