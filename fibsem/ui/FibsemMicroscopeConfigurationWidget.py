import argparse
import napari
import napari.utils.notifications
import fibsem
from PyQt5 import QtWidgets
from fibsem import config as cfg
from fibsem.ui.qtdesigner_files import FibsemMicroscopeConfigurationWidget
from fibsem.ui.FibsemMicroscopeConfigurationWidgetBase import FibsemMicroscopeConfigurationWidgetBase


class FibsemMicroscopeConfigurationWidget(FibsemMicroscopeConfigurationWidget.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(
        self,
        path: str = None,
        viewer: napari.Viewer = None,
        parent=None,
    ):
        super(FibsemMicroscopeConfigurationWidget, self).__init__(parent=parent)
        self.setupUi(self)
        self.setWindowTitle("Microscope Configuration")

        self.viewer = viewer

        self.configuration_widget = FibsemMicroscopeConfigurationWidgetBase(path)
        # add to layout
        self.gridLayout.addWidget(self.configuration_widget)

        self.setup_connections()

    def setup_connections(self):

        # actions
        self.actionSave_Configuration.triggered.connect(self.configuration_widget.save_configuration)
        self.actionLoad_Configuration.triggered.connect(self.configuration_widget.load_configuration_from_file)


def main():

    # parse arguments
    parser = argparse.ArgumentParser(f"Microscope Configuration UI")
    parser.add_argument("--config", type=str, default=cfg.DEFAULT_CONFIGURATION_PATH, help="Path to microscope configuration file")
    args = parser.parse_args()

    # widget viewer
    viewer = napari.Viewer(ndisplay=2)
    microscope_configuration = FibsemMicroscopeConfigurationWidget(path=args.config, viewer=viewer)
    viewer.window.add_dock_widget(
        microscope_configuration, 
        area="right", 
        add_vertical_stretch=False,
        name=f"OpenFIBSEM v{fibsem.__version__} Microscope Configuration",
    )
    napari.run()


if __name__ == "__main__":
    main()