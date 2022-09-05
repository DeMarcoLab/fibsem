
import sys

import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import acquire
from fibsem.acquire import BeamType
from fibsem.structures import BeamType
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.qtdesigner_files import user_dialog as user_gui
from PyQt5 import QtCore, QtWidgets


# TODO: remove microscope from this?
class GUIUserWindow(user_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        microscope: SdbMicroscopeClient,
        msg: str = "Default Message",
        beam_type: BeamType = BeamType.ELECTRON,
        parent=None,
    ):
        super(GUIUserWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowTitle("FIBSEM Ask User")
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        self.microscope = microscope

        # show text
        self.label_message.setText(msg)
        self.label_image.setText("")

        # show image
        if beam_type is not None:
            self.adorned_image = acquire.last_image(
                self.microscope, beam_type=beam_type
            )
            image = ndi.median_filter(self.adorned_image.data, size=3)

            # image widget
            self.wp = fibsem_ui._WidgetPlot(self, display_image=image)
            self.label_image.setLayout(QtWidgets.QVBoxLayout())
            self.label_image.layout().addWidget(self.wp)

            # draw crosshair
            fibsem_ui.draw_crosshair(image, self.wp.canvas)

        self.show()

        # Change buttons to Yes / No


def main():

    from liftout import utils
    microscope, settings = utils.quick_setup()

    app = QtWidgets.QApplication([])

    from fibsem.ui import windows as fibsem_ui_windows

    ret = fibsem_ui_windows.ask_user_interaction(microscope, settings, BeamType.ELECTRON)
    print(f"ret: {ret}")

    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
