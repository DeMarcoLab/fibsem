
import sys

import scipy.ndimage as ndi
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from fibsem import acquire, utils
from fibsem.acquire import BeamType
from fibsem.structures import BeamType
from fibsem.ui import utils as fibsem_ui
from fibsem.ui.qtdesigner_files import user_dialog as user_gui
from PyQt5 import QtCore, QtWidgets

import numpy as np

# TODO: remove microscope from this?
# TODO: only pass iamge to display
class GUIUserWindow(user_gui.Ui_Dialog, QtWidgets.QDialog):
    def __init__(
        self,
        msg: str = "Default Message",
        image: np.ndarray = None,
        parent=None,
    ):
        super(GUIUserWindow, self).__init__(parent=parent)
        self.setupUi(self)

        self.setWindowTitle("FIBSEM Ask User")
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        # show text
        self.label_message.setText(msg)
        self.label_image.setText("")

        # show image
        if image is not None:
            image = ndi.median_filter(image, size=3)

            # show image
            self.label_image = fibsem_ui.set_arr_as_qlabel(image, self.label_image)

        self.show()

        # Change buttons to Yes / No


def main():


    app = QtWidgets.QApplication([])

    from fibsem.ui import windows as fibsem_ui_windows
    from autoscript_sdb_microscope_client.structures import AdornedImage


    arr = np.random.random(size=(1536, 1024))

    img = AdornedImage.load("/home/patrick/github/fibsem/scratch/figure/baseline_eb.tif").data
    img = None

    ret = fibsem_ui_windows.ask_user_interaction(msg="hi user", image=img)
    print(f"ret: {ret}")

    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
