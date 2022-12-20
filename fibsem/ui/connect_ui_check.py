import sys
import napari

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage,QPixmap
from fibsem.ui.qtdesigner_files import connect
from fibsem.ui import utils as ui_utils
from fibsem import utils, acquire
from fibsem.structures import BeamType, ImageSettings, GammaSettings
import napari

class MainWindow(QtWidgets.QMainWindow, connect.Ui_MainWindow):
    def __init__(self,*args,obj=None,**kwargs) -> None:
        super(MainWindow,self).__init__(*args,**kwargs)
        self.setupUi(self)
        self.ConnectButton.clicked.connect(self.connect_to_microscope)
        self.DisconnectButton.clicked.connect(self.disconnect_from_microscope)
        self.RefImage.clicked.connect(self.take_reference_images)
        self.ResetImage.clicked.connect(self.reset_images)
        self.microscope = None
        self.settings = None
        
    
    def connect_to_microscope(self):
        self.microscope, self.settings = utils.setup_session()

    def disconnect_from_microscope(self):

        if self.microscope is None:
            print("No Microscope Connected")
            return

        self.microscope.disconnect()
        self.microscope = None
        print('Microscope Disconnected')

    def take_reference_images(self):
        
        if self.microscope is None:
            print('No Microscope Connected')
            return

        # gamma settings

        gamma_settings = GammaSettings(
            enabled=True,
            min_gamma=0.5,
            max_gamma=1.8,
            scale_factor=0.01,
            threshold=46,
        )

        # set imaging settings
        image_settings = ImageSettings(
                resolution="1536x1024",
                dwell_time=1.0e-6,
                hfw=600.0e-6,
                autocontrast=False,
                beam_type=BeamType.ION,
                gamma=gamma_settings,
                save=True,
                save_path="fibsem\\test_images",
                label=utils.current_timestamp(),
                reduced_area=None,
            )

        # take image with both beams
        eb_image, ib_image = acquire.take_reference_images(self.microscope, image_settings)

        self.EB_Image = ui_utils.set_arr_as_qlabel(eb_image.data, self.EB_Image, shape=(400, 400))
        self.IB_Image = ui_utils.set_arr_as_qlabel_8(ib_image.data, self.IB_Image, shape=(400, 400))

        print(f'EB Data type: {eb_image.data.dtype} IB Data Type: {ib_image.data.dtype}')
        # self.IB_Image = ui_utils.set_arr_as_qlabel(ib_image.data, self.IB_Image, shape=(300, 300))

        # eb_q_image = QImage(eb_image.data,eb_image.data.shape[0],eb_image.data.shape[1],eb_image.data.shape[0]*3,QImage.Format.Format_Grayscale16)
        # eb_pix = QPixmap(eb_q_image)

        # viewer.layers.clear()
        # viewer.add_image(eb_image.data, name="EB Image")
        # viewer.add_image(ib_image.data, name="IB Image")

    def reset_images(self):

        self.EB_Image.setText("No Image to Display")
        self.IB_Image.setText("No Image to Display")
        
    


if __name__ == "__main__":    

    app = QtWidgets.QApplication(sys.argv)
  
    # viewer = napari.Viewer()

    window = MainWindow()
    window.show()
    
    # viewer.window.add_dock_widget(window, name="imaging")
    # napari.run()
    app.exec()