import sys
import napari
from datetime import datetime
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

        # Buttons setup

        self.ConnectButton.clicked.connect(self.connect_to_microscope)
        self.DisconnectButton.clicked.connect(self.disconnect_from_microscope)
        self.RefImage.clicked.connect(self.take_reference_images)
        self.ResetImage.clicked.connect(self.reset_images)
        self.EB_Click.clicked.connect(self.click_EB_Image)
        self.IB_click.clicked.connect(self.click_IB_Image)

        # Initialise microscope object
        self.microscope = None

        # Gamma and Image Settings

        self.gamma_settings = GammaSettings(
            enabled=True,
            min_gamma=0.5,
            max_gamma=1.8,
            scale_factor=0.01,
            threshold=46,
        )

        self.image_settings =  ImageSettings(
                resolution="1536x1024",
                dwell_time=1.0e-6,
                hfw=600.0e-6,
                autocontrast=False,
                beam_type=BeamType.ION,
                gamma=self.gamma_settings,
                save=True,
                save_path="fibsem\\test_images",
                label=utils.current_timestamp(),
                reduced_area=None,
            )
    
    def update_log(self,log:str):

        now = datetime.now()
        timestr = now.strftime("%d/%m  %H:%M:%S")

        self.CLog4.setText(self.CLog3.text())
        self.CLog3.setText(self.CLog2.text())
        self.CLog2.setText(self.CLog.text())

        self.CLog.setText(timestr+ " : " +log)

    def connect_to_microscope(self):

        self.update_log("Attempting to connect...")
        
        try:
            self.microscope, self.settings = utils.setup_session()
            self.update_log('Connected to microscope successfully')
        except:
            self.update_log('Unable to connect to microscope')

    def disconnect_from_microscope(self):

        if self.microscope is None:
            self.update_log("No Microscope Connected")
            return

        self.microscope.disconnect()
        self.microscope = None
        self.update_log('Microscope Disconnected')

    def take_reference_images(self):
        
        if self.microscope is None:
            self.update_log("No Microscope Connected")
            return

        # take image with both beams
        eb_image, ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.EB_Image = ui_utils.set_arr_as_qlabel(eb_image.data, self.EB_Image, shape=(400, 400))
        self.IB_Image = ui_utils.set_arr_as_qlabel_8(ib_image.data, self.IB_Image, shape=(400, 400))

        # viewer.layers.clear()
        # viewer.add_image(eb_image.data, name="EB Image")
        # viewer.add_image(ib_image.data, name="IB Image")

    def click_EB_Image(self):

        if self.microscope is None:
            self.update_log("No Microscope Connected")
            return

        tmp_beam_type = self.image_settings.beam_type
        self.image_settings.beam_type = BeamType.ELECTRON
        eb_image = acquire.new_image(self.microscope, self.image_settings)
        self.EB_Image = ui_utils.set_arr_as_qlabel(eb_image.data, self.EB_Image, shape=(400, 400))
        self.image_settings.beam_type = tmp_beam_type
    
    def click_IB_Image(self):

        if self.microscope is None:
            self.update_log("No Microscope Connected")
            return

        tmp_beam_type = self.image_settings.beam_type
        self.image_settings.beam_type = BeamType.ION
        ib_image = acquire.new_image(self.microscope, self.image_settings)
        self.IB_Image = ui_utils.set_arr_as_qlabel_8(ib_image.data, self.IB_Image, shape=(400, 400))
        self.image_settings.beam_type = tmp_beam_type

    def reset_images(self):

        self.EB_Image.setText("")
        self.IB_Image.setText("")
        
    


if __name__ == "__main__":    

    app = QtWidgets.QApplication(sys.argv)
  
    # viewer = napari.Viewer()

    window = MainWindow()
    window.show()
    
    # viewer.window.add_dock_widget(window, name="imaging")
    # napari.run()
    app.exec()