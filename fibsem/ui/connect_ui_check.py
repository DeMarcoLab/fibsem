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

        # image and gamma settings buttons/boxes/ui objects

        self.reset_image_settings.clicked.connect(self.reset_image_and_gammaSettings)
        self.autocontrast_enable.stateChanged.connect(self.autocontrast_check)
        self.gamma_enabled.stateChanged.connect(self.gamma_check)
        self.gamma_min.valueChanged.connect(self.gamma_min_change)
        self.gamma_max.valueChanged.connect(self.gamma_max_change)
        self.gamma_scalefactor.valueChanged.connect(self.gamma_scalefactor_change)
        self.gamma_threshold.valueChanged.connect(self.gamma_threshold_change)
        self.res_width.valueChanged.connect(self.res_width_change)
        self.res_height.valueChanged.connect(self.res_height_change)
        self.dwell_time_setting.valueChanged.connect(self.image_dwell_time_change)
        self.hfw_box.valueChanged.connect(self.hfw_box_change)


        # Initialise microscope object
        self.microscope = None
        self.CLog.setText("Welcome to OpenFIBSEM! Begin by Connecting to a Microscope")

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
    
    def hfw_box_change(self):
        ### field width in microns in UI!!!!!!!!
        self.image_settings.hfw = self.hfw_box.value() / 1.0e6

    def gamma_threshold_change(self):

        self.gamma_settings.threshold = self.gamma_threshold.value()

    def res_width_change(self):

        res = self.image_settings.resolution.split("x")

        res[0] = str(self.res_width.value())

        self.image_settings.resolution = "x".join(res)

    def res_height_change(self):

        resh = self.image_settings.resolution.split("x")

        resh[1] = str(self.res_height.value())

        self.image_settings.resolution = "x".join(resh)

    def image_dwell_time_change(self):
        ### dwell time in ms!!!!! ease of use for UI!!!!
        self.image_settings.dwell_time = self.dwell_time_setting.value()/1.0e-6

    def gamma_min_change(self):

        self.gamma_settings.min_gamma = self.gamma_min.value()  

    def gamma_max_change(self):

        self.gamma_settings.max_gamma = self.gamma_max.value()  

    def gamma_scalefactor_change(self):

        self.gamma_settings.scale_factor = self.gamma_scalefactor.value()  

    def autocontrast_check(self):

        # check box returns 2 if checked, 0 if unchecked

        if self.autocontrast_enable.checkState() == 2:
            self.image_settings.autocontrast = True
            self.update_log("Autocontrast Enabled")
        elif self.autocontrast_enable.checkState() == 0:
            self.image_settings.autocontrast = False
            self.update_log("Autocontrast Disabled")
    
    def gamma_check(self):

        if self.gamma_enabled.checkState() == 2:
            self.gamma_settings.enabled = True
            self.update_log("Gamma Enabled")
        elif self.gamma_enabled.checkState() == 0:
            self.gamma_settings.enabled = False
            self.update_log("Gamma Disabled")

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

    def reset_image_and_gammaSettings(self):

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
        
        self.gamma_min.setValue(self.gamma_settings.min_gamma)
        self.gamma_max.setValue(self.gamma_settings.max_gamma)
        self.gamma_scalefactor.setValue(self.gamma_settings.scale_factor)
        self.dwell_time_setting.setValue(self.image_settings.dwell_time * 1.0e6)
        self.hfw_box.setValue(int(self.image_settings.hfw * 1e6))

        res_ful = self.image_settings.resolution.split("x")

        self.res_width.setValue(int(res_ful[0]))
        self.res_height.setValue(int(res_ful[1]))


        if self.gamma_settings.enabled:
            self.gamma_enabled.setCheckState(2)
        else:
            self.gamma_enabled.setCheckState(0)

        if self.image_settings.autocontrast:

            self.autocontrast_enable.setCheckState(2)
        else:
            self.autocontrast_enable.setCheckState(0)
        
        self.update_log("Gamma and image settings returned to default values")


        
    


if __name__ == "__main__":    

    app = QtWidgets.QApplication(sys.argv)
  
    # viewer = napari.Viewer()

    window = MainWindow()
    window.show()
    
    # viewer.window.add_dock_widget(window, name="imaging")
    # napari.run()
    app.exec()