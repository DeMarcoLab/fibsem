import sys
import napari
from datetime import datetime

#from PyQt5.QtGui import QImage,QPixmap
from fibsem.ui.qtdesigner_files import connect
from PyQt5 import QtWidgets

from fibsem import utils, acquire
from fibsem.structures import BeamType, ImageSettings, GammaSettings, FibsemImage
from pprint import pprint
import os
import tkinter
from tkinter import filedialog


from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGridLayout, QLabel
import numpy as np

import logging


def set_arr_as_qlabel(
    arr: np.ndarray,
    label: QLabel,
    shape: tuple = (1536//4, 1024//4),
) -> QLabel:

    if arr.dtype == 'uint8':

        image = QImage(
            arr.data,
            arr.shape[1],
            arr.shape[0],
            QImage.Format_Grayscale8,
        )
        label.setPixmap(QPixmap.fromImage(image).scaled(*shape))

        return label
    else:

        image = QImage(
            arr.data,
            arr.shape[1],
            arr.shape[0],
            QImage.Format_Grayscale16,
        )
        label.setPixmap(QPixmap.fromImage(image).scaled(*shape))

        return label


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
        self.EB_Save.clicked.connect(self.save_EB_Image)
        self.IB_Save.clicked.connect(self.save_IB_Image)
        self.open_filepath.clicked.connect(self.save_filepath)

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
        self.autosave_enable.stateChanged.connect(self.autosave_toggle)

        # Gamma and Image Settings

        self.FIB_IB = FibsemImage
        self.FIB_EB = FibsemImage

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

        # Initialise microscope object
        self.microscope = None
        self.CLog.setText("Welcome to OpenFIBSEM! Begin by Connecting to a Microscope")
        self.reset_ui_settings()

    
    def autosave_toggle(self):

        if self.autosave_enable.checkState() == 2:
            self.image_settings.save = True
            self.update_log("Autosave Enabled")
            logging.info(f"UI | Autosave Enabled")
        elif self.autosave_enable.checkState() == 0:
            self.image_settings.save = False
            self.update_log("Autosave Disabled")
            logging.info(f"UI | Autosave Disabled")

        

    def save_filepath(self):
        
        tkinter.Tk().withdraw()
        folder_path = filedialog.askdirectory()
        self.savepath_text.setText(folder_path)
        self.image_settings.save_path = folder_path
        

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
        self.image_settings.dwell_time = self.dwell_time_setting.value()/1.0e6

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
            logging.info(f"UI | AutoContrast Enabled")
        elif self.autocontrast_enable.checkState() == 0:
            self.image_settings.autocontrast = False
            self.update_log("Autocontrast Disabled")
            logging.info(f"UI | AutoContrast Disabled")

    
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

        self.CLog.setText(self.CLog2.text())
        self.CLog2.setText(self.CLog3.text())
        self.CLog3.setText(self.CLog4.text())

        self.CLog4.setText(timestr+ " : " +log)

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

        self.FIB_IB = ib_image
        self.FIB_EB = eb_image

        self.EB_Image = set_arr_as_qlabel(eb_image.data, self.EB_Image, shape=(400, 400))
        self.IB_Image = set_arr_as_qlabel(ib_image.data, self.IB_Image, shape=(400, 400))

        self.reset_ui_settings()

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
        self.FIB_EB = eb_image
        print(f"statement is {eb_image.data.dtype == 'uint16'}")
        self.EB_Image = set_arr_as_qlabel(eb_image.data, self.EB_Image, shape=(400, 400))
        self.image_settings.beam_type = tmp_beam_type
        self.reset_ui_settings()
        self.update_log("EB Image Taken!")
    
    def click_IB_Image(self):

        if self.microscope is None:
            self.update_log("No Microscope Connected")
            return

        tmp_beam_type = self.image_settings.beam_type
        self.image_settings.beam_type = BeamType.ION
        ib_image = acquire.new_image(self.microscope, self.image_settings)
        print(ib_image.data.dtype)
        self.FIB_IB = ib_image
        self.IB_Image = set_arr_as_qlabel(ib_image.data, self.IB_Image, shape=(400, 400))
        self.image_settings.beam_type = tmp_beam_type
        self.reset_ui_settings()

        self.update_log("IB Image Taken!")

    def save_EB_Image(self):
        save_path = os.path.join(self.image_settings.save_path, self.image_settings.label + "_eb")
        self.FIB_EB.save(save_path=save_path)

        self.update_log(f"EB Image Saved to {save_path}.tif!")

    def save_IB_Image(self):
        save_path = os.path.join(self.image_settings.save_path, self.image_settings.label + "_ib")
        self.FIB_IB.save(save_path)

        self.update_log(f"IB Image Saved to {save_path}.tif!")

    def reset_images(self):

        self.EB_Image.setText(" ")
        self.IB_Image.setText(" ")
        print("hello")

    def reset_ui_settings(self):

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

        if self.image_settings.save:
            self.autosave_enable.setCheckState(2)
        else:
            self.autocontrast_enable.setCheckState(0)
        
        self.savepath_text.setText(self.image_settings.save_path)

    

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
        
        self.reset_ui_settings()
        
        self.update_log("Gamma and image settings returned to default values")


        
    


if __name__ == "__main__":    

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    
    app.exec()