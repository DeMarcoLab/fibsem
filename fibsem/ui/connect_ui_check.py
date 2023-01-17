import sys
import pathlib
from datetime import datetime
import re

from fibsem.ui.qtdesigner_files import connect

from fibsem import utils, acquire
from fibsem.structures import BeamType, ImageSettings, GammaSettings, FibsemImage, FibsemStagePosition
from pprint import pprint
import os
import tkinter
from tkinter import filedialog
import fibsem.constants as constants

from qtpy import QtWidgets
from PyQt5.QtCore import QTimer,QDateTime
import numpy as np

import logging
import napari
from napari.settings import get_settings



class MainWindow(QtWidgets.QMainWindow, connect.Ui_MainWindow):
    def __init__(self,*args,obj=None,**kwargs) -> None:
        super(MainWindow,self).__init__(*args,**kwargs)
        self.setupUi(self)

        # 
        self.setup_connections()
        self.lines = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_log)
        self.timer.start(1000)
        
        # Gamma and Image Settings

        self.FIB_IB = FibsemImage(data=np.zeros((1536,1024), dtype=np.uint8))
        self.FIB_EB = FibsemImage(data=np.zeros((1536,1024), dtype=np.uint8))

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
                save=False,
                save_path="fibsem\\test_images",
                label=utils.current_timestamp(),
                reduced_area=None,
            )

        self.CLog8.setText("Welcome to OpenFIBSEM! Begin by Connecting to a Microscope")

        # Initialise microscope object
        self.microscope = None
        self.microscope_settings = None
        self.connect_to_microscope()
        
        self.reset_ui_settings()

        

        if self.microscope is not None:
           self.update_position_ui()


        ### NAPARI settings and initialisation

        
        viewer.grid.enabled = True

        self.update_displays()

    def setup_connections(self):

        # Buttons setup

        self.ConnectButton.clicked.connect(self.connect_to_microscope)
        self.DisconnectButton.clicked.connect(self.disconnect_from_microscope)
        self.RefImage.clicked.connect(self.take_reference_images)
        self.ResetImage.clicked.connect(self.reset_images)
        self.take_image.clicked.connect(self.take_image_beams)
        self.save_button.clicked.connect(self.save_image_beams)
        self.open_filepath.clicked.connect(self.save_filepath)
        

        # image and gamma settings buttons/boxes/ui objects

        self.reset_image_settings.clicked.connect(self.reset_image_and_gammaSettings)
        self.autocontrast_enable.stateChanged.connect(self.autocontrast_check)
        self.gamma_enabled.stateChanged.connect(self.gamma_check)
        self.res_width.valueChanged.connect(self.res_width_change)
        self.res_height.valueChanged.connect(self.res_height_change)
        self.dwell_time_setting.valueChanged.connect(self.image_dwell_time_change)
        self.hfw_box.valueChanged.connect(self.hfw_box_change)
        self.autosave_enable.stateChanged.connect(self.autosave_toggle)

        # Movement controls setup

        self.move_rel_button.clicked.connect(self.move_microscope_rel)
        self.move_abs_button.clicked.connect(self.move_microscope_abs)


    ### Movement Functionality 

    def save_image_beams(self):
        if self.image_label.toPlainText() != "":
            self.image_settings.label = self.image_label.toPlainText()

        if self.check_EB.isChecked():
            self.save_EB_Image()
        if self.check_IB.isChecked():
            self.save_IB_Image()

    def take_image_beams(self):
        self.image_settings.label = utils.current_timestamp()
        if self.check_EB.isChecked():
            self.click_EB_Image()
        if self.check_IB.isChecked():
            self.click_IB_Image()

    def read_abs_positions_meters(self):
        """Reads the current position of the microscope stage
        """
        if self.microscope_settings.system.manufacturer == "Thermo":
            position = FibsemStagePosition(
            x = self.xAbs.value()*constants.MILLIMETRE_TO_METRE,
            y = self.yAbs.value()*constants.MILLIMETRE_TO_METRE,
            z = self.zAbs.value()*constants.MILLIMETRE_TO_METRE,
            
            t = self.tAbs.value()*constants.DEGREES_TO_RADIANS,
            r = self.rAbs.value()*constants.DEGREES_TO_RADIANS, 
            coordinate_system="raw" )
        
        else:
            position = FibsemStagePosition(
                x = self.xAbs.value()*constants.MILLIMETRE_TO_METRE,
                y = self.yAbs.value()*constants.MILLIMETRE_TO_METRE,
                z = self.zAbs.value()*constants.MILLIMETRE_TO_METRE,
                
                t = self.tAbs.value()*constants.DEGREES_TO_RADIANS,
                r = self.rAbs.value()*constants.DEGREES_TO_RADIANS, 
                coordinate_system="raw" )
        return position


    def read_relative_move_meters(self):
        """Reads the current position of the microscope stage
        """
        position = FibsemStagePosition(
        x = self.dXchange.value()*constants.MILLIMETRE_TO_METRE,
        y = self.dYchange.value()*constants.MILLIMETRE_TO_METRE,
        z = self.dZchange.value()*constants.MILLIMETRE_TO_METRE,
        t = self.dTchange.value()*constants.DEGREES_TO_RADIANS,
        r = self.dRchange.value()*constants.DEGREES_TO_RADIANS, 
        coordinate_system="raw" )
        
        return position

    def move_microscope_abs(self):
        """Moves microscope stage in absolute coordinates
        """
              
        new_position = self.read_abs_positions_meters()

        self.microscope.move_stage_absolute(new_position)

        logging.info("Moving Stage in Absolute Coordinates")
        logging.info(f"Moved to x:{(new_position.x*constants.METRE_TO_MILLIMETRE):.3f} mm y:{(new_position.y*constants.METRE_TO_MILLIMETRE):.3f} mm z:{(new_position.z*constants.METRE_TO_MILLIMETRE):.3f} mm r:{new_position.r*constants.RADIANS_TO_DEGREES} deg t:{new_position.t*constants.RADIANS_TO_DEGREES} deg")
        self.update_position_ui()


    def move_microscope_rel(self):
        """Moves the microscope stage relative to the absolute position
        """


        logging.info("Moving Stage in Relative Coordinates")

        move = self.read_relative_move_meters()
        self.microscope.move_stage_relative(
            move
        )
        logging.info(f"Moved by dx:{self.dXchange.value():.3f} mm dy:{self.dYchange.value():.3f} mm dz:{self.dZchange.value():.3f} mm dr:{self.dRchange.value()} degrees dt:{self.dTchange.value()} degrees")

        # Get Stage Position and Set UI Display
        self.update_position_ui()

    def update_position_ui(self):
        position = self.microscope.get_stage_position()

        self.xAbs.setValue(position.x*constants.METRE_TO_MILLIMETRE)
        self.yAbs.setValue(position.y*constants.METRE_TO_MILLIMETRE)
        self.zAbs.setValue(position.z*constants.METRE_TO_MILLIMETRE)
        self.tAbs.setValue(position.t*constants.RADIANS_TO_DEGREES)
        self.rAbs.setValue(position.r*constants.RADIANS_TO_DEGREES)

        self.dXchange.setValue(0)
        self.dYchange.setValue(0)
        self.dZchange.setValue(0)
        self.dTchange.setValue(0)
        self.dRchange.setValue(0)    



    def autosave_toggle(self):
        """Toggles on Autosave which saves image everytime an image is acquired
        """

        if self.autosave_enable.checkState() == 2:
            self.image_settings.save = True
            logging.info("UI | Autosave Enabled")
            
        elif self.autosave_enable.checkState() == 0:
            self.image_settings.save = False
            logging.info("UI | Autosave Disabled")
            

        

    def save_filepath(self):
        """Opens file explorer to choose location to save image files
        """
        
        tkinter.Tk().withdraw()
        folder_path = filedialog.askdirectory()
        self.savepath_text.setText(folder_path)
        self.image_settings.save_path = folder_path
        
    ################# UI Display helper functions  ###########################################

    def hfw_box_change(self):
        ### field width in microns in UI!!!!!!!!
        self.image_settings.hfw = self.hfw_box.value() / 1.0e6


    def res_width_change(self):

        res = self.image_settings.resolution.split("x")

        res[0] = str(self.res_width.value())

        self.image_settings.resolution = "x".join(res)

    def res_height_change(self):

        resh = self.image_settings.resolution.split("x")

        resh[1] = str(self.res_height.value())

        self.image_settings.resolution = "x".join(resh)

    def image_dwell_time_change(self):
        ### dwell time in microseconds!!!!! ease of use for UI!!!!
        self.image_settings.dwell_time = self.dwell_time_setting.value()/1.0e6



    def autocontrast_check(self):
        
        autocontrast_enabled = self.autocontrast_enable.isChecked()
        self.image_settings.autocontrast = autocontrast_enabled
        logging.info(f"UI | Autocontrast Enabled: {autocontrast_enabled}")
        
    
    def gamma_check(self):

        if self.gamma_enabled.checkState() == 2:
            self.gamma_settings.enabled = True
            logging.info("UI | Gamma Enabled")
            
        elif self.gamma_enabled.checkState() == 0:
            self.gamma_settings.enabled = False
            logging.info("UI | Gamma Disabled")
            

    ##################################################################


    def update_log(self):
        
        with open(self.log_path, "r") as f:
            lines = f.read().splitlines()
            lin_len = len(lines)
            
        if self.lines != lin_len:   
            for i in reversed(range(lin_len - self.lines)):
                line_display = lines[-1-i]
                if re.search("napari.loader — DEBUG", line_display):
                    self.lines = lin_len
                    continue
                line_divided = line_display.split(",")
                time = line_divided[0]
                message = line_divided[1].split("—")
                disp_str = f"{time} | {message[-1]}"

                self.lines = lin_len
                self.CLog.setText(self.CLog2.text())
                self.CLog2.setText(self.CLog3.text())
                self.CLog3.setText(self.CLog4.text())
                self.CLog4.setText(self.CLog5.text())
                self.CLog5.setText(self.CLog6.text())
                self.CLog6.setText(self.CLog7.text())
                self.CLog7.setText(self.CLog8.text())

                self.CLog8.setText(disp_str)
      

    def connect_to_microscope(self):
        
        try:
            self.microscope, self.microscope_settings = utils.setup_session()
            self.log_path = os.path.join(self.microscope_settings.image.save_path,"logfile.log")
            logging.info("Microscope Connected")
            self.RefImage.setEnabled(True)
            self.ResetImage.setEnabled(True)
            self.take_image.setEnabled(True)
            self.save_button.setEnabled(True)
            self.move_rel_button.setEnabled(True)
            self.move_abs_button.setEnabled(True)
            self.microscope_status.setText("Microscope Connected")
            self.microscope_status.setStyleSheet("background-color: green")
        except:
            # logging.('Unable to connect to microscope')
            self.microscope_status.setText("Microscope Disconnected")
            self.microscope_status.setStyleSheet("background-color: red")
            self.RefImage.setEnabled(False)
            self.ResetImage.setEnabled(False)
            self.take_image.setEnabled(False)
            self.save_button.setEnabled(False)
            self.move_rel_button.setEnabled(False)
            self.move_abs_button.setEnabled(False)

    def disconnect_from_microscope(self):

        self.microscope.disconnect()
        self.microscope = None
        self.RefImage.setEnabled(False)
        self.ResetImage.setEnabled(False)
        self.take_image.setEnabled(False)
        self.save_button.setEnabled(False)
        self.move_rel_button.setEnabled(False)
        self.move_abs_button.setEnabled(False)
        logging.info('Microscope Disconnected')
        self.microscope_status.setText("Microscope Disconnected")
        self.microscope_status.setStyleSheet("background-color: red")

    def take_reference_images(self):
        
        # take image with both beams
        eb_image, ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.FIB_IB = ib_image
        self.FIB_EB = eb_image

        logging.info("Reference Images Taken")
        
        self.update_displays()

    def update_displays(self):

        
        viewer.layers.clear()
        viewer.add_image(self.FIB_IB.data, name="IB Image")
        viewer.add_image(self.FIB_EB.data, name="EB Image")

        if self.FIB_IB.data.shape[1] != self.res_height.value() or self.FIB_IB.data.shape[0] != self.res_width.value():
            logging.info("Actual Image resolution: " + str(self.FIB_IB.data.shape[1]) + "x" + str(self.FIB_IB.data.shape[0]))
        if self.FIB_EB.data.shape[1] != self.res_height.value() or self.FIB_EB.data.shape[0] != self.res_width.value():
            logging.info("Actual Image resolution: " + str(self.FIB_IB.data.shape[1]) + "x" + str(self.FIB_IB.data.shape[0]))


        self.reset_ui_settings()


    def click_EB_Image(self):


        tmp_beam_type = self.image_settings.beam_type
        self.image_settings.beam_type = BeamType.ELECTRON
        eb_image = acquire.new_image(self.microscope, self.image_settings)

        self.FIB_EB = eb_image

        self.update_displays()

        logging.info("EB Image Taken!")
        
    
    def click_IB_Image(self):

        tmp_beam_type = self.image_settings.beam_type
        self.image_settings.beam_type = BeamType.ION
        ib_image = acquire.new_image(self.microscope, self.image_settings)
        self.FIB_IB = ib_image

        
        self.update_displays()
        logging.info("IB Image Taken!")

    def save_EB_Image(self):
        
        save_path = os.path.join(self.image_settings.save_path, self.image_settings.label + "_eb")
        self.FIB_EB.save(save_path=save_path)

        logging.info(f"EB Image Saved to {save_path}.tif!")
        self.image_label.clear()

    def save_IB_Image(self):
        
        save_path = os.path.join(self.image_settings.save_path, self.image_settings.label + "_ib")
        self.FIB_IB.save(save_path)

        logging.info(f"IB Image Saved to {save_path}.tif!")

    def reset_images(self):

        viewer.layers['EB Image'].data = np.zeros((1,1))
        viewer.layers['IB Image'].data = np.zeros((1,1))

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
        
        logging.info("UI | Image settings returned to default values")

    def reset_ui_settings(self):

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



if __name__ == "__main__":    

    app = QtWidgets.QApplication(sys.argv)


    viewer = napari.Viewer()

    

    window = MainWindow()
   
    # window.show()
    widget = viewer.window.add_dock_widget(window)
    widget.setMinimumWidth(500)

    

    sys.exit(app.exec())
 