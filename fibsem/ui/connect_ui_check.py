import sys
import re
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, MillingSettings
from fibsem.ui.qtdesigner_files import connect
from fibsem import utils, acquire
import fibsem.movement as movement
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, MillingSettings, Point
import fibsem.conversions as conversions
from enum import Enum
import os
import tkinter
from tkinter import filedialog
import fibsem.constants as constants
from qtpy import QtWidgets
from PyQt5.QtCore import QTimer
import numpy as np
import logging
import napari

class MovementMode(Enum):
    Stable = 1
    Eucentric = 2
    # Needle = 3

class MovementType(Enum):
    StableEnabled = 0 
    EucentricEnabled = 1
    TiltEnabled = 2



class MainWindow(QtWidgets.QMainWindow, connect.Ui_MainWindow):
    def __init__(self,*args,obj=None,**kwargs) -> None:
        super(MainWindow,self).__init__(*args,**kwargs)
        self.setupUi(self)

        # setting up ui 
        self.setup_connections()
        self.lines = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_log)
        self.timer.start(1000)
        
        # Gamma and Image Settings

        self.FIB_IB = FibsemImage(data=np.zeros((1536,1024), dtype=np.uint8))
        self.FIB_EB = FibsemImage(data=np.zeros((1536,1024), dtype=np.uint8))

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

        # Milling settings set up
        self.pushButton_milling.clicked.connect(self.milling_protocol)



    def save_image_beams(self):
        if self.image_label.text() != "":
            self.image_settings.label = self.image_label.text()

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

    def milling_protocol(self):
        from FibsemMilling import milling_protocol

        mill_settings = MillingSettings(
            width = self.width_milling.value()*constants.MICRO_TO_SI,
            height = self.height_milling.value()*constants.MICRO_TO_SI,
            depth= self.depth_milling.value()*constants.MICRO_TO_SI,
            rotation= self.rotation_milling.value()*constants.DEGREES_TO_RADIANS,
            centre_x= self.center_x_milling.value()*constants.MILLIMETRE_TO_METRE,
            centre_y= self.center_y_milling.value()*constants.MILLIMETRE_TO_METRE,
            milling_current= self.milling_current.value()*constants.NANO_TO_SI,
            scan_direction= self.scan_direction.currentText(),
            cleaning_cross_section= self.checkBox_cross_section.isChecked(),
        )

        milling_protocol(self.microscope, self.image_settings, mill_settings, application_file="autolamella", patterning_mode="Serial")

########################### Movement Functionality ##########################################

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

    def get_data_from_coord(self, coords: tuple) -> tuple:

        # check inside image dimensions, (y, x)
        eb_shape = self.FIB_EB.data.shape[0], self.FIB_EB.data.shape[1] // 2
        ib_shape = self.FIB_IB.data.shape[0], self.FIB_IB.data.shape[1]

        if (coords[0] > 0 and coords[0] < eb_shape[0]) and (coords[1] > 0 and coords[1] < eb_shape[1]):
            image = self.FIB_EB
            beam_type = BeamType.ELECTRON

        elif (coords[0] > 0 and coords[0] < ib_shape[0]) and (coords[1] > eb_shape[0] and coords[1] < ib_shape[1]):
            image = self.FIB_IB
            coords = (coords[0], coords[1] - ib_shape[1] // 2)
            beam_type = BeamType.ION
        else:
            beam_type, image = None, None
        
        return coords, beam_type, image


    def _double_click(self, layer, event):

        # get coords
        coords = layer.world_to_data(event.position)

        # TODO: dimensions are mixed which makes this confusing to interpret... resolve
        
        coords, beam_type, image = self.get_data_from_coord(coords)

        if beam_type is None:
            napari.utils.notifications.show_info(f"Clicked outside image dimensions. Please click inside the image to move.")
            return

        point = conversions.image_to_microscope_image_coordinates(Point(x=coords[1], y=coords[0]), 
                image.data, image.metadata.pixel_size.x)  

        logging.debug(f"Movement: {self.movement_mode.name} | COORD {coords} | SHIFT {point.x:.2e}, {point.y:.2e} | {beam_type}")

        # move
        self.movement_mode = MovementMode[self.comboBox_movement_mode.currentText()]

        # eucentric is only supported for ION beam
        if beam_type is BeamType.ION and self.movement_mode is MovementMode.Eucentric:
            self.microscope.eucentric_move(
                microscope=self.microscope, 
                settings=self.microscope_settings,
                dy=-point.y
            )

        else:
            # corrected stage movement
            self.microscope.stable_move(
                microscope=self.microscope,
                settings=self.microscope_settings,
                dx=point.x,
                dy=point.y,
                beam_type=beam_type,
            )

        self.update_displays()


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
            self.image_settings.gamma.enabled = True
            logging.info("UI | Gamma Enabled")
            
        elif self.gamma_enabled.checkState() == 0:
            self.image_settings.gamma.enabled = False
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
            self.image_settings = self.microscope_settings.image
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
        self.microscope_settings = None
        self.RefImage.setEnabled(False)
        self.ResetImage.setEnabled(False)
        self.take_image.setEnabled(False)
        self.save_button.setEnabled(False)
        self.move_rel_button.setEnabled(False)
        self.move_abs_button.setEnabled(False)
        logging.info('Microscope Disconnected')
        self.microscope_status.setText("Microscope Disconnected")
        self.microscope_status.setStyleSheet("background-color: red")


###################################### Imaging ##########################################

    def take_reference_images(self):
        
        # take image with both beams
        eb_image, ib_image = acquire.take_reference_images(self.microscope, self.image_settings)

        self.FIB_IB = ib_image
        self.FIB_EB = eb_image

        logging.info("Reference Images Taken")
        
        self.update_displays()

    def update_displays(self):

        
        viewer.layers.clear()
        self.ib_layer = viewer.add_image(self.FIB_IB.data, name="IB Image")
        self.eb_layer = viewer.add_image(self.FIB_EB.data, name="EB Image")

        # if self.FIB_IB.data.shape[1] != self.res_height.value() or self.FIB_IB.data.shape[0] != self.res_width.value():
        #     logging.info("IB | Actual Image resolution: " + str(self.FIB_IB.data.shape[1]) + "x" + str(self.FIB_IB.data.shape[0]))
        # if self.FIB_EB.data.shape[1] != self.res_height.value() or self.FIB_EB.data.shape[0] != self.res_width.value():
        #     logging.info("EB | Actual Image resolution: " + str(self.FIB_IB.data.shape[1]) + "x" + str(self.FIB_IB.data.shape[0]))

        #viewer.camera.zoom = 0.4

        self.ib_layer.mouse_double_click_callbacks.append(self._double_click)
        self.eb_layer.mouse_double_click_callbacks.append(self._double_click)

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
        self.FIB_IB = FibsemImage(data=np.zeros((1536,1024), dtype=np.uint8))
        self.FIB_EB = FibsemImage(data=np.zeros((1536,1024), dtype=np.uint8))

    def reset_image_and_gammaSettings(self):

        settings = utils.load_settings_from_config()
        self.image_settings = settings.image
        
        self.reset_ui_settings()
        
        logging.info("UI | Image settings returned to default values")

    def reset_ui_settings(self):

        self.dwell_time_setting.setValue(self.image_settings.dwell_time * constants.SI_TO_MICRO)
        self.hfw_box.setValue(int(self.image_settings.hfw * 1e6))

        res_ful = self.image_settings.resolution.split("x")

        self.res_width.setValue(int(res_ful[0]))
        self.res_height.setValue(int(res_ful[1]))


        if self.image_settings.gamma.enabled:
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
 