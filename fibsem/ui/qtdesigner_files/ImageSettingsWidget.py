# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ImageSettingsWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(369, 765)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 25, 0, 1, 3)
        self.beam_current = QtWidgets.QDoubleSpinBox(Form)
        self.beam_current.setMaximum(1000.0)
        self.beam_current.setObjectName("beam_current")
        self.gridLayout.addWidget(self.beam_current, 18, 1, 1, 1)
        self.lineEdit_image_label = QtWidgets.QLineEdit(Form)
        self.lineEdit_image_label.setObjectName("lineEdit_image_label")
        self.gridLayout.addWidget(self.lineEdit_image_label, 10, 1, 1, 2)
        self.detector_brightness_label = QtWidgets.QLabel(Form)
        self.detector_brightness_label.setObjectName("detector_brightness_label")
        self.gridLayout.addWidget(self.detector_brightness_label, 14, 2, 1, 1)
        self.detector_type_combobox = QtWidgets.QComboBox(Form)
        self.detector_type_combobox.setObjectName("detector_type_combobox")
        self.gridLayout.addWidget(self.detector_type_combobox, 12, 1, 1, 2)
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 20, 0, 1, 1)
        self.stigmation_x = QtWidgets.QDoubleSpinBox(Form)
        self.stigmation_x.setMaximum(1000.0)
        self.stigmation_x.setObjectName("stigmation_x")
        self.gridLayout.addWidget(self.stigmation_x, 20, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 17, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 21, 0, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 11, 0, 1, 1)
        self.detector_mode_combobox = QtWidgets.QComboBox(Form)
        self.detector_mode_combobox.setObjectName("detector_mode_combobox")
        self.gridLayout.addWidget(self.detector_mode_combobox, 13, 1, 1, 2)
        self.spinBox_resolution_x = QtWidgets.QSpinBox(Form)
        self.spinBox_resolution_x.setMinimum(1)
        self.spinBox_resolution_x.setMaximum(999999999)
        self.spinBox_resolution_x.setProperty("value", 1536)
        self.spinBox_resolution_x.setObjectName("spinBox_resolution_x")
        self.gridLayout.addWidget(self.spinBox_resolution_x, 4, 1, 1, 1)
        self.selected_beam = QtWidgets.QComboBox(Form)
        self.selected_beam.setObjectName("selected_beam")
        self.gridLayout.addWidget(self.selected_beam, 1, 1, 1, 2)
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 15, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 22, 0, 1, 3)
        self.label_image_dwell_time = QtWidgets.QLabel(Form)
        self.label_image_dwell_time.setObjectName("label_image_dwell_time")
        self.gridLayout.addWidget(self.label_image_dwell_time, 5, 0, 1, 1)
        self.label_image_hfw = QtWidgets.QLabel(Form)
        self.label_image_hfw.setObjectName("label_image_hfw")
        self.gridLayout.addWidget(self.label_image_hfw, 6, 0, 1, 1)
        self.detector_contrast_label = QtWidgets.QLabel(Form)
        self.detector_contrast_label.setObjectName("detector_contrast_label")
        self.gridLayout.addWidget(self.detector_contrast_label, 15, 2, 1, 1)
        self.working_distance = QtWidgets.QDoubleSpinBox(Form)
        self.working_distance.setMaximum(1000.0)
        self.working_distance.setObjectName("working_distance")
        self.gridLayout.addWidget(self.working_distance, 17, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 13, 0, 1, 1)
        self.label_image_resolution = QtWidgets.QLabel(Form)
        self.label_image_resolution.setObjectName("label_image_resolution")
        self.gridLayout.addWidget(self.label_image_resolution, 4, 0, 1, 1)
        self.label_image_label = QtWidgets.QLabel(Form)
        self.label_image_label.setObjectName("label_image_label")
        self.gridLayout.addWidget(self.label_image_label, 10, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 2, 0, 1, 1)
        self.checkBox_image_use_autocontrast = QtWidgets.QCheckBox(Form)
        self.checkBox_image_use_autocontrast.setChecked(True)
        self.checkBox_image_use_autocontrast.setObjectName("checkBox_image_use_autocontrast")
        self.gridLayout.addWidget(self.checkBox_image_use_autocontrast, 7, 0, 1, 1)
        self.stigmation_y = QtWidgets.QDoubleSpinBox(Form)
        self.stigmation_y.setMaximum(1000.0)
        self.stigmation_y.setObjectName("stigmation_y")
        self.gridLayout.addWidget(self.stigmation_y, 20, 2, 1, 1)
        self.doubleSpinBox_image_dwell_time = QtWidgets.QDoubleSpinBox(Form)
        self.doubleSpinBox_image_dwell_time.setMaximum(10000000000000.0)
        self.doubleSpinBox_image_dwell_time.setSingleStep(0.01)
        self.doubleSpinBox_image_dwell_time.setProperty("value", 1.0)
        self.doubleSpinBox_image_dwell_time.setObjectName("doubleSpinBox_image_dwell_time")
        self.gridLayout.addWidget(self.doubleSpinBox_image_dwell_time, 5, 1, 1, 1)
        self.checkBox_image_use_autogamma = QtWidgets.QCheckBox(Form)
        self.checkBox_image_use_autogamma.setObjectName("checkBox_image_use_autogamma")
        self.gridLayout.addWidget(self.checkBox_image_use_autogamma, 7, 1, 1, 1)
        self.detector_brightness_slider = QtWidgets.QSlider(Form)
        self.detector_brightness_slider.setMaximum(100)
        self.detector_brightness_slider.setSliderPosition(50)
        self.detector_brightness_slider.setOrientation(QtCore.Qt.Horizontal)
        self.detector_brightness_slider.setInvertedAppearance(False)
        self.detector_brightness_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.detector_brightness_slider.setTickInterval(10)
        self.detector_brightness_slider.setObjectName("detector_brightness_slider")
        self.gridLayout.addWidget(self.detector_brightness_slider, 14, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 12, 0, 1, 1)
        self.shift_y = QtWidgets.QDoubleSpinBox(Form)
        self.shift_y.setMaximum(1000.0)
        self.shift_y.setObjectName("shift_y")
        self.gridLayout.addWidget(self.shift_y, 21, 2, 1, 1)
        self.shift_x = QtWidgets.QDoubleSpinBox(Form)
        self.shift_x.setMaximum(1000.0)
        self.shift_x.setObjectName("shift_x")
        self.gridLayout.addWidget(self.shift_x, 21, 1, 1, 1)
        self.spinBox_resolution_y = QtWidgets.QSpinBox(Form)
        self.spinBox_resolution_y.setMinimum(1)
        self.spinBox_resolution_y.setMaximum(999999999)
        self.spinBox_resolution_y.setProperty("value", 1024)
        self.spinBox_resolution_y.setObjectName("spinBox_resolution_y")
        self.gridLayout.addWidget(self.spinBox_resolution_y, 4, 2, 1, 1)
        self.lineEdit_image_path = QtWidgets.QLineEdit(Form)
        self.lineEdit_image_path.setObjectName("lineEdit_image_path")
        self.gridLayout.addWidget(self.lineEdit_image_path, 9, 1, 1, 2)
        self.label_image_save_path = QtWidgets.QLabel(Form)
        self.label_image_save_path.setObjectName("label_image_save_path")
        self.gridLayout.addWidget(self.label_image_save_path, 9, 0, 1, 1)
        self.checkBox_image_save_image = QtWidgets.QCheckBox(Form)
        self.checkBox_image_save_image.setObjectName("checkBox_image_save_image")
        self.gridLayout.addWidget(self.checkBox_image_save_image, 8, 0, 1, 1)
        self.button_set_beam_settings = QtWidgets.QPushButton(Form)
        self.button_set_beam_settings.setObjectName("button_set_beam_settings")
        self.gridLayout.addWidget(self.button_set_beam_settings, 16, 1, 1, 2)
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 19, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 16, 0, 1, 1)
        self.pushButton_take_image = QtWidgets.QPushButton(Form)
        self.pushButton_take_image.setObjectName("pushButton_take_image")
        self.gridLayout.addWidget(self.pushButton_take_image, 23, 0, 1, 3)
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 14, 0, 1, 1)
        self.detector_contrast_slider = QtWidgets.QSlider(Form)
        self.detector_contrast_slider.setMaximum(100)
        self.detector_contrast_slider.setSliderPosition(50)
        self.detector_contrast_slider.setOrientation(QtCore.Qt.Horizontal)
        self.detector_contrast_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.detector_contrast_slider.setTickInterval(10)
        self.detector_contrast_slider.setObjectName("detector_contrast_slider")
        self.gridLayout.addWidget(self.detector_contrast_slider, 15, 1, 1, 1)
        self.set_detector_button = QtWidgets.QPushButton(Form)
        self.set_detector_button.setObjectName("set_detector_button")
        self.gridLayout.addWidget(self.set_detector_button, 11, 1, 1, 2)
        self.doubleSpinBox_image_hfw = QtWidgets.QDoubleSpinBox(Form)
        self.doubleSpinBox_image_hfw.setMaximum(2700.0)
        self.doubleSpinBox_image_hfw.setSingleStep(0.1)
        self.doubleSpinBox_image_hfw.setProperty("value", 150.0)
        self.doubleSpinBox_image_hfw.setObjectName("doubleSpinBox_image_hfw")
        self.gridLayout.addWidget(self.doubleSpinBox_image_hfw, 6, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)
        self.pushButton_take_all_images = QtWidgets.QPushButton(Form)
        self.pushButton_take_all_images.setObjectName("pushButton_take_all_images")
        self.gridLayout.addWidget(self.pushButton_take_all_images, 24, 0, 1, 3)
        self.beam_voltage = QtWidgets.QDoubleSpinBox(Form)
        self.beam_voltage.setMaximum(100000.0)
        self.beam_voltage.setObjectName("beam_voltage")
        self.gridLayout.addWidget(self.beam_voltage, 19, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 18, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.detector_brightness_label.setText(_translate("Form", "50%"))
        self.label_11.setText(_translate("Form", "Stigmation (x,y)"))
        self.label_7.setText(_translate("Form", "Working Distance (mm)"))
        self.label_12.setText(_translate("Form", "Shift (x,y) (um)"))
        self.label.setText(_translate("Form", "Detector Settings"))
        self.label_5.setText(_translate("Form", "Contrast (%)"))
        self.label_image_dwell_time.setText(_translate("Form", "Dwell Time (us)"))
        self.label_image_hfw.setText(_translate("Form", "Horizontal Field Width (um)"))
        self.detector_contrast_label.setText(_translate("Form", "50%"))
        self.label_3.setText(_translate("Form", "Mode"))
        self.label_image_resolution.setText(_translate("Form", "Resolution (px)"))
        self.label_image_label.setText(_translate("Form", "Label"))
        self.label_title.setText(_translate("Form", "Image Settings"))
        self.checkBox_image_use_autocontrast.setText(_translate("Form", "AutoContrast"))
        self.checkBox_image_use_autogamma.setText(_translate("Form", "AutoGamma"))
        self.label_2.setText(_translate("Form", "Type"))
        self.label_image_save_path.setText(_translate("Form", "Path"))
        self.checkBox_image_save_image.setText(_translate("Form", "Save Image"))
        self.button_set_beam_settings.setText(_translate("Form", "Set Beam Settings"))
        self.label_10.setText(_translate("Form", "Voltage (kV)"))
        self.label_6.setText(_translate("Form", "Beam Settings"))
        self.pushButton_take_image.setText(_translate("Form", "Acquire Image"))
        self.label_4.setText(_translate("Form", "Brightness (%)"))
        self.set_detector_button.setText(_translate("Form", "Set Detector Settings"))
        self.label_8.setText(_translate("Form", "Beam"))
        self.pushButton_take_all_images.setText(_translate("Form", "Acquire All Images"))
        self.label_9.setText(_translate("Form", "Beam Current (pA)"))
