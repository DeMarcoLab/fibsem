# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ImageSettingsWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(535, 878)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_take_all_images = QtWidgets.QPushButton(Form)
        self.pushButton_take_all_images.setObjectName("pushButton_take_all_images")
        self.gridLayout.addWidget(self.pushButton_take_all_images, 31, 0, 1, 2)
        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 0))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -311, 501, 1076))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.selected_beam = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.selected_beam.setObjectName("selected_beam")
        self.gridLayout_2.addWidget(self.selected_beam, 1, 1, 1, 2)
        self.groupBox_utilities = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_utilities.setObjectName("groupBox_utilities")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_utilities)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.crosshair_checkbox = QtWidgets.QCheckBox(self.groupBox_utilities)
        self.crosshair_checkbox.setChecked(True)
        self.crosshair_checkbox.setObjectName("crosshair_checkbox")
        self.gridLayout_6.addWidget(self.crosshair_checkbox, 1, 1, 1, 1)
        self.ion_ruler_checkBox = QtWidgets.QCheckBox(self.groupBox_utilities)
        self.ion_ruler_checkBox.setObjectName("ion_ruler_checkBox")
        self.gridLayout_6.addWidget(self.ion_ruler_checkBox, 1, 2, 1, 1)
        self.scalebar_checkbox = QtWidgets.QCheckBox(self.groupBox_utilities)
        self.scalebar_checkbox.setObjectName("scalebar_checkbox")
        self.gridLayout_6.addWidget(self.scalebar_checkbox, 1, 0, 1, 1)
        self.ion_ruler_label = QtWidgets.QLabel(self.groupBox_utilities)
        self.ion_ruler_label.setText("")
        self.ion_ruler_label.setObjectName("ion_ruler_label")
        self.gridLayout_6.addWidget(self.ion_ruler_label, 2, 0, 1, 3)
        self.pushButton_show_alignment_area = QtWidgets.QPushButton(self.groupBox_utilities)
        self.pushButton_show_alignment_area.setObjectName("pushButton_show_alignment_area")
        self.gridLayout_6.addWidget(self.pushButton_show_alignment_area, 3, 0, 1, 3)
        self.checkBox_advanced_settings = QtWidgets.QCheckBox(self.groupBox_utilities)
        self.checkBox_advanced_settings.setObjectName("checkBox_advanced_settings")
        self.gridLayout_6.addWidget(self.checkBox_advanced_settings, 0, 0, 1, 3)
        self.gridLayout_2.addWidget(self.groupBox_utilities, 33, 0, 1, 3)
        self.groupBox_imaging = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_imaging.setObjectName("groupBox_imaging")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_imaging)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox_image_use_autocontrast = QtWidgets.QCheckBox(self.groupBox_imaging)
        self.checkBox_image_use_autocontrast.setChecked(True)
        self.checkBox_image_use_autocontrast.setObjectName("checkBox_image_use_autocontrast")
        self.gridLayout_3.addWidget(self.checkBox_image_use_autocontrast, 4, 0, 1, 1)
        self.comboBox_presets = QtWidgets.QComboBox(self.groupBox_imaging)
        self.comboBox_presets.setObjectName("comboBox_presets")
        self.gridLayout_3.addWidget(self.comboBox_presets, 3, 1, 1, 1)
        self.checkBox_image_use_autogamma = QtWidgets.QCheckBox(self.groupBox_imaging)
        self.checkBox_image_use_autogamma.setObjectName("checkBox_image_use_autogamma")
        self.gridLayout_3.addWidget(self.checkBox_image_use_autogamma, 4, 1, 1, 1)
        self.label_image_save_path = QtWidgets.QLabel(self.groupBox_imaging)
        self.label_image_save_path.setObjectName("label_image_save_path")
        self.gridLayout_3.addWidget(self.label_image_save_path, 6, 0, 1, 1)
        self.checkBox_image_save_image = QtWidgets.QCheckBox(self.groupBox_imaging)
        self.checkBox_image_save_image.setObjectName("checkBox_image_save_image")
        self.gridLayout_3.addWidget(self.checkBox_image_save_image, 5, 0, 1, 2)
        self.spinBox_image_scan_interlacing = QtWidgets.QSpinBox(self.groupBox_imaging)
        self.spinBox_image_scan_interlacing.setObjectName("spinBox_image_scan_interlacing")
        self.gridLayout_3.addWidget(self.spinBox_image_scan_interlacing, 10, 1, 1, 1)
        self.label_image_dwell_time = QtWidgets.QLabel(self.groupBox_imaging)
        self.label_image_dwell_time.setObjectName("label_image_dwell_time")
        self.gridLayout_3.addWidget(self.label_image_dwell_time, 1, 0, 1, 1)
        self.lineEdit_image_label = QtWidgets.QLineEdit(self.groupBox_imaging)
        self.lineEdit_image_label.setObjectName("lineEdit_image_label")
        self.gridLayout_3.addWidget(self.lineEdit_image_label, 7, 1, 1, 1)
        self.spinBox_image_frame_integration = QtWidgets.QSpinBox(self.groupBox_imaging)
        self.spinBox_image_frame_integration.setObjectName("spinBox_image_frame_integration")
        self.gridLayout_3.addWidget(self.spinBox_image_frame_integration, 11, 1, 1, 1)
        self.spinBox_image_line_integration = QtWidgets.QSpinBox(self.groupBox_imaging)
        self.spinBox_image_line_integration.setObjectName("spinBox_image_line_integration")
        self.gridLayout_3.addWidget(self.spinBox_image_line_integration, 8, 1, 1, 1)
        self.label_presets = QtWidgets.QLabel(self.groupBox_imaging)
        self.label_presets.setObjectName("label_presets")
        self.gridLayout_3.addWidget(self.label_presets, 3, 0, 1, 1)
        self.label_image_label = QtWidgets.QLabel(self.groupBox_imaging)
        self.label_image_label.setObjectName("label_image_label")
        self.gridLayout_3.addWidget(self.label_image_label, 7, 0, 1, 1)
        self.lineEdit_image_path = QtWidgets.QLineEdit(self.groupBox_imaging)
        self.lineEdit_image_path.setObjectName("lineEdit_image_path")
        self.gridLayout_3.addWidget(self.lineEdit_image_path, 6, 1, 1, 1)
        self.comboBox_image_resolution = QtWidgets.QComboBox(self.groupBox_imaging)
        self.comboBox_image_resolution.setObjectName("comboBox_image_resolution")
        self.gridLayout_3.addWidget(self.comboBox_image_resolution, 0, 1, 1, 1)
        self.label_image_resolution = QtWidgets.QLabel(self.groupBox_imaging)
        self.label_image_resolution.setObjectName("label_image_resolution")
        self.gridLayout_3.addWidget(self.label_image_resolution, 0, 0, 1, 1)
        self.doubleSpinBox_image_hfw = QtWidgets.QDoubleSpinBox(self.groupBox_imaging)
        self.doubleSpinBox_image_hfw.setMaximum(100000000.0)
        self.doubleSpinBox_image_hfw.setSingleStep(1.0)
        self.doubleSpinBox_image_hfw.setProperty("value", 150.0)
        self.doubleSpinBox_image_hfw.setObjectName("doubleSpinBox_image_hfw")
        self.gridLayout_3.addWidget(self.doubleSpinBox_image_hfw, 2, 1, 1, 1)
        self.label_image_hfw = QtWidgets.QLabel(self.groupBox_imaging)
        self.label_image_hfw.setObjectName("label_image_hfw")
        self.gridLayout_3.addWidget(self.label_image_hfw, 2, 0, 1, 1)
        self.doubleSpinBox_image_dwell_time = QtWidgets.QDoubleSpinBox(self.groupBox_imaging)
        self.doubleSpinBox_image_dwell_time.setMaximum(10000000000000.0)
        self.doubleSpinBox_image_dwell_time.setSingleStep(0.01)
        self.doubleSpinBox_image_dwell_time.setProperty("value", 1.0)
        self.doubleSpinBox_image_dwell_time.setObjectName("doubleSpinBox_image_dwell_time")
        self.gridLayout_3.addWidget(self.doubleSpinBox_image_dwell_time, 1, 1, 1, 1)
        self.checkBox_image_line_integration = QtWidgets.QCheckBox(self.groupBox_imaging)
        self.checkBox_image_line_integration.setObjectName("checkBox_image_line_integration")
        self.gridLayout_3.addWidget(self.checkBox_image_line_integration, 8, 0, 1, 1)
        self.checkBox_image_scan_interlacing = QtWidgets.QCheckBox(self.groupBox_imaging)
        self.checkBox_image_scan_interlacing.setObjectName("checkBox_image_scan_interlacing")
        self.gridLayout_3.addWidget(self.checkBox_image_scan_interlacing, 10, 0, 1, 1)
        self.checkBox_image_frame_integration = QtWidgets.QCheckBox(self.groupBox_imaging)
        self.checkBox_image_frame_integration.setObjectName("checkBox_image_frame_integration")
        self.gridLayout_3.addWidget(self.checkBox_image_frame_integration, 11, 0, 1, 1)
        self.checkBox_image_drift_correction = QtWidgets.QCheckBox(self.groupBox_imaging)
        self.checkBox_image_drift_correction.setObjectName("checkBox_image_drift_correction")
        self.gridLayout_3.addWidget(self.checkBox_image_drift_correction, 12, 0, 1, 2)
        self.gridLayout_2.addWidget(self.groupBox_imaging, 2, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 36, 0, 1, 2)
        self.label_beam_type = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_beam_type.setObjectName("label_beam_type")
        self.gridLayout_2.addWidget(self.label_beam_type, 1, 0, 1, 1)
        self.groupBox_detector = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_detector.setObjectName("groupBox_detector")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_detector)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.detector_contrast_label = QtWidgets.QLabel(self.groupBox_detector)
        self.detector_contrast_label.setObjectName("detector_contrast_label")
        self.gridLayout_5.addWidget(self.detector_contrast_label, 3, 2, 1, 1)
        self.label_detector_type = QtWidgets.QLabel(self.groupBox_detector)
        self.label_detector_type.setObjectName("label_detector_type")
        self.gridLayout_5.addWidget(self.label_detector_type, 0, 0, 1, 1)
        self.detector_brightness_slider = QtWidgets.QSlider(self.groupBox_detector)
        self.detector_brightness_slider.setMaximum(100)
        self.detector_brightness_slider.setSliderPosition(50)
        self.detector_brightness_slider.setOrientation(QtCore.Qt.Horizontal)
        self.detector_brightness_slider.setInvertedAppearance(False)
        self.detector_brightness_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.detector_brightness_slider.setTickInterval(10)
        self.detector_brightness_slider.setObjectName("detector_brightness_slider")
        self.gridLayout_5.addWidget(self.detector_brightness_slider, 2, 1, 1, 1)
        self.detector_mode_combobox = QtWidgets.QComboBox(self.groupBox_detector)
        self.detector_mode_combobox.setObjectName("detector_mode_combobox")
        self.gridLayout_5.addWidget(self.detector_mode_combobox, 1, 1, 1, 2)
        self.detector_brightness_label = QtWidgets.QLabel(self.groupBox_detector)
        self.detector_brightness_label.setObjectName("detector_brightness_label")
        self.gridLayout_5.addWidget(self.detector_brightness_label, 2, 2, 1, 1)
        self.detector_type_combobox = QtWidgets.QComboBox(self.groupBox_detector)
        self.detector_type_combobox.setObjectName("detector_type_combobox")
        self.gridLayout_5.addWidget(self.detector_type_combobox, 0, 1, 1, 2)
        self.label_detector_brightness = QtWidgets.QLabel(self.groupBox_detector)
        self.label_detector_brightness.setObjectName("label_detector_brightness")
        self.gridLayout_5.addWidget(self.label_detector_brightness, 2, 0, 1, 1)
        self.detector_contrast_slider = QtWidgets.QSlider(self.groupBox_detector)
        self.detector_contrast_slider.setMaximum(100)
        self.detector_contrast_slider.setSliderPosition(50)
        self.detector_contrast_slider.setOrientation(QtCore.Qt.Horizontal)
        self.detector_contrast_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.detector_contrast_slider.setTickInterval(10)
        self.detector_contrast_slider.setObjectName("detector_contrast_slider")
        self.gridLayout_5.addWidget(self.detector_contrast_slider, 3, 1, 1, 1)
        self.label_detector_contrast = QtWidgets.QLabel(self.groupBox_detector)
        self.label_detector_contrast.setObjectName("label_detector_contrast")
        self.gridLayout_5.addWidget(self.label_detector_contrast, 3, 0, 1, 1)
        self.label_detector_mode = QtWidgets.QLabel(self.groupBox_detector)
        self.label_detector_mode.setObjectName("label_detector_mode")
        self.gridLayout_5.addWidget(self.label_detector_mode, 1, 0, 1, 1)
        self.set_detector_button = QtWidgets.QPushButton(self.groupBox_detector)
        self.set_detector_button.setObjectName("set_detector_button")
        self.gridLayout_5.addWidget(self.set_detector_button, 4, 0, 1, 3)
        self.gridLayout_2.addWidget(self.groupBox_detector, 4, 0, 1, 3)
        self.groupBox_beam = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_beam.setObjectName("groupBox_beam")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_beam)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.doubleSpinBox_stigmation_y = QtWidgets.QDoubleSpinBox(self.groupBox_beam)
        self.doubleSpinBox_stigmation_y.setMaximum(1000.0)
        self.doubleSpinBox_stigmation_y.setSingleStep(0.01)
        self.doubleSpinBox_stigmation_y.setObjectName("doubleSpinBox_stigmation_y")
        self.gridLayout_4.addWidget(self.doubleSpinBox_stigmation_y, 3, 2, 1, 1)
        self.spinBox_beam_scan_rotation = QtWidgets.QSpinBox(self.groupBox_beam)
        self.spinBox_beam_scan_rotation.setMaximum(360)
        self.spinBox_beam_scan_rotation.setObjectName("spinBox_beam_scan_rotation")
        self.gridLayout_4.addWidget(self.spinBox_beam_scan_rotation, 5, 1, 1, 2)
        self.doubleSpinBox_stigmation_x = QtWidgets.QDoubleSpinBox(self.groupBox_beam)
        self.doubleSpinBox_stigmation_x.setMinimum(-1000.0)
        self.doubleSpinBox_stigmation_x.setMaximum(1000.0)
        self.doubleSpinBox_stigmation_x.setSingleStep(0.01)
        self.doubleSpinBox_stigmation_x.setObjectName("doubleSpinBox_stigmation_x")
        self.gridLayout_4.addWidget(self.doubleSpinBox_stigmation_x, 3, 1, 1, 1)
        self.label_beam_voltage = QtWidgets.QLabel(self.groupBox_beam)
        self.label_beam_voltage.setObjectName("label_beam_voltage")
        self.gridLayout_4.addWidget(self.label_beam_voltage, 2, 0, 1, 1)
        self.label_beam_working_distance = QtWidgets.QLabel(self.groupBox_beam)
        self.label_beam_working_distance.setObjectName("label_beam_working_distance")
        self.gridLayout_4.addWidget(self.label_beam_working_distance, 0, 0, 1, 1)
        self.doubleSpinBox_working_distance = QtWidgets.QDoubleSpinBox(self.groupBox_beam)
        self.doubleSpinBox_working_distance.setMaximum(1000.0)
        self.doubleSpinBox_working_distance.setSingleStep(0.01)
        self.doubleSpinBox_working_distance.setObjectName("doubleSpinBox_working_distance")
        self.gridLayout_4.addWidget(self.doubleSpinBox_working_distance, 0, 1, 1, 2)
        self.doubleSpinBox_beam_voltage = QtWidgets.QDoubleSpinBox(self.groupBox_beam)
        self.doubleSpinBox_beam_voltage.setMaximum(100000.0)
        self.doubleSpinBox_beam_voltage.setObjectName("doubleSpinBox_beam_voltage")
        self.gridLayout_4.addWidget(self.doubleSpinBox_beam_voltage, 2, 1, 1, 2)
        self.doubleSpinBox_shift_y = QtWidgets.QDoubleSpinBox(self.groupBox_beam)
        self.doubleSpinBox_shift_y.setMinimum(-1000.0)
        self.doubleSpinBox_shift_y.setMaximum(1000.0)
        self.doubleSpinBox_shift_y.setSingleStep(0.01)
        self.doubleSpinBox_shift_y.setObjectName("doubleSpinBox_shift_y")
        self.gridLayout_4.addWidget(self.doubleSpinBox_shift_y, 4, 2, 1, 1)
        self.label_beam_scan_rotation = QtWidgets.QLabel(self.groupBox_beam)
        self.label_beam_scan_rotation.setObjectName("label_beam_scan_rotation")
        self.gridLayout_4.addWidget(self.label_beam_scan_rotation, 5, 0, 1, 1)
        self.label_beam_current = QtWidgets.QLabel(self.groupBox_beam)
        self.label_beam_current.setObjectName("label_beam_current")
        self.gridLayout_4.addWidget(self.label_beam_current, 1, 0, 1, 1)
        self.label_shift = QtWidgets.QLabel(self.groupBox_beam)
        self.label_shift.setObjectName("label_shift")
        self.gridLayout_4.addWidget(self.label_shift, 4, 0, 1, 1)
        self.doubleSpinBox_shift_x = QtWidgets.QDoubleSpinBox(self.groupBox_beam)
        self.doubleSpinBox_shift_x.setMinimum(-1000.0)
        self.doubleSpinBox_shift_x.setMaximum(1000.0)
        self.doubleSpinBox_shift_x.setSingleStep(0.01)
        self.doubleSpinBox_shift_x.setObjectName("doubleSpinBox_shift_x")
        self.gridLayout_4.addWidget(self.doubleSpinBox_shift_x, 4, 1, 1, 1)
        self.doubleSpinBox_beam_current = QtWidgets.QDoubleSpinBox(self.groupBox_beam)
        self.doubleSpinBox_beam_current.setMaximum(1000.0)
        self.doubleSpinBox_beam_current.setObjectName("doubleSpinBox_beam_current")
        self.gridLayout_4.addWidget(self.doubleSpinBox_beam_current, 1, 1, 1, 2)
        self.label_stigmation = QtWidgets.QLabel(self.groupBox_beam)
        self.label_stigmation.setObjectName("label_stigmation")
        self.gridLayout_4.addWidget(self.label_stigmation, 3, 0, 1, 1)
        self.button_set_beam_settings = QtWidgets.QPushButton(self.groupBox_beam)
        self.button_set_beam_settings.setObjectName("button_set_beam_settings")
        self.gridLayout_4.addWidget(self.button_set_beam_settings, 6, 0, 1, 3)
        self.gridLayout_2.addWidget(self.groupBox_beam, 3, 0, 1, 3)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 1, 0, 1, 2)
        self.pushButton_take_image = QtWidgets.QPushButton(Form)
        self.pushButton_take_image.setObjectName("pushButton_take_image")
        self.gridLayout.addWidget(self.pushButton_take_image, 29, 0, 1, 2)
        self.pushButton_acquire_sem_image = QtWidgets.QPushButton(Form)
        self.pushButton_acquire_sem_image.setObjectName("pushButton_acquire_sem_image")
        self.gridLayout.addWidget(self.pushButton_acquire_sem_image, 30, 0, 1, 1)
        self.pushButton_acquire_fib_image = QtWidgets.QPushButton(Form)
        self.pushButton_acquire_fib_image.setObjectName("pushButton_acquire_fib_image")
        self.gridLayout.addWidget(self.pushButton_acquire_fib_image, 30, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.pushButton_take_image, self.pushButton_take_all_images)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_take_all_images.setText(_translate("Form", "Acquire All Images"))
        self.groupBox_utilities.setTitle(_translate("Form", "Utilities"))
        self.crosshair_checkbox.setText(_translate("Form", "Cross Hair"))
        self.ion_ruler_checkBox.setText(_translate("Form", "Ruler "))
        self.scalebar_checkbox.setText(_translate("Form", "Scale Bar"))
        self.pushButton_show_alignment_area.setText(_translate("Form", "Show Alignment Area"))
        self.checkBox_advanced_settings.setText(_translate("Form", "Show Advanced"))
        self.groupBox_imaging.setTitle(_translate("Form", "Imaging"))
        self.checkBox_image_use_autocontrast.setText(_translate("Form", "AutoContrast"))
        self.checkBox_image_use_autogamma.setText(_translate("Form", "AutoGamma"))
        self.label_image_save_path.setText(_translate("Form", "Path"))
        self.checkBox_image_save_image.setText(_translate("Form", "Save Image"))
        self.label_image_dwell_time.setText(_translate("Form", "Dwell Time"))
        self.label_presets.setText(_translate("Form", "Preset"))
        self.label_image_label.setText(_translate("Form", "Filename"))
        self.label_image_resolution.setText(_translate("Form", "Resolution (px)"))
        self.doubleSpinBox_image_hfw.setSuffix(_translate("Form", " um"))
        self.label_image_hfw.setText(_translate("Form", "Field of View"))
        self.doubleSpinBox_image_dwell_time.setSuffix(_translate("Form", " us"))
        self.checkBox_image_line_integration.setText(_translate("Form", "Line Integration"))
        self.checkBox_image_scan_interlacing.setText(_translate("Form", "Scan Interlacing"))
        self.checkBox_image_frame_integration.setText(_translate("Form", "Frame Integration"))
        self.checkBox_image_drift_correction.setText(_translate("Form", "Use Drift Correction"))
        self.label_beam_type.setText(_translate("Form", "Beam"))
        self.groupBox_detector.setTitle(_translate("Form", "Detector"))
        self.detector_contrast_label.setText(_translate("Form", "50%"))
        self.label_detector_type.setText(_translate("Form", "Type"))
        self.detector_brightness_label.setText(_translate("Form", "50%"))
        self.label_detector_brightness.setText(_translate("Form", "Brightness"))
        self.label_detector_contrast.setText(_translate("Form", "Contrast"))
        self.label_detector_mode.setText(_translate("Form", "Mode"))
        self.set_detector_button.setText(_translate("Form", "Set Detector Settings"))
        self.groupBox_beam.setTitle(_translate("Form", "Beam"))
        self.spinBox_beam_scan_rotation.setSuffix(_translate("Form", " deg"))
        self.label_beam_voltage.setText(_translate("Form", "Voltage"))
        self.label_beam_working_distance.setText(_translate("Form", "Working Distance"))
        self.doubleSpinBox_working_distance.setSuffix(_translate("Form", " mm"))
        self.doubleSpinBox_beam_voltage.setSuffix(_translate("Form", " kV"))
        self.doubleSpinBox_shift_y.setSuffix(_translate("Form", " um"))
        self.label_beam_scan_rotation.setText(_translate("Form", "Scan Rotation"))
        self.label_beam_current.setText(_translate("Form", "Beam Current"))
        self.label_shift.setText(_translate("Form", "Shift (x,y)"))
        self.doubleSpinBox_shift_x.setSuffix(_translate("Form", " um"))
        self.doubleSpinBox_beam_current.setSuffix(_translate("Form", " pA"))
        self.label_stigmation.setText(_translate("Form", "Stigmation (x,y)"))
        self.button_set_beam_settings.setText(_translate("Form", "Set Beam Settings"))
        self.pushButton_take_image.setText(_translate("Form", "Acquire Image"))
        self.pushButton_acquire_sem_image.setText(_translate("Form", "Acquire SEM Image"))
        self.pushButton_acquire_fib_image.setText(_translate("Form", "Acquire FIB Image"))
