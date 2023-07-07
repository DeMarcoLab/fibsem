# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemSystemSetupWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(543, 753)
        font = QtGui.QFont()
        font.setPointSize(10)
        Form.setFont(font)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_ipadress = QtWidgets.QLineEdit(Form)
        self.lineEdit_ipadress.setObjectName("lineEdit_ipadress")
        self.gridLayout.addWidget(self.lineEdit_ipadress, 2, 1, 1, 1)
        self.comboBox_manufacturer = QtWidgets.QComboBox(Form)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.comboBox_manufacturer.setFont(font)
        self.comboBox_manufacturer.setObjectName("comboBox_manufacturer")
        self.gridLayout.addWidget(self.comboBox_manufacturer, 3, 1, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tabWidget.setFont(font)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.stage_tab = QtWidgets.QWidget()
        self.stage_tab.setObjectName("stage_tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.stage_tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.needleStageHeightLimitnMmDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.stage_tab)
        self.needleStageHeightLimitnMmDoubleSpinBox.setMinimum(-99.0)
        self.needleStageHeightLimitnMmDoubleSpinBox.setObjectName("needleStageHeightLimitnMmDoubleSpinBox")
        self.gridLayout_2.addWidget(self.needleStageHeightLimitnMmDoubleSpinBox, 6, 1, 1, 1)
        self.rotationFlatToIonLabel = QtWidgets.QLabel(self.stage_tab)
        self.rotationFlatToIonLabel.setObjectName("rotationFlatToIonLabel")
        self.gridLayout_2.addWidget(self.rotationFlatToIonLabel, 2, 0, 1, 1)
        self.label_header_stage = QtWidgets.QLabel(self.stage_tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_header_stage.setFont(font)
        self.label_header_stage.setObjectName("label_header_stage")
        self.gridLayout_2.addWidget(self.label_header_stage, 0, 0, 1, 1)
        self.needleStageHeightLimitnMmLabel = QtWidgets.QLabel(self.stage_tab)
        self.needleStageHeightLimitnMmLabel.setObjectName("needleStageHeightLimitnMmLabel")
        self.gridLayout_2.addWidget(self.needleStageHeightLimitnMmLabel, 6, 0, 1, 1)
        self.tiltFlatToElectronLabel = QtWidgets.QLabel(self.stage_tab)
        self.tiltFlatToElectronLabel.setObjectName("tiltFlatToElectronLabel")
        self.gridLayout_2.addWidget(self.tiltFlatToElectronLabel, 3, 0, 1, 1)
        self.rotationFlatToIonSpinBox = QtWidgets.QDoubleSpinBox(self.stage_tab)
        self.rotationFlatToIonSpinBox.setMinimum(-360.0)
        self.rotationFlatToIonSpinBox.setMaximum(360.0)
        self.rotationFlatToIonSpinBox.setObjectName("rotationFlatToIonSpinBox")
        self.gridLayout_2.addWidget(self.rotationFlatToIonSpinBox, 2, 1, 1, 1)
        self.rotationFlatToElectronSpinBox = QtWidgets.QDoubleSpinBox(self.stage_tab)
        self.rotationFlatToElectronSpinBox.setMinimum(-360.0)
        self.rotationFlatToElectronSpinBox.setMaximum(360.0)
        self.rotationFlatToElectronSpinBox.setObjectName("rotationFlatToElectronSpinBox")
        self.gridLayout_2.addWidget(self.rotationFlatToElectronSpinBox, 1, 1, 1, 1)
        self.tiltFlatToIonLabel = QtWidgets.QLabel(self.stage_tab)
        self.tiltFlatToIonLabel.setObjectName("tiltFlatToIonLabel")
        self.gridLayout_2.addWidget(self.tiltFlatToIonLabel, 4, 0, 1, 1)
        self.tiltFlatToIonSpinBox = QtWidgets.QDoubleSpinBox(self.stage_tab)
        self.tiltFlatToIonSpinBox.setMinimum(-360.0)
        self.tiltFlatToIonSpinBox.setMaximum(360.0)
        self.tiltFlatToIonSpinBox.setObjectName("tiltFlatToIonSpinBox")
        self.gridLayout_2.addWidget(self.tiltFlatToIonSpinBox, 4, 1, 1, 1)
        self.preTiltLabel = QtWidgets.QLabel(self.stage_tab)
        self.preTiltLabel.setObjectName("preTiltLabel")
        self.gridLayout_2.addWidget(self.preTiltLabel, 5, 0, 1, 1)
        self.preTiltSpinBox = QtWidgets.QDoubleSpinBox(self.stage_tab)
        self.preTiltSpinBox.setObjectName("preTiltSpinBox")
        self.gridLayout_2.addWidget(self.preTiltSpinBox, 5, 1, 1, 1)
        self.setStage_button = QtWidgets.QPushButton(self.stage_tab)
        self.setStage_button.setObjectName("setStage_button")
        self.gridLayout_2.addWidget(self.setStage_button, 0, 1, 1, 1)
        self.rotationFlatToElectronLabel = QtWidgets.QLabel(self.stage_tab)
        self.rotationFlatToElectronLabel.setObjectName("rotationFlatToElectronLabel")
        self.gridLayout_2.addWidget(self.rotationFlatToElectronLabel, 1, 0, 1, 1)
        self.tiltFlatToElectronSpinBox = QtWidgets.QDoubleSpinBox(self.stage_tab)
        self.tiltFlatToElectronSpinBox.setMinimum(-360.0)
        self.tiltFlatToElectronSpinBox.setMaximum(360.0)
        self.tiltFlatToElectronSpinBox.setObjectName("tiltFlatToElectronSpinBox")
        self.gridLayout_2.addWidget(self.tiltFlatToElectronSpinBox, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 7, 0, 1, 1)
        self.tabWidget.addTab(self.stage_tab, "")
        self.tab_microscope = QtWidgets.QWidget()
        self.tab_microscope.setObjectName("tab_microscope")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_microscope)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox_stage_tilt = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_stage_tilt.setChecked(True)
        self.checkBox_stage_tilt.setObjectName("checkBox_stage_tilt")
        self.gridLayout_3.addWidget(self.checkBox_stage_tilt, 7, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_microscope)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setUnderline(True)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 9, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.tab_microscope)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_3.addWidget(self.line_2, 4, 0, 1, 2)
        self.checkBox_stage_enabled = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_stage_enabled.setChecked(True)
        self.checkBox_stage_enabled.setObjectName("checkBox_stage_enabled")
        self.gridLayout_3.addWidget(self.checkBox_stage_enabled, 5, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab_microscope)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setUnderline(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 3, 0, 1, 1)
        self.checkBox_stage_rotation = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_stage_rotation.setChecked(True)
        self.checkBox_stage_rotation.setObjectName("checkBox_stage_rotation")
        self.gridLayout_3.addWidget(self.checkBox_stage_rotation, 6, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem1, 5, 2, 1, 1)
        self.checkBox_ib = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_ib.setChecked(True)
        self.checkBox_ib.setObjectName("checkBox_ib")
        self.gridLayout_3.addWidget(self.checkBox_ib, 3, 1, 1, 1)
        self.checkBox_eb = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_eb.setChecked(True)
        self.checkBox_eb.setObjectName("checkBox_eb")
        self.gridLayout_3.addWidget(self.checkBox_eb, 1, 1, 1, 1)
        self.checkBox_needle_rotation = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_needle_rotation.setChecked(True)
        self.checkBox_needle_rotation.setObjectName("checkBox_needle_rotation")
        self.gridLayout_3.addWidget(self.checkBox_needle_rotation, 10, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.tab_microscope)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.tab_microscope)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_3.addWidget(self.line_3, 8, 0, 1, 2)
        self.line_4 = QtWidgets.QFrame(self.tab_microscope)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout_3.addWidget(self.line_4, 12, 0, 1, 2)
        self.line = QtWidgets.QFrame(self.tab_microscope)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_3.addWidget(self.line, 2, 0, 1, 2)
        self.checkBox_gis_enabled = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_gis_enabled.setChecked(True)
        self.checkBox_gis_enabled.setObjectName("checkBox_gis_enabled")
        self.gridLayout_3.addWidget(self.checkBox_gis_enabled, 13, 1, 1, 1)
        self.checkBox_needle_tilt = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_needle_tilt.setChecked(True)
        self.checkBox_needle_tilt.setObjectName("checkBox_needle_tilt")
        self.gridLayout_3.addWidget(self.checkBox_needle_tilt, 11, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tab_microscope)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setUnderline(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 5, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab_microscope)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setUnderline(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.tab_microscope)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setUnderline(True)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 13, 0, 1, 1)
        self.checkBox_needle_enabled = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_needle_enabled.setChecked(True)
        self.checkBox_needle_enabled.setObjectName("checkBox_needle_enabled")
        self.gridLayout_3.addWidget(self.checkBox_needle_enabled, 9, 1, 1, 1)
        self.checkBox_multichem = QtWidgets.QCheckBox(self.tab_microscope)
        self.checkBox_multichem.setChecked(True)
        self.checkBox_multichem.setObjectName("checkBox_multichem")
        self.gridLayout_3.addWidget(self.checkBox_multichem, 14, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem2, 15, 1, 1, 1)
        self.pushButton_save_model = QtWidgets.QPushButton(self.tab_microscope)
        self.pushButton_save_model.setObjectName("pushButton_save_model")
        self.gridLayout_3.addWidget(self.pushButton_save_model, 0, 2, 1, 1)
        self.tabWidget.addTab(self.tab_microscope, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tabWidget_2.setFont(font)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.spinBox_ion_voltage = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_ion_voltage.setMaximum(100000)
        self.spinBox_ion_voltage.setObjectName("spinBox_ion_voltage")
        self.gridLayout_5.addWidget(self.spinBox_ion_voltage, 1, 1, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.tab_2)
        self.label_21.setObjectName("label_21")
        self.gridLayout_5.addWidget(self.label_21, 14, 0, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.tab_2)
        self.label_23.setObjectName("label_23")
        self.gridLayout_5.addWidget(self.label_23, 16, 0, 1, 1)
        self.lineEdit_detector_type_eb = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_detector_type_eb.setObjectName("lineEdit_detector_type_eb")
        self.gridLayout_5.addWidget(self.lineEdit_detector_type_eb, 11, 1, 1, 1)
        self.spinBox_voltage_eb = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_voltage_eb.setMaximum(1000000)
        self.spinBox_voltage_eb.setObjectName("spinBox_voltage_eb")
        self.gridLayout_5.addWidget(self.spinBox_voltage_eb, 8, 1, 1, 1)
        self.lineEdit_detector_mode_eb = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_detector_mode_eb.setObjectName("lineEdit_detector_mode_eb")
        self.gridLayout_5.addWidget(self.lineEdit_detector_mode_eb, 12, 1, 1, 1)
        self.doubleSpinBox_height_ion = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_height_ion.setDecimals(4)
        self.doubleSpinBox_height_ion.setObjectName("doubleSpinBox_height_ion")
        self.gridLayout_5.addWidget(self.doubleSpinBox_height_ion, 4, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.tab_2)
        self.label_16.setObjectName("label_16")
        self.gridLayout_5.addWidget(self.label_16, 9, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(True)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_5.addWidget(self.label_20, 13, 0, 1, 2)
        self.doubleSpinBox_height_eb = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_height_eb.setDecimals(4)
        self.doubleSpinBox_height_eb.setObjectName("doubleSpinBox_height_eb")
        self.gridLayout_5.addWidget(self.doubleSpinBox_height_eb, 10, 1, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.tab_2)
        self.label_19.setObjectName("label_19")
        self.gridLayout_5.addWidget(self.label_19, 12, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.tab_2)
        self.label_25.setObjectName("label_25")
        self.gridLayout_5.addWidget(self.label_25, 18, 0, 1, 1)
        self.lineEdit_plasma_gas = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_plasma_gas.setObjectName("lineEdit_plasma_gas")
        self.gridLayout_5.addWidget(self.lineEdit_plasma_gas, 3, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(True)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_5.addWidget(self.label_7, 0, 0, 1, 2)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem3, 20, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(True)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_5.addWidget(self.label_14, 7, 0, 1, 2)
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_5.addWidget(self.label_12, 5, 0, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.tab_2)
        self.label_24.setObjectName("label_24")
        self.gridLayout_5.addWidget(self.label_24, 17, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setObjectName("label_17")
        self.gridLayout_5.addWidget(self.label_17, 10, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.tab_2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_5.addWidget(self.label_15, 8, 0, 1, 1)
        self.lineEdit_detector_type_ion = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_detector_type_ion.setObjectName("lineEdit_detector_type_ion")
        self.gridLayout_5.addWidget(self.lineEdit_detector_type_ion, 5, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.tab_2)
        self.label_18.setObjectName("label_18")
        self.gridLayout_5.addWidget(self.label_18, 11, 0, 1, 1)
        self.doubleSpinBox_current_eb = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_current_eb.setDecimals(4)
        self.doubleSpinBox_current_eb.setMaximum(100000.0)
        self.doubleSpinBox_current_eb.setObjectName("doubleSpinBox_current_eb")
        self.gridLayout_5.addWidget(self.doubleSpinBox_current_eb, 9, 1, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.tab_2)
        self.label_22.setObjectName("label_22")
        self.gridLayout_5.addWidget(self.label_22, 15, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_5.addWidget(self.label_11, 4, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_5.addWidget(self.label_13, 6, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.tab_2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_5.addWidget(self.label_10, 3, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_5.addWidget(self.label_9, 2, 0, 1, 1)
        self.doubleSpinBox_ion_current = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_ion_current.setDecimals(4)
        self.doubleSpinBox_ion_current.setMaximum(1000000.0)
        self.doubleSpinBox_ion_current.setObjectName("doubleSpinBox_ion_current")
        self.gridLayout_5.addWidget(self.doubleSpinBox_ion_current, 2, 1, 1, 1)
        self.lineEdit_detector_mode_ion = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_detector_mode_ion.setObjectName("lineEdit_detector_mode_ion")
        self.gridLayout_5.addWidget(self.lineEdit_detector_mode_ion, 6, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_5.addWidget(self.label_8, 1, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.tab_2)
        self.label_26.setObjectName("label_26")
        self.gridLayout_5.addWidget(self.label_26, 19, 0, 1, 1)
        self.spinBox_rotation_eb = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_rotation_eb.setMaximum(360)
        self.spinBox_rotation_eb.setObjectName("spinBox_rotation_eb")
        self.gridLayout_5.addWidget(self.spinBox_rotation_eb, 14, 1, 1, 1)
        self.spinBox_rotation_ib = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_rotation_ib.setMaximum(360)
        self.spinBox_rotation_ib.setObjectName("spinBox_rotation_ib")
        self.gridLayout_5.addWidget(self.spinBox_rotation_ib, 15, 1, 1, 1)
        self.spinBox_tilt_eb = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_tilt_eb.setMaximum(360)
        self.spinBox_tilt_eb.setObjectName("spinBox_tilt_eb")
        self.gridLayout_5.addWidget(self.spinBox_tilt_eb, 16, 1, 1, 1)
        self.spinBox_tilt_ib = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_tilt_ib.setMaximum(360)
        self.spinBox_tilt_ib.setObjectName("spinBox_tilt_ib")
        self.gridLayout_5.addWidget(self.spinBox_tilt_ib, 17, 1, 1, 1)
        self.spinBox_pretilt = QtWidgets.QSpinBox(self.tab_2)
        self.spinBox_pretilt.setMaximum(360)
        self.spinBox_pretilt.setObjectName("spinBox_pretilt")
        self.gridLayout_5.addWidget(self.spinBox_pretilt, 18, 1, 1, 1)
        self.doubleSpinBox_needle_height = QtWidgets.QDoubleSpinBox(self.tab_2)
        self.doubleSpinBox_needle_height.setDecimals(4)
        self.doubleSpinBox_needle_height.setObjectName("doubleSpinBox_needle_height")
        self.gridLayout_5.addWidget(self.doubleSpinBox_needle_height, 19, 1, 1, 1)
        self.tabWidget_2.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_33 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_33.setFont(font)
        self.label_33.setObjectName("label_33")
        self.gridLayout_6.addWidget(self.label_33, 6, 0, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.gridLayout_6.addWidget(self.label_30, 3, 0, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_38.setFont(font)
        self.label_38.setObjectName("label_38")
        self.gridLayout_6.addWidget(self.label_38, 11, 0, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_37.setFont(font)
        self.label_37.setObjectName("label_37")
        self.gridLayout_6.addWidget(self.label_37, 10, 0, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_34.setFont(font)
        self.label_34.setObjectName("label_34")
        self.gridLayout_6.addWidget(self.label_34, 7, 0, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.gridLayout_6.addWidget(self.label_29, 2, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem4, 14, 0, 1, 2)
        self.label_39 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_39.setFont(font)
        self.label_39.setObjectName("label_39")
        self.gridLayout_6.addWidget(self.label_39, 12, 0, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(True)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.gridLayout_6.addWidget(self.label_27, 0, 0, 1, 1)
        self.checkBox_gamma = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_gamma.setObjectName("checkBox_gamma")
        self.gridLayout_6.addWidget(self.checkBox_gamma, 13, 1, 1, 1)
        self.checkBox_autocontrast = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_autocontrast.setObjectName("checkBox_autocontrast")
        self.gridLayout_6.addWidget(self.checkBox_autocontrast, 10, 1, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_35.setFont(font)
        self.label_35.setObjectName("label_35")
        self.gridLayout_6.addWidget(self.label_35, 8, 0, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_36.setFont(font)
        self.label_36.setObjectName("label_36")
        self.gridLayout_6.addWidget(self.label_36, 9, 0, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_40.setFont(font)
        self.label_40.setObjectName("label_40")
        self.gridLayout_6.addWidget(self.label_40, 13, 0, 1, 1)
        self.checkBox_save = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_save.setObjectName("checkBox_save")
        self.gridLayout_6.addWidget(self.checkBox_save, 12, 1, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setUnderline(True)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.gridLayout_6.addWidget(self.label_32, 5, 0, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.gridLayout_6.addWidget(self.label_31, 4, 0, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.gridLayout_6.addWidget(self.label_28, 1, 0, 1, 1)
        self.spinBox_res_width = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_res_width.setMaximum(100000)
        self.spinBox_res_width.setObjectName("spinBox_res_width")
        self.gridLayout_6.addWidget(self.spinBox_res_width, 7, 1, 1, 1)
        self.spinBox_res_height = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_res_height.setMaximum(100000)
        self.spinBox_res_height.setObjectName("spinBox_res_height")
        self.gridLayout_6.addWidget(self.spinBox_res_height, 7, 2, 1, 1)
        self.doubleSpinBox_imaging_current = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBox_imaging_current.setDecimals(4)
        self.doubleSpinBox_imaging_current.setMaximum(1000000.0)
        self.doubleSpinBox_imaging_current.setObjectName("doubleSpinBox_imaging_current")
        self.gridLayout_6.addWidget(self.doubleSpinBox_imaging_current, 6, 1, 1, 2)
        self.spinBox_hfw = QtWidgets.QSpinBox(self.tab_3)
        self.spinBox_hfw.setMaximum(1000000)
        self.spinBox_hfw.setObjectName("spinBox_hfw")
        self.gridLayout_6.addWidget(self.spinBox_hfw, 8, 1, 1, 2)
        self.lineEdit_beam_type = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_beam_type.setObjectName("lineEdit_beam_type")
        self.gridLayout_6.addWidget(self.lineEdit_beam_type, 9, 1, 1, 2)
        self.doubleSpinBox_dwell_time_imaging = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBox_dwell_time_imaging.setDecimals(4)
        self.doubleSpinBox_dwell_time_imaging.setMaximum(1000000.0)
        self.doubleSpinBox_dwell_time_imaging.setObjectName("doubleSpinBox_dwell_time_imaging")
        self.gridLayout_6.addWidget(self.doubleSpinBox_dwell_time_imaging, 11, 1, 1, 2)
        self.doubleSpinBox_dwell_time_milling = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBox_dwell_time_milling.setDecimals(4)
        self.doubleSpinBox_dwell_time_milling.setMaximum(100000.0)
        self.doubleSpinBox_dwell_time_milling.setObjectName("doubleSpinBox_dwell_time_milling")
        self.gridLayout_6.addWidget(self.doubleSpinBox_dwell_time_milling, 4, 1, 1, 2)
        self.doubleSpinBox_rate = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBox_rate.setDecimals(4)
        self.doubleSpinBox_rate.setObjectName("doubleSpinBox_rate")
        self.gridLayout_6.addWidget(self.doubleSpinBox_rate, 3, 1, 1, 2)
        self.doubleSpinBox_spotsize = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBox_spotsize.setDecimals(4)
        self.doubleSpinBox_spotsize.setMaximum(100000.0)
        self.doubleSpinBox_spotsize.setObjectName("doubleSpinBox_spotsize")
        self.gridLayout_6.addWidget(self.doubleSpinBox_spotsize, 2, 1, 1, 2)
        self.doubleSpinBox_milling_current = QtWidgets.QDoubleSpinBox(self.tab_3)
        self.doubleSpinBox_milling_current.setDecimals(4)
        self.doubleSpinBox_milling_current.setMaximum(100000.0)
        self.doubleSpinBox_milling_current.setObjectName("doubleSpinBox_milling_current")
        self.gridLayout_6.addWidget(self.doubleSpinBox_milling_current, 1, 1, 1, 2)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.gridLayout_4.addWidget(self.tabWidget_2, 1, 0, 1, 1)
        self.pushButton_save_defaults = QtWidgets.QPushButton(self.tab)
        self.pushButton_save_defaults.setObjectName("pushButton_save_defaults")
        self.gridLayout_4.addWidget(self.pushButton_save_defaults, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.gridLayout.addWidget(self.tabWidget, 13, 0, 1, 2)
        self.label_ip_address = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_ip_address.setFont(font)
        self.label_ip_address.setObjectName("label_ip_address")
        self.gridLayout.addWidget(self.label_ip_address, 2, 0, 1, 1)
        self.label_manufacturer = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_manufacturer.setFont(font)
        self.label_manufacturer.setObjectName("label_manufacturer")
        self.gridLayout.addWidget(self.label_manufacturer, 3, 0, 1, 1)
        self.microscope_button = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.microscope_button.setFont(font)
        self.microscope_button.setObjectName("microscope_button")
        self.gridLayout.addWidget(self.microscope_button, 4, 0, 1, 2)
        self.label_title = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.rotationFlatToIonLabel.setText(_translate("Form", "Rotation Flat To Ion (deg)"))
        self.label_header_stage.setText(_translate("Form", "Stage "))
        self.needleStageHeightLimitnMmLabel.setText(_translate("Form", "Needle Stage Height Limit (mm)"))
        self.tiltFlatToElectronLabel.setText(_translate("Form", "Tilt Flat To Electron (deg)"))
        self.tiltFlatToIonLabel.setText(_translate("Form", "Tilt Flat To Ion (deg)"))
        self.preTiltLabel.setText(_translate("Form", "Shuttle Pre Tilt (deg)"))
        self.setStage_button.setText(_translate("Form", "Set Stage Parameters"))
        self.rotationFlatToElectronLabel.setText(_translate("Form", "Rotation Flat To Electron (deg)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.stage_tab), _translate("Form", "Stage"))
        self.checkBox_stage_tilt.setText(_translate("Form", "Tilt"))
        self.label_5.setText(_translate("Form", "Manipulator"))
        self.checkBox_stage_enabled.setText(_translate("Form", "Enabled"))
        self.label_3.setText(_translate("Form", "Ion Beam"))
        self.checkBox_stage_rotation.setText(_translate("Form", "Rotation"))
        self.checkBox_ib.setText(_translate("Form", "Enabled"))
        self.checkBox_eb.setText(_translate("Form", "Enabled"))
        self.checkBox_needle_rotation.setText(_translate("Form", "Rotation"))
        self.label.setText(_translate("Form", "Model parameters"))
        self.checkBox_gis_enabled.setText(_translate("Form", "Enabled"))
        self.checkBox_needle_tilt.setText(_translate("Form", "Tilt"))
        self.label_4.setText(_translate("Form", "Stage"))
        self.label_2.setText(_translate("Form", "Electron Beam"))
        self.label_6.setText(_translate("Form", "GIS"))
        self.checkBox_needle_enabled.setText(_translate("Form", "Enabled"))
        self.checkBox_multichem.setText(_translate("Form", "MultiChem"))
        self.pushButton_save_model.setText(_translate("Form", "Save Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_microscope), _translate("Form", "Microscope Model"))
        self.label_21.setText(_translate("Form", "Rotation Flat To Electron (°)"))
        self.label_23.setText(_translate("Form", "Tilt Flat To Electron (°)"))
        self.label_16.setText(_translate("Form", "Current (nA)"))
        self.label_20.setText(_translate("Form", "Stage"))
        self.label_19.setText(_translate("Form", "Detector Mode"))
        self.label_25.setText(_translate("Form", "Shuttle  Pretilt (°)"))
        self.label_7.setText(_translate("Form", "Ion Beam"))
        self.label_14.setText(_translate("Form", "Electron Beam"))
        self.label_12.setText(_translate("Form", "Detector Type"))
        self.label_24.setText(_translate("Form", "Tilt Flat To Ion (°)"))
        self.label_17.setText(_translate("Form", "Eucentric Height  (m)"))
        self.label_15.setText(_translate("Form", "Voltage (V)"))
        self.label_18.setText(_translate("Form", "Detector Type"))
        self.label_22.setText(_translate("Form", "Rotation Flat To Ion (°)"))
        self.label_11.setText(_translate("Form", "Eucentric Height (m)"))
        self.label_13.setText(_translate("Form", "Detector Mode"))
        self.label_10.setText(_translate("Form", "Plasma Gas"))
        self.label_9.setText(_translate("Form", "Current (nA)"))
        self.label_8.setText(_translate("Form", "Voltage (V)"))
        self.label_26.setText(_translate("Form", "Needle Stage Hieght Limit (m)"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_2), _translate("Form", "System"))
        self.label_33.setText(_translate("Form", "Imaging Current (nA)"))
        self.label_30.setText(_translate("Form", "Rate (mm3/A/s)"))
        self.label_38.setText(_translate("Form", "Dwell Time (µs)"))
        self.label_37.setText(_translate("Form", "Autocontrast"))
        self.label_34.setText(_translate("Form", "Resolution"))
        self.label_29.setText(_translate("Form", "Spot Size (nm)"))
        self.label_39.setText(_translate("Form", "Save"))
        self.label_27.setText(_translate("Form", "Milling"))
        self.checkBox_gamma.setText(_translate("Form", "Enabled"))
        self.checkBox_autocontrast.setText(_translate("Form", "Enabled"))
        self.label_35.setText(_translate("Form", "Horizontal Field Width (µm)"))
        self.label_36.setText(_translate("Form", "Beam"))
        self.label_40.setText(_translate("Form", "Gamma "))
        self.checkBox_save.setText(_translate("Form", "Enabled"))
        self.label_32.setText(_translate("Form", "imaging"))
        self.label_31.setText(_translate("Form", "Dwell time (µs)"))
        self.label_28.setText(_translate("Form", "Milling Current (nA)"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("Form", "User"))
        self.pushButton_save_defaults.setText(_translate("Form", "Save Defaults To Yaml"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Defaults"))
        self.label_ip_address.setText(_translate("Form", "IP address"))
        self.label_manufacturer.setText(_translate("Form", "Manufacturer "))
        self.microscope_button.setText(_translate("Form", "Connect to Microscope"))
        self.label_title.setText(_translate("Form", "System"))
