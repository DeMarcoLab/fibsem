# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemMillingWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(722, 837)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_run_milling = QtWidgets.QPushButton(Form)
        self.pushButton_run_milling.setObjectName("pushButton_run_milling")
        self.gridLayout.addWidget(self.pushButton_run_milling, 26, 0, 1, 2)
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setMinimumSize(QtCore.QSize(0, 0))
        self.frame.setObjectName("frame")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.checkBox_relative_move = QtWidgets.QCheckBox(self.frame)
        self.checkBox_relative_move.setChecked(True)
        self.checkBox_relative_move.setObjectName("checkBox_relative_move")
        self.gridLayout_4.addWidget(self.checkBox_relative_move, 25, 1, 1, 1)
        self.label_pattern_set = QtWidgets.QLabel(self.frame)
        self.label_pattern_set.setObjectName("label_pattern_set")
        self.gridLayout_4.addWidget(self.label_pattern_set, 17, 0, 1, 1)
        self.doubleSpinBox_hfw = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_hfw.setEnabled(True)
        self.doubleSpinBox_hfw.setReadOnly(True)
        self.doubleSpinBox_hfw.setMinimum(10.0)
        self.doubleSpinBox_hfw.setMaximum(1000000000.0)
        self.doubleSpinBox_hfw.setProperty("value", 150.0)
        self.doubleSpinBox_hfw.setObjectName("doubleSpinBox_hfw")
        self.gridLayout_4.addWidget(self.doubleSpinBox_hfw, 7, 1, 1, 1)
        self.doubleSpinBox_spacing = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_spacing.setProperty("value", 1.0)
        self.doubleSpinBox_spacing.setObjectName("doubleSpinBox_spacing")
        self.gridLayout_4.addWidget(self.doubleSpinBox_spacing, 12, 1, 1, 1)
        self.label_patterns_header = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_patterns_header.setFont(font)
        self.label_patterns_header.setObjectName("label_patterns_header")
        self.gridLayout_4.addWidget(self.label_patterns_header, 15, 0, 1, 2)
        self.doubleSpinBox_dwell_time = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_dwell_time.setDecimals(4)
        self.doubleSpinBox_dwell_time.setMinimum(0.0)
        self.doubleSpinBox_dwell_time.setMaximum(4000000.0)
        self.doubleSpinBox_dwell_time.setSingleStep(0.01)
        self.doubleSpinBox_dwell_time.setProperty("value", 0.0)
        self.doubleSpinBox_dwell_time.setObjectName("doubleSpinBox_dwell_time")
        self.gridLayout_4.addWidget(self.doubleSpinBox_dwell_time, 10, 1, 1, 1)
        self.comboBox_application_file = QtWidgets.QComboBox(self.frame)
        self.comboBox_application_file.setObjectName("comboBox_application_file")
        self.gridLayout_4.addWidget(self.comboBox_application_file, 8, 1, 1, 1)
        self.comboBox_preset = QtWidgets.QComboBox(self.frame)
        self.comboBox_preset.setObjectName("comboBox_preset")
        self.gridLayout_4.addWidget(self.comboBox_preset, 13, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_4.addWidget(self.pushButton, 27, 0, 1, 2)
        self.pushButton_remove_milling_stage = QtWidgets.QPushButton(self.frame)
        self.pushButton_remove_milling_stage.setObjectName("pushButton_remove_milling_stage")
        self.gridLayout_4.addWidget(self.pushButton_remove_milling_stage, 3, 1, 1, 1)
        self.label_milling_stage = QtWidgets.QLabel(self.frame)
        self.label_milling_stage.setObjectName("label_milling_stage")
        self.gridLayout_4.addWidget(self.label_milling_stage, 2, 0, 1, 1)
        self.pushButton_add_milling_stage = QtWidgets.QPushButton(self.frame)
        self.pushButton_add_milling_stage.setObjectName("pushButton_add_milling_stage")
        self.gridLayout_4.addWidget(self.pushButton_add_milling_stage, 3, 0, 1, 1)
        self.label_spacing = QtWidgets.QLabel(self.frame)
        self.label_spacing.setObjectName("label_spacing")
        self.gridLayout_4.addWidget(self.label_spacing, 12, 0, 1, 1)
        self.label_hfw = QtWidgets.QLabel(self.frame)
        self.label_hfw.setObjectName("label_hfw")
        self.gridLayout_4.addWidget(self.label_hfw, 7, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_4.addWidget(self.line_2, 14, 0, 1, 2)
        self.label_preset = QtWidgets.QLabel(self.frame)
        self.label_preset.setObjectName("label_preset")
        self.gridLayout_4.addWidget(self.label_preset, 13, 0, 1, 1)
        self.label_dwell_time = QtWidgets.QLabel(self.frame)
        self.label_dwell_time.setObjectName("label_dwell_time")
        self.gridLayout_4.addWidget(self.label_dwell_time, 10, 0, 1, 1)
        self.doubleSpinBox_centre_y = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_centre_y.setMinimum(-1e+16)
        self.doubleSpinBox_centre_y.setMaximum(1e+23)
        self.doubleSpinBox_centre_y.setSingleStep(0.1)
        self.doubleSpinBox_centre_y.setObjectName("doubleSpinBox_centre_y")
        self.gridLayout_4.addWidget(self.doubleSpinBox_centre_y, 20, 1, 1, 1)
        self.comboBox_patterns = QtWidgets.QComboBox(self.frame)
        self.comboBox_patterns.setObjectName("comboBox_patterns")
        self.gridLayout_4.addWidget(self.comboBox_patterns, 17, 1, 1, 1)
        self.label_application_file = QtWidgets.QLabel(self.frame)
        self.label_application_file.setObjectName("label_application_file")
        self.gridLayout_4.addWidget(self.label_application_file, 8, 0, 1, 1)
        self.label_rate = QtWidgets.QLabel(self.frame)
        self.label_rate.setObjectName("label_rate")
        self.gridLayout_4.addWidget(self.label_rate, 9, 0, 1, 1)
        self.doubleSpinBox_spot_size = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_spot_size.setDecimals(4)
        self.doubleSpinBox_spot_size.setMinimum(0.0)
        self.doubleSpinBox_spot_size.setMaximum(100000.0)
        self.doubleSpinBox_spot_size.setSingleStep(0.01)
        self.doubleSpinBox_spot_size.setProperty("value", 0.0)
        self.doubleSpinBox_spot_size.setObjectName("doubleSpinBox_spot_size")
        self.gridLayout_4.addWidget(self.doubleSpinBox_spot_size, 11, 1, 1, 1)
        self.label_spot_size = QtWidgets.QLabel(self.frame)
        self.label_spot_size.setObjectName("label_spot_size")
        self.gridLayout_4.addWidget(self.label_spot_size, 11, 0, 1, 1)
        self.label_centre_y = QtWidgets.QLabel(self.frame)
        self.label_centre_y.setObjectName("label_centre_y")
        self.gridLayout_4.addWidget(self.label_centre_y, 20, 0, 1, 1)
        self.label_milling_current = QtWidgets.QLabel(self.frame)
        self.label_milling_current.setObjectName("label_milling_current")
        self.gridLayout_4.addWidget(self.label_milling_current, 6, 0, 1, 1)
        self.gridLayout_patterns = QtWidgets.QGridLayout()
        self.gridLayout_patterns.setObjectName("gridLayout_patterns")
        self.gridLayout_4.addLayout(self.gridLayout_patterns, 22, 0, 2, 2)
        self.line_3 = QtWidgets.QFrame(self.frame)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_4.addWidget(self.line_3, 4, 0, 1, 2)
        self.doubleSpinBox_rate = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_rate.setDecimals(4)
        self.doubleSpinBox_rate.setMinimum(0.0)
        self.doubleSpinBox_rate.setMaximum(100000.0)
        self.doubleSpinBox_rate.setSingleStep(0.01)
        self.doubleSpinBox_rate.setProperty("value", 0.0)
        self.doubleSpinBox_rate.setObjectName("doubleSpinBox_rate")
        self.gridLayout_4.addWidget(self.doubleSpinBox_rate, 9, 1, 1, 1)
        self.label_centre_x = QtWidgets.QLabel(self.frame)
        self.label_centre_x.setObjectName("label_centre_x")
        self.gridLayout_4.addWidget(self.label_centre_x, 19, 0, 1, 1)
        self.doubleSpinBox_centre_x = QtWidgets.QDoubleSpinBox(self.frame)
        self.doubleSpinBox_centre_x.setMinimum(-1e+28)
        self.doubleSpinBox_centre_x.setMaximum(1e+18)
        self.doubleSpinBox_centre_x.setSingleStep(0.1)
        self.doubleSpinBox_centre_x.setObjectName("doubleSpinBox_centre_x")
        self.gridLayout_4.addWidget(self.doubleSpinBox_centre_x, 19, 1, 1, 1)
        self.checkBox_live_update = QtWidgets.QCheckBox(self.frame)
        self.checkBox_live_update.setObjectName("checkBox_live_update")
        self.gridLayout_4.addWidget(self.checkBox_live_update, 25, 0, 1, 1)
        self.comboBox_milling_current = QtWidgets.QComboBox(self.frame)
        self.comboBox_milling_current.setObjectName("comboBox_milling_current")
        self.gridLayout_4.addWidget(self.comboBox_milling_current, 6, 1, 1, 1)
        self.label_milling_header = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_milling_header.setFont(font)
        self.label_milling_header.setObjectName("label_milling_header")
        self.gridLayout_4.addWidget(self.label_milling_header, 5, 0, 1, 2)
        self.comboBox_milling_stage = QtWidgets.QComboBox(self.frame)
        self.comboBox_milling_stage.setObjectName("comboBox_milling_stage")
        self.gridLayout_4.addWidget(self.comboBox_milling_stage, 2, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 24, 0, 1, 2)
        self.pushButton_importMilling = QtWidgets.QPushButton(self.frame)
        self.pushButton_importMilling.setObjectName("pushButton_importMilling")
        self.gridLayout_4.addWidget(self.pushButton_importMilling, 1, 0, 1, 1)
        self.pushButton_exportMilling = QtWidgets.QPushButton(self.frame)
        self.pushButton_exportMilling.setObjectName("pushButton_exportMilling")
        self.gridLayout_4.addWidget(self.pushButton_exportMilling, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.frame, 16, 0, 1, 2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 29, 0, 1, 2)
        self.label_info = QtWidgets.QLabel(Form)
        self.label_info.setText("")
        self.label_info.setWordWrap(True)
        self.label_info.setObjectName("label_info")
        self.gridLayout.addWidget(self.label_info, 28, 0, 1, 2)
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 18, 0, 1, 2)
        self.progressBar_milling = QtWidgets.QProgressBar(Form)
        self.progressBar_milling.setProperty("value", 24)
        self.progressBar_milling.setObjectName("progressBar_milling")
        self.gridLayout.addWidget(self.progressBar_milling, 27, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.frame, self.comboBox_milling_stage)
        Form.setTabOrder(self.comboBox_milling_stage, self.pushButton_add_milling_stage)
        Form.setTabOrder(self.pushButton_add_milling_stage, self.pushButton_remove_milling_stage)
        Form.setTabOrder(self.pushButton_remove_milling_stage, self.checkBox_live_update)
        Form.setTabOrder(self.checkBox_live_update, self.pushButton)
        Form.setTabOrder(self.pushButton, self.pushButton_run_milling)
        Form.setTabOrder(self.pushButton_run_milling, self.comboBox_milling_current)
        Form.setTabOrder(self.comboBox_milling_current, self.doubleSpinBox_hfw)
        Form.setTabOrder(self.doubleSpinBox_hfw, self.comboBox_application_file)
        Form.setTabOrder(self.comboBox_application_file, self.doubleSpinBox_rate)
        Form.setTabOrder(self.doubleSpinBox_rate, self.doubleSpinBox_dwell_time)
        Form.setTabOrder(self.doubleSpinBox_dwell_time, self.doubleSpinBox_spot_size)
        Form.setTabOrder(self.doubleSpinBox_spot_size, self.doubleSpinBox_spacing)
        Form.setTabOrder(self.doubleSpinBox_spacing, self.comboBox_preset)
        Form.setTabOrder(self.comboBox_preset, self.comboBox_patterns)
        Form.setTabOrder(self.comboBox_patterns, self.doubleSpinBox_centre_x)
        Form.setTabOrder(self.doubleSpinBox_centre_x, self.doubleSpinBox_centre_y)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_run_milling.setText(_translate("Form", "Run Milling"))
        self.checkBox_relative_move.setText(_translate("Form", "Keep Relative Orientation"))
        self.label_pattern_set.setText(_translate("Form", "Pattern"))
        self.label_patterns_header.setText(_translate("Form", "Patterns"))
        self.pushButton.setText(_translate("Form", "Update Pattern"))
        self.pushButton_remove_milling_stage.setText(_translate("Form", "Remove"))
        self.label_milling_stage.setText(_translate("Form", "Milling Stage"))
        self.pushButton_add_milling_stage.setText(_translate("Form", "Add"))
        self.label_spacing.setText(_translate("Form", "Spacing"))
        self.label_hfw.setText(_translate("Form", "Horizontal Field Width (um)"))
        self.label_preset.setText(_translate("Form", "Preset"))
        self.label_dwell_time.setText(_translate("Form", "Dwell Time (us)"))
        self.label_application_file.setText(_translate("Form", "Application File"))
        self.label_rate.setText(_translate("Form", "Rate (mm3/A/s)"))
        self.label_spot_size.setText(_translate("Form", "Spot Size (um)"))
        self.label_centre_y.setText(_translate("Form", "Centre Y (um)"))
        self.label_milling_current.setText(_translate("Form", "Current (A)"))
        self.label_centre_x.setText(_translate("Form", "Centre X (um)"))
        self.checkBox_live_update.setText(_translate("Form", "Live Update"))
        self.label_milling_header.setText(_translate("Form", "Milling"))
        self.pushButton_importMilling.setText(_translate("Form", "Import Milling Stages"))
        self.pushButton_exportMilling.setText(_translate("Form", "Export Milling Stages"))
