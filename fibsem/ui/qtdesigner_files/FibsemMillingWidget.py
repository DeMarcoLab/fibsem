# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemMillingWidgetui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(394, 837)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_patterns = QtWidgets.QWidget()
        self.tab_patterns.setObjectName("tab_patterns")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_patterns)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.comboBox_milling_stage = QtWidgets.QComboBox(self.tab_patterns)
        self.comboBox_milling_stage.setObjectName("comboBox_milling_stage")
        self.gridLayout_4.addWidget(self.comboBox_milling_stage, 1, 1, 1, 1)
        self.comboBox_milling_current = QtWidgets.QComboBox(self.tab_patterns)
        self.comboBox_milling_current.setObjectName("comboBox_milling_current")
        self.gridLayout_4.addWidget(self.comboBox_milling_current, 5, 1, 1, 1)
        self.label_milling_header = QtWidgets.QLabel(self.tab_patterns)
        self.label_milling_header.setObjectName("label_milling_header")
        self.gridLayout_4.addWidget(self.label_milling_header, 4, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 23, 0, 1, 2)
        self.label_preset = QtWidgets.QLabel(self.tab_patterns)
        self.label_preset.setObjectName("label_preset")
        self.gridLayout_4.addWidget(self.label_preset, 12, 0, 1, 1)
        self.doubleSpinBox_centre_y = QtWidgets.QDoubleSpinBox(self.tab_patterns)
        self.doubleSpinBox_centre_y.setMinimum(-1e+16)
        self.doubleSpinBox_centre_y.setMaximum(1e+23)
        self.doubleSpinBox_centre_y.setSingleStep(0.1)
        self.doubleSpinBox_centre_y.setObjectName("doubleSpinBox_centre_y")
        self.gridLayout_4.addWidget(self.doubleSpinBox_centre_y, 19, 1, 1, 1)
        self.label_spacing = QtWidgets.QLabel(self.tab_patterns)
        self.label_spacing.setObjectName("label_spacing")
        self.gridLayout_4.addWidget(self.label_spacing, 11, 0, 1, 1)
        self.label_hfw = QtWidgets.QLabel(self.tab_patterns)
        self.label_hfw.setObjectName("label_hfw")
        self.gridLayout_4.addWidget(self.label_hfw, 6, 0, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.tab_patterns)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_4.addWidget(self.line_2, 13, 0, 1, 2)
        self.doubleSpinBox_centre_x = QtWidgets.QDoubleSpinBox(self.tab_patterns)
        self.doubleSpinBox_centre_x.setMinimum(-1e+28)
        self.doubleSpinBox_centre_x.setMaximum(1e+18)
        self.doubleSpinBox_centre_x.setSingleStep(0.1)
        self.doubleSpinBox_centre_x.setObjectName("doubleSpinBox_centre_x")
        self.gridLayout_4.addWidget(self.doubleSpinBox_centre_x, 18, 1, 1, 1)
        self.label_rate = QtWidgets.QLabel(self.tab_patterns)
        self.label_rate.setObjectName("label_rate")
        self.gridLayout_4.addWidget(self.label_rate, 8, 0, 1, 1)
        self.checkBox_live_update = QtWidgets.QCheckBox(self.tab_patterns)
        self.checkBox_live_update.setObjectName("checkBox_live_update")
        self.gridLayout_4.addWidget(self.checkBox_live_update, 24, 0, 1, 1)
        self.label_milling_current = QtWidgets.QLabel(self.tab_patterns)
        self.label_milling_current.setObjectName("label_milling_current")
        self.gridLayout_4.addWidget(self.label_milling_current, 5, 0, 1, 1)
        self.label_dwell_time = QtWidgets.QLabel(self.tab_patterns)
        self.label_dwell_time.setObjectName("label_dwell_time")
        self.gridLayout_4.addWidget(self.label_dwell_time, 9, 0, 1, 1)
        self.pushButton_add_milling_stage = QtWidgets.QPushButton(self.tab_patterns)
        self.pushButton_add_milling_stage.setObjectName("pushButton_add_milling_stage")
        self.gridLayout_4.addWidget(self.pushButton_add_milling_stage, 2, 0, 1, 1)
        self.comboBox_patterns = QtWidgets.QComboBox(self.tab_patterns)
        self.comboBox_patterns.setObjectName("comboBox_patterns")
        self.gridLayout_4.addWidget(self.comboBox_patterns, 16, 1, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.tab_patterns)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_4.addWidget(self.line_3, 3, 0, 1, 2)
        self.label_centre_x = QtWidgets.QLabel(self.tab_patterns)
        self.label_centre_x.setObjectName("label_centre_x")
        self.gridLayout_4.addWidget(self.label_centre_x, 18, 0, 1, 1)
        self.label_application_file = QtWidgets.QLabel(self.tab_patterns)
        self.label_application_file.setObjectName("label_application_file")
        self.gridLayout_4.addWidget(self.label_application_file, 7, 0, 1, 1)
        self.doubleSpinBox_spot_size = QtWidgets.QDoubleSpinBox(self.tab_patterns)
        self.doubleSpinBox_spot_size.setMinimum(0.01)
        self.doubleSpinBox_spot_size.setSingleStep(0.01)
        self.doubleSpinBox_spot_size.setObjectName("doubleSpinBox_spot_size")
        self.gridLayout_4.addWidget(self.doubleSpinBox_spot_size, 10, 1, 1, 1)
        self.label_spot_size = QtWidgets.QLabel(self.tab_patterns)
        self.label_spot_size.setObjectName("label_spot_size")
        self.gridLayout_4.addWidget(self.label_spot_size, 10, 0, 1, 1)
        self.doubleSpinBox_rate = QtWidgets.QDoubleSpinBox(self.tab_patterns)
        self.doubleSpinBox_rate.setMinimum(0.01)
        self.doubleSpinBox_rate.setSingleStep(0.01)
        self.doubleSpinBox_rate.setObjectName("doubleSpinBox_rate")
        self.gridLayout_4.addWidget(self.doubleSpinBox_rate, 8, 1, 1, 1)
        self.label_centre_y = QtWidgets.QLabel(self.tab_patterns)
        self.label_centre_y.setObjectName("label_centre_y")
        self.gridLayout_4.addWidget(self.label_centre_y, 19, 0, 1, 1)
        self.label_milling_stage = QtWidgets.QLabel(self.tab_patterns)
        self.label_milling_stage.setObjectName("label_milling_stage")
        self.gridLayout_4.addWidget(self.label_milling_stage, 1, 0, 1, 1)
        self.gridLayout_patterns = QtWidgets.QGridLayout()
        self.gridLayout_patterns.setObjectName("gridLayout_patterns")
        self.gridLayout_4.addLayout(self.gridLayout_patterns, 21, 0, 2, 2)
        self.comboBox_preset = QtWidgets.QComboBox(self.tab_patterns)
        self.comboBox_preset.setObjectName("comboBox_preset")
        self.gridLayout_4.addWidget(self.comboBox_preset, 12, 1, 1, 1)
        self.pushButton_remove_milling_stage = QtWidgets.QPushButton(self.tab_patterns)
        self.pushButton_remove_milling_stage.setObjectName("pushButton_remove_milling_stage")
        self.gridLayout_4.addWidget(self.pushButton_remove_milling_stage, 2, 1, 1, 1)
        self.doubleSpinBox_spacing = QtWidgets.QDoubleSpinBox(self.tab_patterns)
        self.doubleSpinBox_spacing.setProperty("value", 1.0)
        self.doubleSpinBox_spacing.setObjectName("doubleSpinBox_spacing")
        self.gridLayout_4.addWidget(self.doubleSpinBox_spacing, 11, 1, 1, 1)
        self.doubleSpinBox_hfw = QtWidgets.QDoubleSpinBox(self.tab_patterns)
        self.doubleSpinBox_hfw.setEnabled(True)
        self.doubleSpinBox_hfw.setReadOnly(True)
        self.doubleSpinBox_hfw.setMinimum(10.0)
        self.doubleSpinBox_hfw.setMaximum(1000000000.0)
        self.doubleSpinBox_hfw.setProperty("value", 150.0)
        self.doubleSpinBox_hfw.setObjectName("doubleSpinBox_hfw")
        self.gridLayout_4.addWidget(self.doubleSpinBox_hfw, 6, 1, 1, 1)
        self.doubleSpinBox_dwell_time = QtWidgets.QDoubleSpinBox(self.tab_patterns)
        self.doubleSpinBox_dwell_time.setMinimum(0.01)
        self.doubleSpinBox_dwell_time.setSingleStep(0.01)
        self.doubleSpinBox_dwell_time.setObjectName("doubleSpinBox_dwell_time")
        self.gridLayout_4.addWidget(self.doubleSpinBox_dwell_time, 9, 1, 1, 1)
        self.comboBox_application_file = QtWidgets.QComboBox(self.tab_patterns)
        self.comboBox_application_file.setObjectName("comboBox_application_file")
        self.gridLayout_4.addWidget(self.comboBox_application_file, 7, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.tab_patterns)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_4.addWidget(self.pushButton, 26, 0, 1, 2)
        self.label_patterns_header = QtWidgets.QLabel(self.tab_patterns)
        self.label_patterns_header.setObjectName("label_patterns_header")
        self.gridLayout_4.addWidget(self.label_patterns_header, 14, 0, 1, 2)
        self.label_pattern_set = QtWidgets.QLabel(self.tab_patterns)
        self.label_pattern_set.setObjectName("label_pattern_set")
        self.gridLayout_4.addWidget(self.label_pattern_set, 16, 0, 1, 1)
        self.checkBox_move_all_patterns = QtWidgets.QCheckBox(self.tab_patterns)
        self.checkBox_move_all_patterns.setObjectName("checkBox_move_all_patterns")
        self.gridLayout_4.addWidget(self.checkBox_move_all_patterns, 25, 0, 1, 1)
        self.tabWidget.addTab(self.tab_patterns, "")
        self.gridLayout.addWidget(self.tabWidget, 17, 0, 1, 2)
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
        self.gridLayout.addWidget(self.line, 19, 0, 1, 2)
        self.pushButton_run_milling = QtWidgets.QPushButton(Form)
        self.pushButton_run_milling.setObjectName("pushButton_run_milling")
        self.gridLayout.addWidget(self.pushButton_run_milling, 27, 0, 1, 2)
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
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_milling_header.setText(_translate("Form", "Milling"))
        self.label_preset.setText(_translate("Form", "Preset"))
        self.label_spacing.setText(_translate("Form", "Spacing"))
        self.label_hfw.setText(_translate("Form", "Horizontal Field Width (um)"))
        self.label_rate.setText(_translate("Form", "Rate (mm3/A/s)"))
        self.checkBox_live_update.setText(_translate("Form", "Live Update"))
        self.label_milling_current.setText(_translate("Form", "Current (A)"))
        self.label_dwell_time.setText(_translate("Form", "Dwell Time (us)"))
        self.pushButton_add_milling_stage.setText(_translate("Form", "Add"))
        self.label_centre_x.setText(_translate("Form", "Centre X (um)"))
        self.label_application_file.setText(_translate("Form", "Application File"))
        self.label_spot_size.setText(_translate("Form", "Spot Size (um)"))
        self.label_centre_y.setText(_translate("Form", "Centre Y (um)"))
        self.label_milling_stage.setText(_translate("Form", "Milling Stage"))
        self.pushButton_remove_milling_stage.setText(_translate("Form", "Remove"))
        self.pushButton.setText(_translate("Form", "Update Pattern"))
        self.label_patterns_header.setText(_translate("Form", "Patterns"))
        self.label_pattern_set.setText(_translate("Form", "Pattern"))
        self.checkBox_move_all_patterns.setText(_translate("Form", "Move All Patterns Together"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_patterns), _translate("Form", "Patterns"))
        self.pushButton_run_milling.setText(_translate("Form", "Run Milling"))
        self.label_title.setText(_translate("Form", "Milling"))
