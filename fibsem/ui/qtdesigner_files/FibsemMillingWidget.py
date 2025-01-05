# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemMillingWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(780, 890)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.progressBar_milling_stages = QtWidgets.QProgressBar(Form)
        self.progressBar_milling_stages.setProperty("value", 24)
        self.progressBar_milling_stages.setObjectName("progressBar_milling_stages")
        self.gridLayout.addWidget(self.progressBar_milling_stages, 28, 0, 1, 2)
        self.pushButton_stop_milling = QtWidgets.QPushButton(Form)
        self.pushButton_stop_milling.setObjectName("pushButton_stop_milling")
        self.gridLayout.addWidget(self.pushButton_stop_milling, 26, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 30, 0, 1, 2)
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setMinimumSize(QtCore.QSize(0, 0))
        self.frame.setObjectName("frame")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_milling_instructions = QtWidgets.QLabel(self.frame)
        self.label_milling_instructions.setObjectName("label_milling_instructions")
        self.gridLayout_4.addWidget(self.label_milling_instructions, 25, 0, 1, 2)
        self.pushButton_add_milling_stage = QtWidgets.QPushButton(self.frame)
        self.pushButton_add_milling_stage.setObjectName("pushButton_add_milling_stage")
        self.gridLayout_4.addWidget(self.pushButton_add_milling_stage, 3, 0, 1, 1)
        self.label_milling_stage = QtWidgets.QLabel(self.frame)
        self.label_milling_stage.setObjectName("label_milling_stage")
        self.gridLayout_4.addWidget(self.label_milling_stage, 2, 0, 1, 1)
        self.scrollArea_milling_stage = QtWidgets.QScrollArea(self.frame)
        self.scrollArea_milling_stage.setWidgetResizable(True)
        self.scrollArea_milling_stage.setObjectName("scrollArea_milling_stage")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 728, 809))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.groupBox_milling_settings = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_milling_settings.setObjectName("groupBox_milling_settings")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_milling_settings)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_dwell_time = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_dwell_time.setObjectName("label_dwell_time")
        self.gridLayout_3.addWidget(self.label_dwell_time, 8, 0, 1, 1)
        self.label_voltage = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_voltage.setObjectName("label_voltage")
        self.gridLayout_3.addWidget(self.label_voltage, 2, 0, 1, 1)
        self.label_spacing = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_spacing.setObjectName("label_spacing")
        self.gridLayout_3.addWidget(self.label_spacing, 10, 0, 1, 1)
        self.label_patterning_mode = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_patterning_mode.setObjectName("label_patterning_mode")
        self.gridLayout_3.addWidget(self.label_patterning_mode, 0, 0, 1, 1)
        self.label_preset = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_preset.setObjectName("label_preset")
        self.gridLayout_3.addWidget(self.label_preset, 6, 0, 1, 1)
        self.label_milling_current = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_milling_current.setObjectName("label_milling_current")
        self.gridLayout_3.addWidget(self.label_milling_current, 3, 0, 1, 1)
        self.comboBox_patterning_mode = QtWidgets.QComboBox(self.groupBox_milling_settings)
        self.comboBox_patterning_mode.setObjectName("comboBox_patterning_mode")
        self.gridLayout_3.addWidget(self.comboBox_patterning_mode, 0, 1, 1, 1)
        self.comboBox_preset = QtWidgets.QComboBox(self.groupBox_milling_settings)
        self.comboBox_preset.setObjectName("comboBox_preset")
        self.gridLayout_3.addWidget(self.comboBox_preset, 6, 1, 1, 1)
        self.doubleSpinBox_spacing = QtWidgets.QDoubleSpinBox(self.groupBox_milling_settings)
        self.doubleSpinBox_spacing.setProperty("value", 1.0)
        self.doubleSpinBox_spacing.setObjectName("doubleSpinBox_spacing")
        self.gridLayout_3.addWidget(self.doubleSpinBox_spacing, 10, 1, 1, 1)
        self.label_application_file = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_application_file.setObjectName("label_application_file")
        self.gridLayout_3.addWidget(self.label_application_file, 5, 0, 1, 1)
        self.doubleSpinBox_milling_current = QtWidgets.QDoubleSpinBox(self.groupBox_milling_settings)
        self.doubleSpinBox_milling_current.setObjectName("doubleSpinBox_milling_current")
        self.gridLayout_3.addWidget(self.doubleSpinBox_milling_current, 3, 1, 1, 1)
        self.doubleSpinBox_rate = QtWidgets.QDoubleSpinBox(self.groupBox_milling_settings)
        self.doubleSpinBox_rate.setDecimals(4)
        self.doubleSpinBox_rate.setMinimum(0.0)
        self.doubleSpinBox_rate.setMaximum(100000.0)
        self.doubleSpinBox_rate.setSingleStep(0.01)
        self.doubleSpinBox_rate.setProperty("value", 0.0)
        self.doubleSpinBox_rate.setObjectName("doubleSpinBox_rate")
        self.gridLayout_3.addWidget(self.doubleSpinBox_rate, 7, 1, 1, 1)
        self.doubleSpinBox_dwell_time = QtWidgets.QDoubleSpinBox(self.groupBox_milling_settings)
        self.doubleSpinBox_dwell_time.setDecimals(4)
        self.doubleSpinBox_dwell_time.setMinimum(0.0)
        self.doubleSpinBox_dwell_time.setMaximum(4000000.0)
        self.doubleSpinBox_dwell_time.setSingleStep(0.01)
        self.doubleSpinBox_dwell_time.setProperty("value", 0.0)
        self.doubleSpinBox_dwell_time.setObjectName("doubleSpinBox_dwell_time")
        self.gridLayout_3.addWidget(self.doubleSpinBox_dwell_time, 8, 1, 1, 1)
        self.comboBox_application_file = QtWidgets.QComboBox(self.groupBox_milling_settings)
        self.comboBox_application_file.setObjectName("comboBox_application_file")
        self.gridLayout_3.addWidget(self.comboBox_application_file, 5, 1, 1, 1)
        self.label_rate = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_rate.setObjectName("label_rate")
        self.gridLayout_3.addWidget(self.label_rate, 7, 0, 1, 1)
        self.doubleSpinBox_spot_size = QtWidgets.QDoubleSpinBox(self.groupBox_milling_settings)
        self.doubleSpinBox_spot_size.setDecimals(4)
        self.doubleSpinBox_spot_size.setMinimum(0.0)
        self.doubleSpinBox_spot_size.setMaximum(100000.0)
        self.doubleSpinBox_spot_size.setSingleStep(0.01)
        self.doubleSpinBox_spot_size.setProperty("value", 0.0)
        self.doubleSpinBox_spot_size.setObjectName("doubleSpinBox_spot_size")
        self.gridLayout_3.addWidget(self.doubleSpinBox_spot_size, 9, 1, 1, 1)
        self.label_spot_size = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_spot_size.setObjectName("label_spot_size")
        self.gridLayout_3.addWidget(self.label_spot_size, 9, 0, 1, 1)
        self.spinBox_voltage = QtWidgets.QSpinBox(self.groupBox_milling_settings)
        self.spinBox_voltage.setMaximum(1000000)
        self.spinBox_voltage.setProperty("value", 30000)
        self.spinBox_voltage.setObjectName("spinBox_voltage")
        self.gridLayout_3.addWidget(self.spinBox_voltage, 2, 1, 1, 1)
        self.label_hfw = QtWidgets.QLabel(self.groupBox_milling_settings)
        self.label_hfw.setObjectName("label_hfw")
        self.gridLayout_3.addWidget(self.label_hfw, 1, 0, 1, 1)
        self.doubleSpinBox_hfw = QtWidgets.QDoubleSpinBox(self.groupBox_milling_settings)
        self.doubleSpinBox_hfw.setEnabled(True)
        self.doubleSpinBox_hfw.setReadOnly(True)
        self.doubleSpinBox_hfw.setMinimum(10.0)
        self.doubleSpinBox_hfw.setMaximum(1000000000.0)
        self.doubleSpinBox_hfw.setProperty("value", 150.0)
        self.doubleSpinBox_hfw.setObjectName("doubleSpinBox_hfw")
        self.gridLayout_3.addWidget(self.doubleSpinBox_hfw, 1, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_milling_settings, 0, 0, 1, 1)
        self.groupBox_aligment = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_aligment.setObjectName("groupBox_aligment")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_aligment)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.checkBox_alignment_enabled = QtWidgets.QCheckBox(self.groupBox_aligment)
        self.checkBox_alignment_enabled.setObjectName("checkBox_alignment_enabled")
        self.gridLayout_7.addWidget(self.checkBox_alignment_enabled, 1, 0, 1, 1)
        self.checkBox_alignment_interval_enabled = QtWidgets.QCheckBox(self.groupBox_aligment)
        self.checkBox_alignment_interval_enabled.setObjectName("checkBox_alignment_interval_enabled")
        self.gridLayout_7.addWidget(self.checkBox_alignment_interval_enabled, 1, 1, 1, 1)
        self.doubleSpinBox_alignment_interval = QtWidgets.QDoubleSpinBox(self.groupBox_aligment)
        self.doubleSpinBox_alignment_interval.setObjectName("doubleSpinBox_alignment_interval")
        self.gridLayout_7.addWidget(self.doubleSpinBox_alignment_interval, 2, 1, 1, 1)
        self.label_alignment_interval = QtWidgets.QLabel(self.groupBox_aligment)
        self.label_alignment_interval.setObjectName("label_alignment_interval")
        self.gridLayout_7.addWidget(self.label_alignment_interval, 2, 0, 1, 1)
        self.pushButton_alignment_edit_area = QtWidgets.QPushButton(self.groupBox_aligment)
        self.pushButton_alignment_edit_area.setObjectName("pushButton_alignment_edit_area")
        self.gridLayout_7.addWidget(self.pushButton_alignment_edit_area, 4, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_aligment, 1, 0, 1, 1)
        self.groupBox_pattern_settings = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_pattern_settings.setObjectName("groupBox_pattern_settings")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_pattern_settings)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.comboBox_patterns = QtWidgets.QComboBox(self.groupBox_pattern_settings)
        self.comboBox_patterns.setObjectName("comboBox_patterns")
        self.gridLayout_5.addWidget(self.comboBox_patterns, 0, 1, 1, 2)
        self.label_centre_x = QtWidgets.QLabel(self.groupBox_pattern_settings)
        self.label_centre_x.setObjectName("label_centre_x")
        self.gridLayout_5.addWidget(self.label_centre_x, 1, 0, 1, 1)
        self.gridLayout_patterns = QtWidgets.QGridLayout()
        self.gridLayout_patterns.setObjectName("gridLayout_patterns")
        self.gridLayout_5.addLayout(self.gridLayout_patterns, 3, 0, 1, 3)
        self.doubleSpinBox_centre_x = QtWidgets.QDoubleSpinBox(self.groupBox_pattern_settings)
        self.doubleSpinBox_centre_x.setMinimum(-1e+28)
        self.doubleSpinBox_centre_x.setMaximum(1e+18)
        self.doubleSpinBox_centre_x.setSingleStep(0.1)
        self.doubleSpinBox_centre_x.setObjectName("doubleSpinBox_centre_x")
        self.gridLayout_5.addWidget(self.doubleSpinBox_centre_x, 1, 1, 1, 1)
        self.label_pattern_set = QtWidgets.QLabel(self.groupBox_pattern_settings)
        self.label_pattern_set.setObjectName("label_pattern_set")
        self.gridLayout_5.addWidget(self.label_pattern_set, 0, 0, 1, 1)
        self.doubleSpinBox_centre_y = QtWidgets.QDoubleSpinBox(self.groupBox_pattern_settings)
        self.doubleSpinBox_centre_y.setMinimum(-1e+16)
        self.doubleSpinBox_centre_y.setMaximum(1e+23)
        self.doubleSpinBox_centre_y.setSingleStep(0.1)
        self.doubleSpinBox_centre_y.setObjectName("doubleSpinBox_centre_y")
        self.gridLayout_5.addWidget(self.doubleSpinBox_centre_y, 1, 2, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_pattern_settings, 3, 0, 1, 1)
        self.groupBox_options = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_options.setObjectName("groupBox_options")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_options)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.checkBox_show_milling_patterns = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_show_milling_patterns.setObjectName("checkBox_show_milling_patterns")
        self.gridLayout_6.addWidget(self.checkBox_show_milling_patterns, 1, 0, 1, 1)
        self.checkBox_relative_move = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_relative_move.setChecked(True)
        self.checkBox_relative_move.setObjectName("checkBox_relative_move")
        self.gridLayout_6.addWidget(self.checkBox_relative_move, 0, 0, 1, 1)
        self.checkBox_show_milling_crosshair = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_show_milling_crosshair.setChecked(True)
        self.checkBox_show_milling_crosshair.setObjectName("checkBox_show_milling_crosshair")
        self.gridLayout_6.addWidget(self.checkBox_show_milling_crosshair, 1, 1, 1, 1)
        self.checkBox_show_advanced = QtWidgets.QCheckBox(self.groupBox_options)
        self.checkBox_show_advanced.setObjectName("checkBox_show_advanced")
        self.gridLayout_6.addWidget(self.checkBox_show_advanced, 0, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_options, 4, 0, 1, 1)
        self.groupBox_strategy = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_strategy.setObjectName("groupBox_strategy")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_strategy)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_strategy_name = QtWidgets.QLabel(self.groupBox_strategy)
        self.label_strategy_name.setObjectName("label_strategy_name")
        self.gridLayout_8.addWidget(self.label_strategy_name, 0, 0, 1, 1)
        self.comboBox_strategy_name = QtWidgets.QComboBox(self.groupBox_strategy)
        self.comboBox_strategy_name.setObjectName("comboBox_strategy_name")
        self.gridLayout_8.addWidget(self.comboBox_strategy_name, 0, 1, 1, 1)
        self.gridLayout_strategy = QtWidgets.QGridLayout()
        self.gridLayout_strategy.setObjectName("gridLayout_strategy")
        self.gridLayout_8.addLayout(self.gridLayout_strategy, 1, 0, 1, 2)
        self.gridLayout_2.addWidget(self.groupBox_strategy, 2, 0, 1, 1)
        self.scrollArea_milling_stage.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_4.addWidget(self.scrollArea_milling_stage, 4, 0, 4, 2)
        self.pushButton_remove_milling_stage = QtWidgets.QPushButton(self.frame)
        self.pushButton_remove_milling_stage.setObjectName("pushButton_remove_milling_stage")
        self.gridLayout_4.addWidget(self.pushButton_remove_milling_stage, 3, 1, 1, 1)
        self.comboBox_milling_stage = QtWidgets.QComboBox(self.frame)
        self.comboBox_milling_stage.setObjectName("comboBox_milling_stage")
        self.gridLayout_4.addWidget(self.comboBox_milling_stage, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.frame, 16, 0, 1, 2)
        self.progressBar_milling = QtWidgets.QProgressBar(Form)
        self.progressBar_milling.setProperty("value", 24)
        self.progressBar_milling.setObjectName("progressBar_milling")
        self.gridLayout.addWidget(self.progressBar_milling, 29, 0, 1, 2)
        self.pushButton_pause_milling = QtWidgets.QPushButton(Form)
        self.pushButton_pause_milling.setObjectName("pushButton_pause_milling")
        self.gridLayout.addWidget(self.pushButton_pause_milling, 26, 0, 1, 1)
        self.listWidget_active_milling_stages = QtWidgets.QListWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget_active_milling_stages.sizePolicy().hasHeightForWidth())
        self.listWidget_active_milling_stages.setSizePolicy(sizePolicy)
        self.listWidget_active_milling_stages.setMaximumSize(QtCore.QSize(16777215, 100))
        self.listWidget_active_milling_stages.setObjectName("listWidget_active_milling_stages")
        self.gridLayout.addWidget(self.listWidget_active_milling_stages, 24, 0, 1, 2)
        self.pushButton_run_milling = QtWidgets.QPushButton(Form)
        self.pushButton_run_milling.setObjectName("pushButton_run_milling")
        self.gridLayout.addWidget(self.pushButton_run_milling, 25, 0, 1, 2)
        self.label_milling_information = QtWidgets.QLabel(Form)
        self.label_milling_information.setObjectName("label_milling_information")
        self.gridLayout.addWidget(self.label_milling_information, 27, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.frame, self.comboBox_milling_stage)
        Form.setTabOrder(self.comboBox_milling_stage, self.pushButton_add_milling_stage)
        Form.setTabOrder(self.pushButton_add_milling_stage, self.pushButton_remove_milling_stage)
        Form.setTabOrder(self.pushButton_remove_milling_stage, self.pushButton_run_milling)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_stop_milling.setText(_translate("Form", "Stop Milling"))
        self.label_milling_instructions.setText(_translate("Form", "Controls:"))
        self.pushButton_add_milling_stage.setText(_translate("Form", "Add"))
        self.label_milling_stage.setText(_translate("Form", "Milling Stage"))
        self.groupBox_milling_settings.setTitle(_translate("Form", "Milling Settings"))
        self.label_dwell_time.setText(_translate("Form", "Dwell Time (us)"))
        self.label_voltage.setText(_translate("Form", "Voltage (V)"))
        self.label_spacing.setText(_translate("Form", "Spacing"))
        self.label_patterning_mode.setText(_translate("Form", "Patterning Mode"))
        self.label_preset.setText(_translate("Form", "Preset"))
        self.label_milling_current.setText(_translate("Form", "Current (nA)"))
        self.label_application_file.setText(_translate("Form", "Application File"))
        self.label_rate.setText(_translate("Form", "Rate (mm3/A/s)"))
        self.label_spot_size.setText(_translate("Form", "Spot Size (um)"))
        self.label_hfw.setText(_translate("Form", "Field of View (um)"))
        self.groupBox_aligment.setTitle(_translate("Form", "Alignment"))
        self.checkBox_alignment_enabled.setText(_translate("Form", "Initial Alignment"))
        self.checkBox_alignment_interval_enabled.setText(_translate("Form", "Interval Alignment"))
        self.label_alignment_interval.setText(_translate("Form", "Interval (s)"))
        self.pushButton_alignment_edit_area.setText(_translate("Form", "Edit Alignment Area"))
        self.groupBox_pattern_settings.setTitle(_translate("Form", "Pattern"))
        self.label_centre_x.setText(_translate("Form", "Centre (um)"))
        self.label_pattern_set.setText(_translate("Form", "Pattern"))
        self.groupBox_options.setTitle(_translate("Form", "Options"))
        self.checkBox_show_milling_patterns.setText(_translate("Form", "Show Milling Patterns"))
        self.checkBox_relative_move.setText(_translate("Form", "Keep Relative Orientation"))
        self.checkBox_show_milling_crosshair.setText(_translate("Form", "Show Milling Crosshair"))
        self.checkBox_show_advanced.setText(_translate("Form", "Show Advanced"))
        self.groupBox_strategy.setTitle(_translate("Form", "Milling Strategy"))
        self.label_strategy_name.setText(_translate("Form", "Strategy Name"))
        self.pushButton_remove_milling_stage.setText(_translate("Form", "Remove"))
        self.pushButton_pause_milling.setText(_translate("Form", "Pause Milling"))
        self.pushButton_run_milling.setText(_translate("Form", "Run Milling"))
        self.label_milling_information.setText(_translate("Form", "TextLabel"))
