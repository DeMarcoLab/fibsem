# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemGISWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(506, 637)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_position = QtWidgets.QLabel(Form)
        self.label_position.setObjectName("label_position")
        self.gridLayout_2.addWidget(self.label_position, 4, 0, 1, 1)
        self.run_button = QtWidgets.QPushButton(Form)
        self.run_button.setObjectName("run_button")
        self.gridLayout_2.addWidget(self.run_button, 18, 0, 1, 2)
        self.hfw_spinbox = QtWidgets.QDoubleSpinBox(Form)
        self.hfw_spinbox.setMaximum(10000.0)
        self.hfw_spinbox.setObjectName("hfw_spinbox")
        self.gridLayout_2.addWidget(self.hfw_spinbox, 15, 1, 1, 1)
        self.GIS_radioButton = QtWidgets.QRadioButton(Form)
        self.GIS_radioButton.setObjectName("GIS_radioButton")
        self.gridLayout_2.addWidget(self.GIS_radioButton, 1, 0, 1, 1)
        self.gas_combobox = QtWidgets.QComboBox(Form)
        self.gas_combobox.setObjectName("gas_combobox")
        self.gridLayout_2.addWidget(self.gas_combobox, 10, 1, 1, 1)
        self.label_time = QtWidgets.QLabel(Form)
        self.label_time.setObjectName("label_time")
        self.gridLayout_2.addWidget(self.label_time, 13, 0, 1, 1)
        self.timeDuration_spinbox = QtWidgets.QDoubleSpinBox(Form)
        self.timeDuration_spinbox.setObjectName("timeDuration_spinbox")
        self.gridLayout_2.addWidget(self.timeDuration_spinbox, 13, 1, 1, 1)
        self.current_position_label = QtWidgets.QLabel(Form)
        self.current_position_label.setObjectName("current_position_label")
        self.gridLayout_2.addWidget(self.current_position_label, 7, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 19, 0, 1, 2)
        self.beamtype_combobox = QtWidgets.QComboBox(Form)
        self.beamtype_combobox.setObjectName("beamtype_combobox")
        self.beamtype_combobox.addItem("")
        self.beamtype_combobox.addItem("")
        self.gridLayout_2.addWidget(self.beamtype_combobox, 14, 1, 1, 1)
        self.app_file_label = QtWidgets.QLabel(Form)
        self.app_file_label.setObjectName("app_file_label")
        self.gridLayout_2.addWidget(self.app_file_label, 17, 0, 1, 1)
        self.position_combobox = QtWidgets.QComboBox(Form)
        self.position_combobox.setObjectName("position_combobox")
        self.gridLayout_2.addWidget(self.position_combobox, 4, 1, 1, 1)
        self.label_gas = QtWidgets.QLabel(Form)
        self.label_gas.setObjectName("label_gas")
        self.gridLayout_2.addWidget(self.label_gas, 10, 0, 1, 1)
        self.insertGIS_button = QtWidgets.QPushButton(Form)
        self.insertGIS_button.setObjectName("insertGIS_button")
        self.gridLayout_2.addWidget(self.insertGIS_button, 3, 1, 1, 1)
        self.move_GIS_button = QtWidgets.QPushButton(Form)
        self.move_GIS_button.setObjectName("move_GIS_button")
        self.gridLayout_2.addWidget(self.move_GIS_button, 7, 1, 1, 1)
        self.label_beam_type = QtWidgets.QLabel(Form)
        self.label_beam_type.setObjectName("label_beam_type")
        self.gridLayout_2.addWidget(self.label_beam_type, 14, 0, 1, 1)
        self.multichem_radioButton = QtWidgets.QRadioButton(Form)
        self.multichem_radioButton.setObjectName("multichem_radioButton")
        self.gridLayout_2.addWidget(self.multichem_radioButton, 1, 1, 1, 1)
        self.label_hfw = QtWidgets.QLabel(Form)
        self.label_hfw.setObjectName("label_hfw")
        self.gridLayout_2.addWidget(self.label_hfw, 15, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout_2.addWidget(self.label_title, 0, 0, 1, 2)
        self.blankBeamcheckbox = QtWidgets.QCheckBox(Form)
        self.blankBeamcheckbox.setObjectName("blankBeamcheckbox")
        self.gridLayout_2.addWidget(self.blankBeamcheckbox, 16, 1, 1, 1)
        self.GIS_insert_status_label = QtWidgets.QLabel(Form)
        self.GIS_insert_status_label.setText("")
        self.GIS_insert_status_label.setObjectName("GIS_insert_status_label")
        self.gridLayout_2.addWidget(self.GIS_insert_status_label, 3, 0, 1, 1)
        self.warm_button = QtWidgets.QPushButton(Form)
        self.warm_button.setObjectName("warm_button")
        self.gridLayout_2.addWidget(self.warm_button, 12, 1, 1, 1)
        self.temp_label = QtWidgets.QLabel(Form)
        self.temp_label.setObjectName("temp_label")
        self.gridLayout_2.addWidget(self.temp_label, 12, 0, 1, 1)
        self.app_file_combobox = QtWidgets.QComboBox(Form)
        self.app_file_combobox.setObjectName("app_file_combobox")
        self.gridLayout_2.addWidget(self.app_file_combobox, 17, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_position.setText(_translate("Form", "Position"))
        self.run_button.setText(_translate("Form", "Run"))
        self.GIS_radioButton.setText(_translate("Form", "GIS"))
        self.label_time.setText(_translate("Form", "Time Duration (us)"))
        self.current_position_label.setText(_translate("Form", "Current Position:"))
        self.beamtype_combobox.setItemText(0, _translate("Form", "ELECTRON"))
        self.beamtype_combobox.setItemText(1, _translate("Form", "ION"))
        self.app_file_label.setText(_translate("Form", "App. File"))
        self.label_gas.setText(_translate("Form", "Select Gas"))
        self.insertGIS_button.setText(_translate("Form", "Insert"))
        self.move_GIS_button.setText(_translate("Form", "Move to Position"))
        self.label_beam_type.setText(_translate("Form", "BeamType"))
        self.multichem_radioButton.setText(_translate("Form", "MultiChem"))
        self.label_hfw.setText(_translate("Form", "HFW (um)"))
        self.label_title.setText(_translate("Form", "Gas Injection System"))
        self.blankBeamcheckbox.setText(_translate("Form", "Blank Beam"))
        self.warm_button.setText(_translate("Form", "Warm up"))
        self.temp_label.setText(_translate("Form", "Temp: "))
