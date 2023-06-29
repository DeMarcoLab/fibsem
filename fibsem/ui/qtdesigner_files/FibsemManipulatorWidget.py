# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemManipulatorWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(286, 489)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.moveRelative_button = QtWidgets.QPushButton(Form)
        self.moveRelative_button.setObjectName("moveRelative_button")
        self.gridLayout_2.addWidget(self.moveRelative_button, 11, 0, 1, 2)
        self.savedPosition_combobox = QtWidgets.QComboBox(Form)
        self.savedPosition_combobox.setObjectName("savedPosition_combobox")
        self.gridLayout_2.addWidget(self.savedPosition_combobox, 13, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 14, 0, 1, 2)
        self.savedPositionName_lineEdit = QtWidgets.QLineEdit(Form)
        self.savedPositionName_lineEdit.setObjectName("savedPositionName_lineEdit")
        self.gridLayout_2.addWidget(self.savedPositionName_lineEdit, 12, 1, 1, 1)
        self.dr_label = QtWidgets.QLabel(Form)
        self.dr_label.setObjectName("dr_label")
        self.gridLayout_2.addWidget(self.dr_label, 9, 0, 1, 1)
        self.dX_spinbox = QtWidgets.QDoubleSpinBox(Form)
        self.dX_spinbox.setMinimum(-1000.0)
        self.dX_spinbox.setMaximum(1000.0)
        self.dX_spinbox.setObjectName("dX_spinbox")
        self.gridLayout_2.addWidget(self.dX_spinbox, 6, 1, 1, 1)
        self.dy_label = QtWidgets.QLabel(Form)
        self.dy_label.setObjectName("dy_label")
        self.gridLayout_2.addWidget(self.dy_label, 7, 0, 1, 1)
        self.move_type_comboBox = QtWidgets.QComboBox(Form)
        self.move_type_comboBox.setObjectName("move_type_comboBox")
        self.move_type_comboBox.addItem("")
        self.move_type_comboBox.addItem("")
        self.gridLayout_2.addWidget(self.move_type_comboBox, 5, 0, 1, 2)
        self.label_title = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout_2.addWidget(self.label_title, 0, 0, 1, 2)
        self.dz_label = QtWidgets.QLabel(Form)
        self.dz_label.setObjectName("dz_label")
        self.gridLayout_2.addWidget(self.dz_label, 8, 0, 1, 1)
        self.beam_type_combobox = QtWidgets.QComboBox(Form)
        self.beam_type_combobox.setObjectName("beam_type_combobox")
        self.beam_type_combobox.addItem("")
        self.beam_type_combobox.addItem("")
        self.gridLayout_2.addWidget(self.beam_type_combobox, 10, 1, 1, 1)
        self.dZ_spinbox = QtWidgets.QDoubleSpinBox(Form)
        self.dZ_spinbox.setMinimum(-1000.0)
        self.dZ_spinbox.setMaximum(1000.0)
        self.dZ_spinbox.setObjectName("dZ_spinbox")
        self.gridLayout_2.addWidget(self.dZ_spinbox, 8, 1, 1, 1)
        self.dY_spinbox = QtWidgets.QDoubleSpinBox(Form)
        self.dY_spinbox.setMinimum(-1000.0)
        self.dY_spinbox.setMaximum(1000.0)
        self.dY_spinbox.setObjectName("dY_spinbox")
        self.gridLayout_2.addWidget(self.dY_spinbox, 7, 1, 1, 1)
        self.manipulatorStatus_label = QtWidgets.QLabel(Form)
        self.manipulatorStatus_label.setText("")
        self.manipulatorStatus_label.setObjectName("manipulatorStatus_label")
        self.gridLayout_2.addWidget(self.manipulatorStatus_label, 2, 0, 1, 2)
        self.dR_spinbox = QtWidgets.QDoubleSpinBox(Form)
        self.dR_spinbox.setMinimum(-365.0)
        self.dR_spinbox.setMaximum(365.0)
        self.dR_spinbox.setObjectName("dR_spinbox")
        self.gridLayout_2.addWidget(self.dR_spinbox, 9, 1, 1, 1)
        self.goToPosition_button = QtWidgets.QPushButton(Form)
        self.goToPosition_button.setObjectName("goToPosition_button")
        self.gridLayout_2.addWidget(self.goToPosition_button, 13, 0, 1, 1)
        self.addSavedPosition_button = QtWidgets.QPushButton(Form)
        self.addSavedPosition_button.setObjectName("addSavedPosition_button")
        self.gridLayout_2.addWidget(self.addSavedPosition_button, 12, 0, 1, 1)
        self.insertManipulator_button = QtWidgets.QPushButton(Form)
        self.insertManipulator_button.setObjectName("insertManipulator_button")
        self.gridLayout_2.addWidget(self.insertManipulator_button, 4, 0, 1, 2)
        self.dx_label = QtWidgets.QLabel(Form)
        self.dx_label.setObjectName("dx_label")
        self.gridLayout_2.addWidget(self.dx_label, 6, 0, 1, 1)
        self.beam_type_label = QtWidgets.QLabel(Form)
        self.beam_type_label.setObjectName("beam_type_label")
        self.gridLayout_2.addWidget(self.beam_type_label, 10, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.moveRelative_button.setText(_translate("Form", "Move "))
        self.dr_label.setText(_translate("Form", "dR (deg)"))
        self.dy_label.setText(_translate("Form", "dY (um)"))
        self.move_type_comboBox.setItemText(0, _translate("Form", "Relative Move"))
        self.move_type_comboBox.setItemText(1, _translate("Form", "Corrected Move"))
        self.label_title.setText(_translate("Form", "Manipulator"))
        self.dz_label.setText(_translate("Form", "dZ (um)"))
        self.beam_type_combobox.setItemText(0, _translate("Form", "ION"))
        self.beam_type_combobox.setItemText(1, _translate("Form", "ELECTRON"))
        self.goToPosition_button.setText(_translate("Form", "Go To Position"))
        self.addSavedPosition_button.setText(_translate("Form", "Save Position"))
        self.insertManipulator_button.setText(_translate("Form", "Insert"))
        self.dx_label.setText(_translate("Form", "dX (um)"))
        self.beam_type_label.setText(_translate("Form", "Beam Type"))
