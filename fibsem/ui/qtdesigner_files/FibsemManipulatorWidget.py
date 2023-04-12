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
        Form.resize(286, 376)
        self.gridLayoutWidget = QtWidgets.QWidget(Form)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 281, 293))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.insertManipulator_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.insertManipulator_button.setObjectName("insertManipulator_button")
        self.gridLayout.addWidget(self.insertManipulator_button, 1, 1, 1, 1)
        self.insertPosition_combobox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.insertPosition_combobox.setObjectName("insertPosition_combobox")
        self.gridLayout.addWidget(self.insertPosition_combobox, 10, 2, 1, 1)
        self.zCoordinate_spinbox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.zCoordinate_spinbox.setDecimals(4)
        self.zCoordinate_spinbox.setMinimum(-99.0)
        self.zCoordinate_spinbox.setObjectName("zCoordinate_spinbox")
        self.gridLayout.addWidget(self.zCoordinate_spinbox, 4, 2, 1, 1)
        self.rotationCoordinate_spinbox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.rotationCoordinate_spinbox.setMinimum(-99.0)
        self.rotationCoordinate_spinbox.setObjectName("rotationCoordinate_spinbox")
        self.gridLayout.addWidget(self.rotationCoordinate_spinbox, 5, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 10, 1, 1, 1)
        self.xCoordinate_spinbox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.xCoordinate_spinbox.setDecimals(4)
        self.xCoordinate_spinbox.setMinimum(-99.0)
        self.xCoordinate_spinbox.setObjectName("xCoordinate_spinbox")
        self.gridLayout.addWidget(self.xCoordinate_spinbox, 2, 2, 1, 1)
        self.addSavedPosition_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.addSavedPosition_button.setObjectName("addSavedPosition_button")
        self.gridLayout.addWidget(self.addSavedPosition_button, 9, 1, 1, 1)
        self.label_movement_stage_x = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_movement_stage_x.setObjectName("label_movement_stage_x")
        self.gridLayout.addWidget(self.label_movement_stage_x, 2, 1, 1, 1)
        self.label_movement_stage_z = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_movement_stage_z.setObjectName("label_movement_stage_z")
        self.gridLayout.addWidget(self.label_movement_stage_z, 4, 1, 1, 1)
        self.yCoordinate_spinbox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.yCoordinate_spinbox.setDecimals(4)
        self.yCoordinate_spinbox.setMinimum(-99.0)
        self.yCoordinate_spinbox.setObjectName("yCoordinate_spinbox")
        self.gridLayout.addWidget(self.yCoordinate_spinbox, 3, 2, 1, 1)
        self.tiltCoordinate_spinbox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.tiltCoordinate_spinbox.setMinimum(-99.0)
        self.tiltCoordinate_spinbox.setObjectName("tiltCoordinate_spinbox")
        self.gridLayout.addWidget(self.tiltCoordinate_spinbox, 6, 2, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 9, 2, 1, 1)
        self.retractManipulator_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.retractManipulator_button.setObjectName("retractManipulator_button")
        self.gridLayout.addWidget(self.retractManipulator_button, 1, 2, 1, 1)
        self.label_movement_stage_y = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_movement_stage_y.setObjectName("label_movement_stage_y")
        self.gridLayout.addWidget(self.label_movement_stage_y, 3, 1, 1, 1)
        self.label_movement_stage_rotation = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_movement_stage_rotation.setObjectName("label_movement_stage_rotation")
        self.gridLayout.addWidget(self.label_movement_stage_rotation, 5, 1, 1, 1)
        self.label_title = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 1, 1, 2)
        self.label_movement_stage_tilt = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_movement_stage_tilt.setObjectName("label_movement_stage_tilt")
        self.gridLayout.addWidget(self.label_movement_stage_tilt, 6, 1, 1, 1)
        self.movetoposition_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.movetoposition_button.setObjectName("movetoposition_button")
        self.gridLayout.addWidget(self.movetoposition_button, 7, 1, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 8, 1, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.insertManipulator_button.setText(_translate("Form", "Insert"))
        self.pushButton.setText(_translate("Form", "Go To Position"))
        self.addSavedPosition_button.setText(_translate("Form", "Save Position"))
        self.label_movement_stage_x.setText(_translate("Form", "X Coordinate (mm)"))
        self.label_movement_stage_z.setText(_translate("Form", "Z Coordinate (mm)"))
        self.retractManipulator_button.setText(_translate("Form", "Retract"))
        self.label_movement_stage_y.setText(_translate("Form", "Y Coordinate (mm)"))
        self.label_movement_stage_rotation.setText(_translate("Form", "Rotation (deg)"))
        self.label_title.setText(_translate("Form", "Manipulator"))
        self.label_movement_stage_tilt.setText(_translate("Form", "Tilt (deg)"))
        self.movetoposition_button.setText(_translate("Form", "Move to Position"))
