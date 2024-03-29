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
        Form.resize(570, 883)
        Form.setMinimumSize(QtCore.QSize(0, 450))
        font = QtGui.QFont()
        font.setPointSize(10)
        Form.setFont(font)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 4, 0, 1, 3)
        self.comboBox_configuration = QtWidgets.QComboBox(Form)
        self.comboBox_configuration.setObjectName("comboBox_configuration")
        self.gridLayout.addWidget(self.comboBox_configuration, 0, 1, 1, 1)
        self.pushButton_connect_to_microscope = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_connect_to_microscope.setFont(font)
        self.pushButton_connect_to_microscope.setObjectName("pushButton_connect_to_microscope")
        self.gridLayout.addWidget(self.pushButton_connect_to_microscope, 1, 0, 1, 3)
        self.toolButton_import_configuration = QtWidgets.QToolButton(Form)
        self.toolButton_import_configuration.setObjectName("toolButton_import_configuration")
        self.gridLayout.addWidget(self.toolButton_import_configuration, 0, 2, 1, 1)
        self.pushButton_apply_configuration = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_apply_configuration.setFont(font)
        self.pushButton_apply_configuration.setObjectName("pushButton_apply_configuration")
        self.gridLayout.addWidget(self.pushButton_apply_configuration, 3, 0, 1, 3)
        self.label_configuration = QtWidgets.QLabel(Form)
        self.label_configuration.setObjectName("label_configuration")
        self.gridLayout.addWidget(self.label_configuration, 0, 0, 1, 1)
        self.label_connection_information = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_connection_information.setFont(font)
        self.label_connection_information.setObjectName("label_connection_information")
        self.gridLayout.addWidget(self.label_connection_information, 5, 0, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton_connect_to_microscope.setText(_translate("Form", "Connect to Microscope"))
        self.toolButton_import_configuration.setText(_translate("Form", "..."))
        self.pushButton_apply_configuration.setText(_translate("Form", "Apply Microscope Configuration"))
        self.label_configuration.setText(_translate("Form", "Configuration"))
        self.label_connection_information.setText(_translate("Form", "No Connected"))
