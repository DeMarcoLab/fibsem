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
        Form.resize(274, 549)
        self.gridLayoutWidget = QtWidgets.QWidget(Form)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 271, 241))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 6, 0, 1, 1)
        self.hfw_spinbox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.hfw_spinbox.setObjectName("hfw_spinbox")
        self.gridLayout.addWidget(self.hfw_spinbox, 6, 1, 1, 1)
        self.insertGIS_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.insertGIS_button.setObjectName("insertGIS_button")
        self.gridLayout.addWidget(self.insertGIS_button, 1, 0, 1, 1)
        self.retractGIS_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.retractGIS_button.setObjectName("retractGIS_button")
        self.gridLayout.addWidget(self.retractGIS_button, 1, 1, 1, 1)
        self.position_combobox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.position_combobox.setObjectName("position_combobox")
        self.gridLayout.addWidget(self.position_combobox, 2, 1, 1, 1)
        self.gas_combobox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.gas_combobox.setObjectName("gas_combobox")
        self.gridLayout.addWidget(self.gas_combobox, 3, 1, 1, 1)
        self.timeDuration_spinbox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.timeDuration_spinbox.setObjectName("timeDuration_spinbox")
        self.gridLayout.addWidget(self.timeDuration_spinbox, 4, 1, 1, 1)
        self.beamtype_combobox = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.beamtype_combobox.setObjectName("beamtype_combobox")
        self.gridLayout.addWidget(self.beamtype_combobox, 5, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.gridLayoutWidget)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 7, 1, 1, 1)
        self.run_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.run_button.setObjectName("run_button")
        self.gridLayout.addWidget(self.run_button, 8, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_5.setText(_translate("Form", "HFW"))
        self.insertGIS_button.setText(_translate("Form", "Insert"))
        self.retractGIS_button.setText(_translate("Form", "Retract"))
        self.label.setText(_translate("Form", "Position"))
        self.label_2.setText(_translate("Form", "Select Gas"))
        self.label_title.setText(_translate("Form", "GIS"))
        self.label_4.setText(_translate("Form", "BeamType"))
        self.label_3.setText(_translate("Form", "Time Duration"))
        self.checkBox.setText(_translate("Form", "CheckBox"))
        self.run_button.setText(_translate("Form", "Run"))
