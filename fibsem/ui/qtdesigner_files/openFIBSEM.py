# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'openFIBSEM.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_save_path = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_save_path.setObjectName("lineEdit_save_path")
        self.gridLayout.addWidget(self.lineEdit_save_path, 2, 1, 1, 1)
        self.label_num_classes = QtWidgets.QLabel(Dialog)
        self.label_num_classes.setObjectName("label_num_classes")
        self.gridLayout.addWidget(self.label_num_classes, 4, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)
        self.label_save_path = QtWidgets.QLabel(Dialog)
        self.label_save_path.setObjectName("label_save_path")
        self.gridLayout.addWidget(self.label_save_path, 2, 0, 1, 1)
        self.pushButton_load_data = QtWidgets.QPushButton(Dialog)
        self.pushButton_load_data.setObjectName("pushButton_load_data")
        self.gridLayout.addWidget(self.pushButton_load_data, 5, 0, 1, 2)
        self.label_progress = QtWidgets.QLabel(Dialog)
        self.label_progress.setAlignment(QtCore.Qt.AlignCenter)
        self.label_progress.setObjectName("label_progress")
        self.gridLayout.addWidget(self.label_progress, 6, 0, 1, 2)
        self.spinBox_num_classes = QtWidgets.QSpinBox(Dialog)
        self.spinBox_num_classes.setMaximum(255)
        self.spinBox_num_classes.setProperty("value", 1)
        self.spinBox_num_classes.setObjectName("spinBox_num_classes")
        self.gridLayout.addWidget(self.spinBox_num_classes, 4, 1, 1, 1)
        self.pushButton_next = QtWidgets.QPushButton(Dialog)
        self.pushButton_next.setObjectName("pushButton_next")
        self.gridLayout.addWidget(self.pushButton_next, 8, 1, 1, 1)
        self.label_raw_data = QtWidgets.QLabel(Dialog)
        self.label_raw_data.setObjectName("label_raw_data")
        self.gridLayout.addWidget(self.label_raw_data, 1, 0, 1, 1)
        self.lineEdit_raw_data = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_raw_data.setObjectName("lineEdit_raw_data")
        self.gridLayout.addWidget(self.lineEdit_raw_data, 1, 1, 1, 1)
        self.pushButton_previous = QtWidgets.QPushButton(Dialog)
        self.pushButton_previous.setObjectName("pushButton_previous")
        self.gridLayout.addWidget(self.pushButton_previous, 8, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 9, 0, 1, 2)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_num_classes.setText(_translate("Dialog", "Num Classes"))
        self.label_title.setText(_translate("Dialog", "FIBSEM Labelling"))
        self.label_save_path.setText(_translate("Dialog", "Save Path"))
        self.pushButton_load_data.setText(_translate("Dialog", "Load Data"))
        self.label_progress.setText(_translate("Dialog", "Progress"))
        self.pushButton_next.setText(_translate("Dialog", "Next"))
        self.label_raw_data.setText(_translate("Dialog", "Raw Data"))
        self.pushButton_previous.setText(_translate("Dialog", "Previous"))
