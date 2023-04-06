# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FibsemUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(568, 645)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.general = QtWidgets.QWidget()
        self.general.setObjectName("general")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.general)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.general)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.pushButton, 2, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem, 3, 0, 1, 2)
        self.label_ip_address = QtWidgets.QLabel(self.general)
        self.label_ip_address.setObjectName("label_ip_address")
        self.gridLayout_2.addWidget(self.label_ip_address, 0, 0, 1, 1)
        self.label_manufacturer = QtWidgets.QLabel(self.general)
        self.label_manufacturer.setMinimumSize(QtCore.QSize(150, 0))
        self.label_manufacturer.setObjectName("label_manufacturer")
        self.gridLayout_2.addWidget(self.label_manufacturer, 1, 0, 1, 1)
        self.comboBox_manufacturer = QtWidgets.QComboBox(self.general)
        self.comboBox_manufacturer.setObjectName("comboBox_manufacturer")
        self.gridLayout_2.addWidget(self.comboBox_manufacturer, 1, 1, 1, 1)
        self.lineEdit_ip_address = QtWidgets.QLineEdit(self.general)
        self.lineEdit_ip_address.setObjectName("lineEdit_ip_address")
        self.gridLayout_2.addWidget(self.lineEdit_ip_address, 0, 1, 1, 1)
        self.tabWidget.addTab(self.general, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 1)
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 568, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Connect to Microscope"))
        self.label_ip_address.setText(_translate("MainWindow", "IP Address"))
        self.label_manufacturer.setText(_translate("MainWindow", "Manufacturer"))
        self.lineEdit_ip_address.setText(_translate("MainWindow", "localhost"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.general), _translate("MainWindow", "General"))
        self.label_title.setText(_translate("MainWindow", "OpenFIBSEM"))
