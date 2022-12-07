#!/usr/bin/env python

'''
Created on Dec 5, 2022

@author: harrywang
'''
from PyQt5 import QtGui, QtCore, QtWidgets

organization = 'None'
application = 'Ui_MainWindow'

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup()
        # self._setup_app_menu()
        # self._setup_status_bar()

    def _setup(self):

        if not hasattr(self, 'children_dict'):
            self.children_dict = dict()

        self.setDockOptions(QtWidgets.QMainWindow.AnimatedDocks |
                            QtWidgets.QMainWindow.AllowNestedDocks |
                            QtWidgets.QMainWindow.AllowTabbedDocks)

        if not hasattr(self, 'settings'):
            self.settings = QtCore.QSettings(organization, application)

        self.show()
        self.raise_()

        