#!/usr/bin/env python

'''
Created on Dec 5, 2022

@author: harry.wang0210
'''
# This Python file uses the following encoding: utf-8
import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtWidgets
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py

# Load Widgets
from MainWindowDocks import MainWindowDocks
from TimelineView import GLWidget
from UnitSelection import UnitSelection
from ChannelDetail import ChannelDetail
logger = logging.getLogger(__name__)

organization = 'NYCU'
application = 'SpikeSorter2'


class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''

    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg

    def __str__(self):
        return self.msg

    def __unicode__(self):
        return self.msg


class QSSLoader:
    """ Load qss file to change style """

    def __init__(self):
        pass

    @staticmethod
    def read_qss_file(qss_file_name):
        with open(qss_file_name, 'r',  encoding='UTF-8') as file:
            return file.read()


class SpikeSorter2(MainWindowDocks):
    def __init__(self, parent=None):
        super().__init__(parent)
        # if not hasattr(self, 'settings'):
        #     self.settings = QtCore.QSettings(organization, application)

        geom = QtWidgets.QDesktopWidget().availableGeometry()
        self.resize(int(geom.width()), int(geom.bottom()))
        self.move(geom.topLeft().x(), geom.topLeft().y())
        self.load_style()

        self.setup()
        # self.restore_layout()

    def setup(self):
        self.generate_dock(ChannelDetail, attr_name='channel_detail')
        self.generate_dock(GLWidget, attr_name='B')
        self.generate_dock(GLWidget, attr_name='C')

        self.generate_dock(GLWidget, attr_name='D')

        self.generate_right_tool_widget(
            UnitSelection, attr_name='unit_selection')

        self.resizeDocks([self.children_dict["channel_detail_dock"],
                          self.children_dict["unit_selection_dock"]],
                         [2, 1], QtCore.Qt.Horizontal)

        self.splitDockWidget(self.children_dict["channel_detail_dock"],
                             self.children_dict["B_dock"], QtCore.Qt.Horizontal)
        self.splitDockWidget(self.children_dict["B_dock"],
                             self.children_dict["C_dock"], QtCore.Qt.Horizontal)

    def load_style(self):
        """
        Load the QSS format app style.
        """
        # style_file = '/home/user/qt-material/examples/exporter/dark_teal.qss'
        style_file = 'UI/style.qss'
        style_sheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(style_sheet)

    def closeEvent(self, event):
        """Close app."""
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpikeSorter2()
    window.show()

    sys.exit(app.exec())
