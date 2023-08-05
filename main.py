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
from DataStructure.data import SpikeSorterData
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
        if not hasattr(self, 'settings'):
            self.settings = QtCore.QSettings(organization, application)

        geom = QtWidgets.QDesktopWidget().availableGeometry()
        self.resize(int(geom.width()), int(geom.bottom()))
        self.move(geom.topLeft().x(), geom.topLeft().y())
        self.load_style()
        self.setupUi()
        self.connect_menu_actions()
        self.connect_widgets()
        self.setWindowTitle("SpikeSorterGL")
        self.restore_layout()

    def load_style(self):
        """
        Load the QSS format app style.
        """
        # style_file = '/home/user/qt-material/examples/exporter/dark_teal.qss'
        style_file = 'UI/style.qss'
        style_sheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(style_sheet)

    def connect_menu_actions(self):
        """Connect slots to actions on menu bar."""

        self.children_dict['FileMenu']["Open"].triggered.connect(
            self.open_file, QtCore.Qt.UniqueConnection)
        self.children_dict['FileMenu']["Save"].triggered.connect(
            self.save_channel)
        self.children_dict['FileMenu']["SaveAll"].triggered.connect(
            self.save_all)
        self.children_dict['FileMenu']["Exit"].triggered.connect(
            self.close)

        self.children_dict['EditMenu']["Undo"].triggered.connect(
            self.undo)
        self.children_dict['EditMenu']["Redo"].triggered.connect(
            self.redo)

        self.children_dict['SettingMenu']["SaveLayout"].triggered.connect(
            self.save_layout)
        self.children_dict['SettingMenu']["RestoreLayout"].triggered.connect(
            self.restore_layout)

        self.children_dict['Help'].triggered.connect(self.help)

    def connect_widgets(self):
        self.children_dict["ChannelDetail"].signal_spike_chan_changed.connect(
            self.children_dict["TimelineView"].spike_chan_changed)

    def open_file(self):
        """Open file manager and load selected file."""
        self.children_dict["ChannelDetail"].open_file()

    def save_channel(self):
        """Save single channel."""
        print("save_channel")
        pass

    def save_all(self):
        """Save all channels."""
        print("save_all")
        self.save_channel()
        pass

    def undo(self):
        """Undo the last change."""
        print("undo")
        pass

    def redo(self):
        """Redo the change."""
        print("redo")
        pass

    def save_layout(self, id=0):
        """Save the layout changes."""
        self.settings.setValue(
            "geometry_{:d}".format(id), self.saveGeometry())
        self.settings.setValue(
            "windowState_{:d}".format(id), self.saveState())

    def restore_layout(self, id=0):
        """Rstore_ the layout changes."""
        if not self.settings.value("geometry_{:d}".format(id)) is None:
            self.restoreGeometry(
                self.settings.value("geometry_{:d}".format(id)))
        # else:
        #     geom = QtGui.QDesktopWidget().availableGeometry()
        #     self.resize(geom.width(), geom.bottom())
        #     self.move(geom.topLeft().x(), geom.topLeft().y())

        if not self.settings.value("windowState_{:d}".format(id)) is None:
            self.restoreState(
                self.settings.value("windowState_{:d}".format(id)))

    def help(self):
        """Show help documentation."""
        print("help")

    def closeEvent(self, event):
        """Close app."""
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpikeSorter2()
    window.show()

    sys.exit(app.exec())
