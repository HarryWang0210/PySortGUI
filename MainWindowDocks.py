#!/usr/bin/env python

'''
Created on Dec 5, 2022

@author: harrywang
'''
from PyQt5 import QtGui, QtCore, QtWidgets

organization = 'None'
application = 'MainWindowDocks'

class MainWindowDocks(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._setup()
        self._setup_app_menu()
        self._setup_status_bar()

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

    # STATUS BAR -------------------------------------------------------------

    def _setup_status_bar(self):

        self.children_dict['status_bar'] = QtWidgets.QStatusBar()
        status_bar = self.children_dict['status_bar']
        self.setStatusBar(status_bar)
        status_bar.children_dict = dict()
        # status_bar.children_dict[
        #    'progress_bar'] = progress_bar = QtGui.QProgressBar()
        # status_bar.addWidget(progress_bar, 1)

    #------------------------------------------------------------------ APP MENU
    def _setup_app_menu(self):

        self.children_dict['menu_bar'] = QtWidgets.QMenuBar()
        MenuBar = self.children_dict['menu_bar']
        self.setMenuBar(MenuBar)

        # pdb.set_trace()
        ExitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Close', self)
        ExitAction.setShortcut('Ctrl+Q')
        ExitAction.setStatusTip('Exit application')
        ExitAction.triggered.connect(self.close)

        OpenAction = QtWidgets.QAction('&Open File...', self)
        OpenAction.setShortcut('Ctrl+O')
        OpenAction.setStatusTip('Load H5 File')
        # OpenAction.triggered.connect(self.close)

        FileMenu = MenuBar.addMenu('&File')
        FileMenu.addAction(OpenAction)
        FileMenu.addAction(ExitAction)
        
        # utilities = MenuBar.addMenu('&Utilities')
        # utilities.addAction('Refresh batch analysis').triggered.connect(self.refresh_batch)

        # file_utilities = MenuBar.addMenu('&File Utilities')
        # file_utilities.addAction('Sync Data...(rsync)').triggered.connect(self.sync_data)
        #    file_utilities.addAction(ExitAction)

        #ToolsMenu = MenuBar.addMenu('&Tools')
        # ToolsMenu.addAction(
        #    'Jupyter Console').triggered.connect(self.add_widget)
        # ToolsMenu.addAction(ExitAction)

        view_menu = MenuBar.addMenu('&View')
        # view_menu.addAction('Save layout').triggered.connect(self.save_layout)
        # view_menu.addAction('Restore layout').triggered.connect(
        #     self.restore_layout)

    def generate_dock(self, widget_class=None, name=None):
        pass

    def closeEvent(self, event):
        super(MainWindowDocks, self).closeEvent(event)