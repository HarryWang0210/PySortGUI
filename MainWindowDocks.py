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

        self.setCorner(QtCore.Qt.TopRightCorner,
                       QtCore.Qt.RightDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomRightCorner,
                       QtCore.Qt.RightDockWidgetArea)
        self.setCorner(QtCore.Qt.TopLeftCorner,
                       QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomLeftCorner,
                       QtCore.Qt.BottomDockWidgetArea)

        if not hasattr(self, 'settings'):
            self.settings = QtCore.QSettings(organization, application)

        self.show()
        self.raise_()

    def _setup_status_bar(self):
        """Generate status bar."""
        self.children_dict['status_bar'] = QtWidgets.QStatusBar()
        status_bar = self.children_dict['status_bar']
        self.setStatusBar(status_bar)

        # status_bar.children_dict = dict()
        # status_bar.children_dict[
        #    'progress_bar'] = progress_bar = QtGui.QProgressBar()
        # status_bar.addWidget(progress_bar, 1)

    def _setup_app_menu(self):
        """Generate menu bar."""
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
        OpenAction.triggered.connect(self.close)

        FileMenu = MenuBar.addMenu('&File')
        FileMenu.addAction(OpenAction)
        FileMenu.addAction(ExitAction)

        EditMenu = MenuBar.addMenu('&Edit')

        # utilities = MenuBar.addMenu('&Utilities')
        # utilities.addAction('Refresh batch analysis').triggered.connect(self.refresh_batch)

        # file_utilities = MenuBar.addMenu('&File Utilities')
        # file_utilities.addAction('Sync Data...(rsync)').triggered.connect(self.sync_data)
        #    file_utilities.addAction(ExitAction)

        # ToolsMenu = MenuBar.addMenu('&Tools')
        # ToolsMenu.addAction(
        #    'Jupyter Console').triggered.connect(self.add_widget)
        # ToolsMenu.addAction(ExitAction)

        ViewMenu = MenuBar.addMenu('&View')

        SettingMenu = MenuBar.addMenu('&Setting')
        SettingMenu.addAction(
            'Save layout').triggered.connect(self.save_layout)
        SettingMenu.addAction('Restore layout').triggered.connect(
            self.restore_layout)

        HelpMenu = MenuBar.addMenu('&Help')

    def generate_dock(self, widget_class=None, name=None, attr_name=None, position=None, **kwargs):
        """
        Generate the dock object.
        :param widget_class: class name of widget
        :param name: dock title
        :param attr_name: key save in children_dict
        :param position: default position of the  dock
        """

        if widget_class is None:
            return

        if name is None:
            name = widget_class.__name__

        if attr_name is None:
            attr_name = widget_class.__name__

        obj = widget_class(**kwargs)
        dock = QtWidgets.QDockWidget(name, None)
        dock.setWidget(obj)
        dock.setObjectName(name)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable)

        self.children_dict[attr_name] = obj
        self.children_dict[attr_name + '_dock'] = dock

        if position:
            self.addDockWidget(position, dock)

    def generate_right_tool_widget(self, widget_class=None, name=None, attr_name=None, **kwargs):
        """
        Generate the right side tool dock object.
        :param widget_class: class name of widget
        :param name: dock title
        :param attr_name: key save in children_dict
        :param position: default position of the  dock
        """

        if widget_class is None:
            return

        if name is None:
            name = widget_class.__name__

        if attr_name is None:
            attr_name = widget_class.__name__

        obj = widget_class(**kwargs)
        dock = QtWidgets.QDockWidget(name, None)
        dock.setWidget(obj)
        dock.setObjectName(name)
        dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        self.children_dict[attr_name] = obj
        self.children_dict[attr_name + '_dock'] = dock
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def save_layout(self):
        """Save the layout changes."""
        pass

    def restore_layout(self):
        """Rstore_ the layout changes."""
        pass

    def closeEvent(self, event):
        """Close app."""
        super().closeEvent(event)
