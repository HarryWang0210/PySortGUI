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
        if not hasattr(self, 'settings'):
            self.settings = QtCore.QSettings(organization, application)

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
        #     'progress_bar'] = progress_bar = QtWidgets.QProgressBar()
        # status_bar.addPermanentWidget(progress_bar)

    def _setup_app_menu(self):
        """Generate menu bar."""
        self.children_dict['menu_bar'] = QtWidgets.QMenuBar()
        MenuBar = self.children_dict['menu_bar']
        self.setMenuBar(MenuBar)

        # Open
        OpenAction = QtWidgets.QAction('&Open File...', self)
        OpenAction.setShortcut('Ctrl+O')
        OpenAction.setStatusTip('Load H5 File')
        OpenAction.triggered.connect(self.open_file)

        SaveAction = QtWidgets.QAction('&Save', self)
        SaveAction.setShortcut('Ctrl+S')
        SaveAction.setStatusTip('Save current channel')
        SaveAction.triggered.connect(self.save_channel)

        SaveAllAction = QtWidgets.QAction('&Save All', self)
        SaveAllAction.setShortcut('Ctrl+Alt+S')
        SaveAllAction.setStatusTip('Save all channels')
        SaveAllAction.triggered.connect(self.save_all)

        ExitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        ExitAction.setShortcut('Ctrl+Q')
        ExitAction.setStatusTip('Exit application')
        ExitAction.triggered.connect(self.close)

        FileMenu = MenuBar.addMenu('&File')
        FileMenu.addAction(OpenAction)
        FileMenu.addAction(SaveAction)
        FileMenu.addAction(SaveAllAction)
        FileMenu.addAction(ExitAction)

        # Edit
        UndoAction = QtWidgets.QAction('&Undo', self)
        UndoAction.setShortcut('Ctrl+Z')
        UndoAction.triggered.connect(self.undo)

        RedoAction = QtWidgets.QAction('&Redo', self)
        RedoAction.setShortcut('Ctrl+Y')
        RedoAction.triggered.connect(self.redo)

        EditMenu = MenuBar.addMenu('&Edit')
        EditMenu.addAction(UndoAction)
        EditMenu.addAction(RedoAction)

        # utilities = MenuBar.addMenu('&Utilities')
        # utilities.addAction('Refresh batch analysis').triggered.connect(self.refresh_batch)

        # file_utilities = MenuBar.addMenu('&File Utilities')
        # file_utilities.addAction('Sync Data...(rsync)').triggered.connect(self.sync_data)
        #    file_utilities.addAction(ExitAction)

        # ToolsMenu = MenuBar.addMenu('&Tools')
        # ToolsMenu.addAction(
        #    'Jupyter Console').triggered.connect(self.add_widget)
        # ToolsMenu.addAction(ExitAction)

        # View
        ViewMenu = MenuBar.addMenu('&View')
        ChannelDetailAction = QtWidgets.QAction('&Channel Detail', self)
        ChannelDetailAction.setCheckable(True)
        ChannelDetailAction.triggered.connect(self.ViewTest)
        ViewMenu.addAction(ChannelDetailAction)
        WaveformsViewAction = QtWidgets.QAction('&Waveforms View', self)
        WaveformsViewAction.setCheckable(True)
        WaveformsViewAction.triggered.connect(self.ViewTest)
        ViewMenu.addAction(WaveformsViewAction)
        ClustersViewAction = QtWidgets.QAction('&Clusters View', self)
        ClustersViewAction.setCheckable(True)
        ChannelDetailAction.triggered.connect(self.ViewTest)
        ViewMenu.addAction(ClustersViewAction)
        TimelineViewAction = QtWidgets.QAction('&Timeline View', self)
        TimelineViewAction.setCheckable(True)
        TimelineViewAction.triggered.connect(self.ViewTest)
        ViewMenu.addAction(TimelineViewAction)

        # Setting
        SettingMenu = MenuBar.addMenu('&Setting')
        SettingMenu.addAction(
            'Save layout').triggered.connect(self.save_layout)
        SettingMenu.addAction('Restore layout').triggered.connect(
            self.restore_layout)

        # Help
        HelpMenu = MenuBar.addMenu('&Help')

    def generate_dock(self, widget_class=None, name=None, attr_name=None, **kwargs):
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
        obj.setMinimumWidth(400)
        obj.setMinimumHeight(50)

        dock = QtWidgets.QDockWidget(name, None)
        dock.setWidget(obj)
        dock.setObjectName(name)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable)

        self.children_dict[attr_name] = obj
        self.children_dict[attr_name + '_dock'] = dock

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

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

    def open_file(self):
        """Open file manager and load selected file."""
        pass

    def save_channel(self):
        """Save single channel."""
        pass

    def save_all(self):
        """Save all channels."""
        pass

    def undo(self):
        """Undo the last change."""
        pass

    def redo(self):
        """Redo the change."""
        pass

    def ViewTest(self):
        """Temporary function"""
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

    def closeEvent(self, event):
        """Close app."""
        super().closeEvent(event)
