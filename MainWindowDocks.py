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

        # self.setCorner(QtCore.Qt.TopRightCorner,
        #                QtCore.Qt.RightDockWidgetArea)
        # self.setCorner(QtCore.Qt.BottomRightCorner,
        #                QtCore.Qt.RightDockWidgetArea)
        # self.setCorner(QtCore.Qt.TopLeftCorner,
        #                QtCore.Qt.LeftDockWidgetArea)
        # self.setCorner(QtCore.Qt.BottomLeftCorner,
        #                QtCore.Qt.BottomDockWidgetArea)

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

        # File Menu
        OpenAction = QtWidgets.QAction('&Open File...', self)
        OpenAction.setShortcut('Ctrl+O')
        OpenAction.setAutoRepeat(False)
        OpenAction.setStatusTip('Load H5 File')
        OpenAction.triggered.connect(self.open_file)

        SaveAction = QtWidgets.QAction('&Save', self)
        SaveAction.setShortcut('Ctrl+S')
        SaveAction.setAutoRepeat(False)
        SaveAction.setStatusTip('Save current channel')
        SaveAction.triggered.connect(self.save_channel)

        SaveAllAction = QtWidgets.QAction('&Save All', self)
        SaveAllAction.setShortcut('Ctrl+Alt+S')
        SaveAllAction.setAutoRepeat(False)
        SaveAllAction.setStatusTip('Save all channels')
        SaveAllAction.triggered.connect(self.save_all)

        ExitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        ExitAction.setShortcut('Ctrl+Q')
        ExitAction.setAutoRepeat(False)
        ExitAction.setStatusTip('Exit application')
        ExitAction.triggered.connect(self.close)

        FileMenu = MenuBar.addMenu('&File')
        FileMenu.addAction(OpenAction)
        FileMenu.addAction(SaveAction)
        FileMenu.addAction(SaveAllAction)
        FileMenu.addAction(ExitAction)

        # Edit Menu
        UndoAction = QtWidgets.QAction('&Undo', self)
        UndoAction.setShortcut('Ctrl+Z')
        UndoAction.setAutoRepeat(False)
        UndoAction.triggered.connect(self.undo)

        RedoAction = QtWidgets.QAction('&Redo', self)
        RedoAction.setShortcut('Ctrl+Y')
        RedoAction.setAutoRepeat(False)
        RedoAction.triggered.connect(self.redo)

        EditMenu = MenuBar.addMenu('&Edit')
        EditMenu.addAction(UndoAction)
        EditMenu.addAction(RedoAction)

        # View
        ViewMenu = MenuBar.addMenu('&View')
        ChannelDetailAction = QtWidgets.QAction('&Channel Detail', self)
        ChannelDetailAction.setCheckable(True)
        ChannelDetailAction.toggled[bool].connect(
            self.control_view, QtCore.Qt.UniqueConnection)
        ViewMenu.addAction(ChannelDetailAction)
        WaveformsViewAction = QtWidgets.QAction('&Waveforms View', self)
        WaveformsViewAction.setCheckable(True)
        WaveformsViewAction.toggled[bool].connect(
            self.control_view, QtCore.Qt.UniqueConnection)
        ViewMenu.addAction(WaveformsViewAction)
        ClustersViewAction = QtWidgets.QAction('&Clusters View', self)
        ClustersViewAction.setCheckable(True)
        ClustersViewAction.toggled[bool].connect(
            self.control_view, QtCore.Qt.UniqueConnection)
        ViewMenu.addAction(ClustersViewAction)
        TimelineViewAction = QtWidgets.QAction('&Timeline View', self)
        TimelineViewAction.setCheckable(True)
        TimelineViewAction.toggled[bool].connect(
            self.control_view, QtCore.Qt.UniqueConnection)
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
        name = attr_name
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
        print("undo")
        pass

    def redo(self):
        """Redo the change."""
        pass

    def control_view(self, checked=False):
        """Control View widget show or close."""
        print(checked)
        print(self.sender().text())
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
