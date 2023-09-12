#!/usr/bin/env python

'''
Created on Dec 5, 2022

@author: harrywang
'''
from PyQt5 import QtGui, QtCore, QtWidgets
from Widgets.ChannelDetail import ChannelDetail
# from Widgets.WaveformsView import WaveformsView
from Widgets.WaveformsView_graph import WaveformsView

from Widgets.ClustersView import ClustersView
# from Widgets.TimelineView import TimelineView
from Widgets.TimelineView_graph import TimelineView
from Widgets.UnitOperateTools import UnitOperateTools

from Widgets.extend.ISIView import ISIView

organization = 'None'
application = 'MainWindowDocks'


class MainWindowDocks(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        if not hasattr(self, 'children_dict'):
            self.children_dict = dict()

    def setupUi(self):
        self.setDockOptions(QtWidgets.QMainWindow.AnimatedDocks |
                            QtWidgets.QMainWindow.AllowNestedDocks |
                            QtWidgets.QMainWindow.AllowTabbedDocks)
        self._init_docks()
        self._setup_app_menu()
        self._setup_status_bar()
        self.show()
        self.raise_()

    def _setup_status_bar(self):
        """Generate status bar."""
        status_bar = QtWidgets.QStatusBar()
        self.children_dict['status_bar'] = status_bar
        self.setStatusBar(status_bar)

        # status_bar.children_dict = dict()
        # status_bar.children_dict[
        #     'progress_bar'] = progress_bar = QtWidgets.QProgressBar()
        # status_bar.addPermanentWidget(progress_bar)

    def _setup_app_menu(self):
        """Generate menu bar."""
        self.menu_bar = QtWidgets.QMenuBar()
        self.children_dict['menu_bar'] = self.menu_bar
        self.setMenuBar(self.menu_bar)
        self._add_file_menu()
        self._add_edit_menu()
        self._add_view_menu()
        self._add_setting_menu()
        self._add_help_menu()

    def _add_file_menu(self):
        # File Menu
        FileMenu = self.menu_bar.addMenu('&File')
        FileMenu_dict = dict()

        OpenAction = QtWidgets.QAction('Open File...', self)
        OpenAction.setShortcut('Ctrl+O')
        OpenAction.setAutoRepeat(False)
        OpenAction.setStatusTip('Load H5 File')
        FileMenu.addAction(OpenAction)
        FileMenu_dict["Open"] = OpenAction

        SaveAction = QtWidgets.QAction('Save', self)
        SaveAction.setShortcut('Ctrl+S')
        SaveAction.setAutoRepeat(False)
        SaveAction.setStatusTip('Save current channel')
        FileMenu.addAction(SaveAction)
        FileMenu_dict["Save"] = SaveAction

        SaveAllAction = QtWidgets.QAction('Save All', self)
        SaveAllAction.setShortcut('Ctrl+Alt+S')
        SaveAllAction.setAutoRepeat(False)
        SaveAllAction.setStatusTip('Save all channels')
        FileMenu.addAction(SaveAllAction)
        FileMenu_dict["SaveAll"] = SaveAllAction

        ExitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), 'Exit', self)
        ExitAction.setShortcut('Ctrl+Q')
        ExitAction.setAutoRepeat(False)
        ExitAction.setStatusTip('Exit application')
        FileMenu.addAction(ExitAction)
        FileMenu_dict["Exit"] = ExitAction

        self.children_dict["FileMenu"] = FileMenu_dict

    def _add_edit_menu(self):
        # Edit Menu
        EditMenu = self.menu_bar.addMenu('&Edit')
        EditMenu_dict = dict()

        UndoAction = QtWidgets.QAction('Undo', self)
        UndoAction.setShortcut('Ctrl+Z')
        UndoAction.setAutoRepeat(False)
        EditMenu.addAction(UndoAction)
        EditMenu_dict["Undo"] = UndoAction

        RedoAction = QtWidgets.QAction('Redo', self)
        RedoAction.setShortcut('Ctrl+Y')
        RedoAction.setAutoRepeat(False)
        EditMenu.addAction(RedoAction)
        EditMenu_dict["Redo"] = RedoAction

        self.children_dict["EditMenu"] = EditMenu_dict

    def _add_view_menu(self):
        # View
        ViewMenu = self.menu_bar.addMenu('&View')
        ViewMenu_dict = dict()

        ChannelDetailAction = self.children_dict['ChannelDetail_dock'].toggleViewAction(
        )
        ViewMenu.addAction(ChannelDetailAction)
        ViewMenu_dict["ChannelDetail"] = ChannelDetailAction

        WaveformsViewAction = self.children_dict['WaveformsView_dock'].toggleViewAction(
        )
        ViewMenu.addAction(WaveformsViewAction)
        ViewMenu_dict["WaveformsView"] = WaveformsViewAction

        ClustersViewAction = self.children_dict['ClustersView_dock'].toggleViewAction(
        )
        ViewMenu.addAction(ClustersViewAction)
        ViewMenu_dict["ClustersView"] = ClustersViewAction

        TimelineViewAction = self.children_dict['TimelineView_dock'].toggleViewAction(
        )
        ViewMenu.addAction(TimelineViewAction)
        ViewMenu_dict["TimelineView"] = TimelineViewAction

        ISIViewAction = self.children_dict['ISIView_dock'].toggleViewAction()
        ViewMenu.addAction(ISIViewAction)
        ViewMenu_dict["ISIView"] = ISIViewAction

        self.children_dict["ViewMenu"] = ViewMenu_dict

    def _add_setting_menu(self):
        # Setting
        SettingMenu = self.menu_bar.addMenu('&Setting')
        SettingMenu_dict = dict()

        SaveLayoutAction = QtWidgets.QAction('Save layout', self)
        SettingMenu.addAction(SaveLayoutAction)
        SettingMenu_dict["SaveLayout"] = SaveLayoutAction

        RestoreLayoutAction = QtWidgets.QAction('Restore layout', self)
        SettingMenu.addAction(RestoreLayoutAction)
        SettingMenu_dict["RestoreLayout"] = RestoreLayoutAction

        self.children_dict["SettingMenu"] = SettingMenu_dict

    def _add_help_menu(self):
        # Help
        HelpAction = QtWidgets.QAction('&Help', self)
        self.menu_bar.addAction(HelpAction)
        self.children_dict["Help"] = HelpAction

    def _init_docks(self):
        self._generate_dock(ChannelDetail)
        self._generate_dock(WaveformsView)
        self._generate_dock(ISIView)
        self._generate_dock(ClustersView)

        self._generate_dock(TimelineView)
        self._generate_right_tool_widget(UnitOperateTools)

        geom = QtWidgets.QDesktopWidget().availableGeometry()
        self.splitDockWidget(self.children_dict["ChannelDetail_dock"],
                             self.children_dict["ISIView_dock"], QtCore.Qt.Horizontal)
        self.splitDockWidget(self.children_dict["ISIView_dock"],
                             self.children_dict["ClustersView_dock"], QtCore.Qt.Horizontal)
        self.resizeDocks([self.children_dict["ChannelDetail_dock"],
                          self.children_dict["ISIView_dock"],
                          self.children_dict["ClustersView_dock"],
                          self.children_dict["UnitOperateTools_dock"]],
                         [int(geom.width() / 6), int(geom.width() / 3),
                          int(geom.width() / 3), int(geom.width() / 6)], QtCore.Qt.Horizontal)
        self.tabifyDockWidget(
            self.children_dict["ISIView_dock"], self.children_dict["WaveformsView_dock"])
        self.resizeDocks([self.children_dict["ChannelDetail_dock"],
                          self.children_dict["TimelineView_dock"]],
                         [int(geom.bottom() / 3) * 2, int(geom.bottom() / 3)], QtCore.Qt.Vertical)

    def _generate_dock(self, widget_class=None, name=None, attr_name=None, **kwargs):
        """
        Generate the dock object.
        :param widget_class: class name of widget
        :param name: dock title
        :param attr_name: key save in children_dict
        """

        if widget_class is None:
            return
        obj = widget_class(**kwargs)

        if name is None:
            name = obj.window_title

        if attr_name is None:
            attr_name = widget_class.__name__

        dock = QtWidgets.QDockWidget(name, None)
        dock.setWidget(obj)
        dock.setObjectName(name)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                         QtWidgets.QDockWidget.DockWidgetMovable)

        self.children_dict[attr_name] = obj
        self.children_dict[attr_name + '_dock'] = dock

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _generate_right_tool_widget(self, widget_class=None, name=None, attr_name=None, **kwargs):
        """
        Generate the right side tool dock object.
        :param widget_class: class name of widget
        :param name: dock title
        :param attr_name: key save in children_dict
        """

        if widget_class is None:
            return
        obj = widget_class(**kwargs)

        if name is None:
            name = obj.window_title

        if attr_name is None:
            attr_name = widget_class.__name__

        dock = QtWidgets.QDockWidget(name, None)
        dock.setWidget(obj)
        dock.setObjectName(name)
        dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        self.children_dict[attr_name] = obj
        self.children_dict[attr_name + '_dock'] = dock
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
