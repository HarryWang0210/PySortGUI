from UI.UnitOperateToolsUIv2_ui import Ui_UnitOperateTools
from PyQt5 import QtCore, QtGui, QtWidgets


class UnitOperateTools(QtWidgets.QWidget, Ui_UnitOperateTools):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Unit Operate Tools"
        self.setupUi(self)
