from UI.Unit_selectionUIv2_ui import Ui_UnitOperateTools
from PyQt5 import QtCore, QtGui, QtWidgets


class UnitSelection(QtWidgets.QWidget, Ui_UnitOperateTools):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)
