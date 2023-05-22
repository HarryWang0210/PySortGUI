from UI.Unit_selectionUIv2_ui import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets


class UnitSelection(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
