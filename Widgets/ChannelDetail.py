from UI.ChannelDetail_ui import Ui_ChannelDetail
from PyQt5 import QtCore, QtGui, QtWidgets


class ChannelDetail(QtWidgets.QWidget, Ui_ChannelDetail):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Channel Detail"
        self.setupUi(self)
