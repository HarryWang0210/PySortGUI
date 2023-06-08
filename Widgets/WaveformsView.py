from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *
from Widgets.WidgetsInterface import WidgetsInterface


class WaveformsView(QOpenGLWidget, WidgetsInterface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
