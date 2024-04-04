from OpenGL.GL import *
from PyQt5.QtWidgets import QOpenGLWidget


class ISIView(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "ISI View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
