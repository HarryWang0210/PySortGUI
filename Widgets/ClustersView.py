from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *


class ClustersView(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Clusters View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
