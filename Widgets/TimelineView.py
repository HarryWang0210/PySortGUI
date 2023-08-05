import sys
from UI.TimelineView_ui import Ui_TimelineView
from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, QOpenGLContext
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
import numpy as np
from DataStructure.data import SpikeSorterData


class TimelineView(QtWidgets.QWidget, Ui_TimelineView):
    signal_data_file_name_changed = QtCore.pyqtSignal(str)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Timeline View"
        self.setupUi(self)
        self.openGLWidget = TimelineView_widget(self)
        self.openglLayout.addWidget(self.openGLWidget)

    def spike_chan_changed(self, data):
        self.openGLWidget.get_info(data)
        self.openGLWidget.raw_visible = True
        self.openGLWidget.update()
        # self.opengl_widget.makeCurrent()
        # self.opengl_widget.paintGL()
        # self.opengl_widget.swapBuffers()


class TimelineView_widget(QOpenGLWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.raw_visible = False

    def initializeGL(self):
        glClearColor(0.35, 0.35, 0.35, 0.8)  # background color

    def get_info(self, data):
        self.data = data
        # initial scale data between -1 and 1 to fit the window
        self.data_scale = np.median(np.abs(self.data)) * 10

        # self.data = np.random.random(100)
        self.thr = - np.median(np.abs(self.data))
        self.num_data_show = 1000  # initial number of data points show in window
        self.offset = 0
        self.scale_x = 1.0
        self.scale_y = 1.0

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.raw_visible:

            glColor3f(1.0, 1.0, 1.0)
            data_scale = self.data_scale / self.scale_y
            num_data_show = int(self.num_data_show / self.scale_x)
            line_strip_data = []
            for i, y in enumerate(self.data[self.offset: self.offset + num_data_show + 1]):
                x = -1.0 + i * 2 / num_data_show

                line_strip_data.extend([x, y / data_scale])
            self.draw_line_strip(line_strip_data)
            # glPopMatrix()
            glViewport(0, 0, self.width(), self.height())

            # threshold
            thr = self.thr / data_scale
            glColor3f(0.0, 1.0, 0.0)
            # self.shader_program.setUniformValue(color_location, 0.0, 1.0, 0.0)
            self.draw_line_strip([-1.0, thr, 1.0, thr], True)

            # self.shader_program.release()

    def draw_line_strip(self, data, is_line=False):
        # 畫出範圍內的圖

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, data)
        if is_line:
            glDrawArrays(GL_LINES, 0, len(data) // 2)  # GL_LINES只能直線
        else:
            glDrawArrays(GL_LINE_STRIP, 0, len(data) // 2)  # GL_LINE_STRIP可以折線
        glDisableClientState(GL_VERTEX_ARRAY)

    def wheelEvent(self, wheel_event):
        # wheel_event.pixelDelta 只能用在MacOS觸控板
        self.offset += int(wheel_event.angleDelta().y() /
                           120 * int(self.num_data_show / self.scale_x) / 10)
        if self.offset <= 0:
            self.offset = 0
        self.update()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == Qt.Key_Left:
            self.offset += - int(self.num_data_show / self.scale_x / 100)
            if self.offset <= 0:
                self.offset = 0
            self.update()
        elif event.key() == Qt.Key_Right:
            self.offset += int(self.num_data_show / self.scale_x / 100)
            self.update()

        elif event.key() == Qt.Key_D:
            self.scale_x *= 1.1
            self.update()

        elif event.key() == Qt.Key_A:
            self.scale_x /= 1.1
            self.update()

        elif event.key() == Qt.Key_Up:
            self.scale_y *= 1.1
            self.update()

        elif event.key() == Qt.Key_Down:
            self.scale_y /= 1.1
            self.update()


# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.gl_widget = TimelineView(self)
#         layout = QHBoxLayout()
#         layout.addWidget(self.gl_widget)
#         self.setLayout(layout)
#         self.setWindowTitle("Timeline View")
#         self.setGeometry(100, 100, 800, 600)

#     # def keyPressEvent(self, event):
#     #     self.gl_widget.keyPressEvent(event)


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     # context = QOpenGLContext()
#     widget = MainWindow()
#     widget.show()
#     sys.exit(app.exec_())
