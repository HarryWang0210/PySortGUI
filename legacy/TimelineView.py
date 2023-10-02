from UI.TimelineView_ui import Ui_TimelineView
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from OpenGL.GL import *
import numpy as np
from DataStructure.data import SpikeSorterData
from legacy.GLWidget import GLWidget
import seaborn as sns


class TimelineView(QtWidgets.QWidget, Ui_TimelineView):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Timeline View"
        self.setupUi(self)
        self.openGLWidget = TimelineViewGL(self)
        self.openglLayout.addWidget(self.openGLWidget)
        self.data = None
        self.setup_connections()

    def setup_connections(self):
        self.thr_pushButton.clicked.connect(self.show_thr)
        self.events_pushButton.clicked.connect(self.show_events)
        self.spikes_pushButton.clicked.connect(self.show_spikes)
        # self.raw_pushButton.clicked.connect(self.show_raw)

    def data_file_name_changed(self, data):
        self.data = data
        self.openGLWidget.visible = False
        self.openGLWidget.update()

    def spike_chan_changed(self, meta_data):
        self.openGLWidget.get_raw(self.data.get_raw(int(meta_data["ID"])))
        self.openGLWidget.get_thr(meta_data["Threshold"])
        self.openGLWidget.get_spikes(
            self.data.get_spikes(int(meta_data["ID"]), meta_data["Label"]))
        self.openGLWidget.init_param()
        self.openGLWidget.visible = True
        self.openGLWidget.update()

    def show_thr(self, checked):
        self.openGLWidget.show_thr = checked
        self.openGLWidget.update()

    def show_events(self, checked):
        self.openGLWidget.show_events = checked
        self.openGLWidget.update()

    def show_spikes(self, checked):
        self.openGLWidget.show_spikes = checked
        self.openGLWidget.update()


class TimelineViewGL(GLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.MIN_DATA_SHOW = 100
        self.MAX_DATA_SHOW = 30000

        self.show_thr = False
        self.has_thr = False
        self.show_events = False
        self.has_events = False
        self.show_spikes = False
        self.has_spikes = False

        self.raw = None
        self.offset = 0
        self.data_scale = 1.0
        self.num_data_show = 1000  # initial number of data points show in window
        self.color_palette = sns.color_palette(None, 64)

    def get_raw(self, raw):
        self.raw = raw

    def get_thr(self, thr):
        try:
            self.thr = float(thr)
            self.has_thr = True
        except:
            self.thr = 0.0
            self.has_thr = False

    def get_events(self, events):
        self.events = events
        self.has_events = True

    def get_spikes(self, spikes):
        if spikes["units_info"] is None:
            self.has_spikes = False
            self.spikes = None
        else:
            self.has_spikes = True
            self.spikes = spikes

    def init_param(self):
        self.data_scale = np.median(np.abs(self.raw)) * 10
        self.num_data_show = 1000  # initial number of data points show in window

    def initializeGL(self):
        super().initializeGL()
        self.create_vbo(n=1)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.visible:
            # vertices_data = self.create_vertices_data()
            # 使用修改后的着色器程序
            self.shader_program.bind()
            self.model_matrix.setToIdentity()
            # self.model_matrix.scale(self.x_scale, self.y_scale)
            self.shader_program.setUniformValue(
                "model_matrix", self.model_matrix)

            self.draw_raw()

            if self.has_thr and self.show_thr:
                self.draw_thr()

            if self.has_events and self.show_events:
                self.draw_events()

            if self.has_spikes and self.show_spikes:
                self.draw_spikes()
            # 解綁
            self.shader_program.release()

    def draw_raw(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # check whether scale too small or too large
        lastpoint = self.offset + int(self.num_data_show)
        if lastpoint > np.min([len(self.raw), self.MAX_DATA_SHOW]):
            self.num_data_show = np.min(
                [len(self.raw) - self.offset, self.MAX_DATA_SHOW])
        elif lastpoint < self.MIN_DATA_SHOW:
            self.num_data_show = self.MIN_DATA_SHOW

        raw = np.vstack(
            (np.linspace(-1.0, 1.0, num=int(self.num_data_show)),
             self.raw[self.offset: self.offset +
                      int(self.num_data_show)] / self.data_scale,
             np.zeros(int(self.num_data_show)))
        ).T

        raw_color = np.array([1.0, 1.0, 1.0])
        vertices_data = np.hstack(
            (raw, np.tile(raw_color, (raw.shape[0], 1)))).flatten().astype(np.float32)

        # 将顶点数据上传到VBO
        glBufferData(GL_ARRAY_BUFFER, vertices_data.nbytes,
                     vertices_data, GL_STATIC_DRAW)

        # 设置顶点属性指针
        position = self.shader_program.attributeLocation("position")
        color = self.shader_program.attributeLocation("color")
        self.shader_program.enableAttributeArray(position)
        self.shader_program.enableAttributeArray(color)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE,
                              6 * 4, ctypes.c_void_p(3 * 4))

        glDrawArrays(GL_LINE_STRIP, 0, int(self.num_data_show))

        self.shader_program.disableAttributeArray(position)
        self.shader_program.disableAttributeArray(color)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_thr(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        thr = np.array([[-1.0, self.thr / self.data_scale, 0.0],
                        [1.0, self.thr / self.data_scale, 0.0]])

        thr_color = np.array([0.0, 1.0, 0.0])
        vertices_data = np.hstack(
            (thr, np.tile(thr_color, (thr.shape[0], 1)))).flatten().astype(np.float32)

        # 将顶点数据上传到VBO
        glBufferData(GL_ARRAY_BUFFER, vertices_data.nbytes,
                     vertices_data, GL_STATIC_DRAW)

        # 设置顶点属性指针
        position = self.shader_program.attributeLocation("position")
        color = self.shader_program.attributeLocation("color")
        self.shader_program.enableAttributeArray(position)
        self.shader_program.enableAttributeArray(color)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE,
                              6 * 4, ctypes.c_void_p(3 * 4))

        glDrawArrays(GL_LINES, 0, 2)

        self.shader_program.disableAttributeArray(position)
        self.shader_program.disableAttributeArray(color)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw_events(self):
        """TODO"""
        pass

    def draw_spikes(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        vertices_data = self.ts_to_vertices_data(
            self.spikes["timestamps"], self.spikes["units_id"], "spikes")

        # 将顶点数据上传到VBO
        glBufferData(GL_ARRAY_BUFFER, vertices_data.nbytes,
                     vertices_data, GL_STATIC_DRAW)

        # 设置顶点属性指针
        position = self.shader_program.attributeLocation("position")
        color = self.shader_program.attributeLocation("color")
        self.shader_program.enableAttributeArray(position)
        self.shader_program.enableAttributeArray(color)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE,
                              6 * 4, ctypes.c_void_p(3 * 4))

        for i in range(len(vertices_data) // (6 * 2)):
            glDrawArrays(GL_LINES, 0 + i * 2, 2)

        self.shader_program.disableAttributeArray(position)
        self.shader_program.disableAttributeArray(color)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def ts_to_vertices_data(self, ts, color_id, data_type):
        lastpoint = self.offset + int(self.num_data_show)
        ts_mask = np.all([ts >= self.offset, ts < lastpoint], axis=0)
        xticks = np.linspace(-1.0, 1.0, num=int(self.num_data_show))

        position = []
        if data_type == "spikes":
            for x in ts[ts_mask]:
                position.append(np.array([[xticks[x - self.offset], -1.0, 0.0],
                                          [xticks[x - self.offset], self.thr / self.data_scale, 0.0]]))
        color = []
        for c in color_id[ts_mask]:
            color.append(
                np.array([self.color_palette[int(c)],
                          self.color_palette[int(c)]]))
        position = np.vstack(position)
        color = np.vstack(color)
        vertices_data = np.hstack(
            (position, color)).flatten().astype(np.float32)
        return vertices_data

    def wheelEvent(self, wheel_event):
        modifiers = QApplication.keyboardModifiers()

        if (modifiers == Qt.ShiftModifier):
            self.x_scale = 1 + wheel_event.angleDelta().y() / 1000
            self.num_data_show /= self.x_scale

        elif (modifiers == (Qt.AltModifier | Qt.ShiftModifier)):
            self.y_scale = 1 + wheel_event.angleDelta().x() / 1000
            self.data_scale /= self.y_scale

        else:
            # wheel_event.pixelDelta 只能用在MacOS觸控板
            self.offset -= int(wheel_event.angleDelta().y() /
                               120 * int(self.num_data_show) / 10)
            if self.offset <= 0:
                self.offset = 0
        self.update()
