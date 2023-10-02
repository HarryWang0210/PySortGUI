from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from OpenGL.GL import *
import numpy as np
import seaborn as sns

from DataStructure.data import SpikeSorterData
from legacy.GLWidget import GLWidget
from Widgets.WidgetsInterface import WidgetsInterface


class WaveformsView(GLWidget, WidgetsInterface):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.show_thr = False
        self.has_thr = False
        self.show_waveforms = False
        self.has_waveforms = False

        self.data = None
        self.thr = 0.0
        self.waveforms = None
        self.color_palette = sns.color_palette(None, 64)
        self.data_scale = 1.0

    def data_file_name_changed(self, data):
        self.data = data
        self.visible = False
        self.update()

    def spike_chan_changed(self, meta_data):
        self.get_thr(meta_data["Threshold"])
        self.get_spikes(
            self.data.get_spikes(int(meta_data["ID"]), meta_data["Label"]))
        self.visible = True
        self.update()

    def get_thr(self, thr):
        try:
            self.thr = float(thr)
            self.has_thr = True
            self.show_thr = True
        except:
            self.thr = 0.0
            self.has_thr = False
            self.show_thr = False

    def get_spikes(self, spikes):
        if spikes["units_info"] is None:
            self.has_spikes = False
            self.spikes = None
            self.data_scale = 1.0
            self.has_waveforms, self.show_waveforms = False, False
        else:
            self.has_spikes = True
            self.spikes = spikes
            self.data_scale = np.max(np.abs(spikes["waveforms"]))
            self.has_waveforms, self.show_waveforms = True, True

    def waveforms_to_vertices_data(self, waveforms, units_id):
        # self.spikes["waveforms"].shape[1]
        position = []
        xticks = np.linspace(-1.0, 1.0, num=waveforms.shape[1])
        for wav in waveforms:
            position.append(np.vstack((xticks,
                                       wav / self.data_scale,
                                       np.zeros(waveforms.shape[1]))).T)
        color = []
        for c in units_id:
            color.append(
                np.tile(self.color_palette[int(c)], (waveforms.shape[1], 1)))

        position = np.vstack(position)
        color = np.vstack(color)
        vertices_data = np.hstack(
            (position, color)).flatten().astype(np.float32)
        return vertices_data

    def initializeGL(self):
        super().initializeGL()
        self.create_vbo(n=1)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.visible:
            # 使用修改后的着色器程序
            self.shader_program.bind()
            self.model_matrix.setToIdentity()
            # self.model_matrix.scale(self.x_scale, self.y_scale)
            self.shader_program.setUniformValue(
                "model_matrix", self.model_matrix)

            if self.has_thr and self.show_thr:
                self.draw_thr()

            if self.has_waveforms and self.show_waveforms:
                self.draw_waveforms()

            # 解綁
            self.shader_program.release()

    def draw_thr(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        thr = np.array([[-1.0, self.thr / self.data_scale, 0.0],
                        [1.0, self.thr / self.data_scale, 0.0]])
        thr_color = np.array([1.0, 1.0, 1.0])
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

    def draw_waveforms(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        vertices_data = self.waveforms_to_vertices_data(
            self.spikes["waveforms"], self.spikes["units_id"])

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

        for i in range(len(vertices_data) // (6 * self.spikes["waveforms"].shape[1])):
            glDrawArrays(
                GL_LINE_STRIP, 0 + i * self.spikes["waveforms"].shape[1], self.spikes["waveforms"].shape[1])

        self.shader_program.disableAttributeArray(position)
        self.shader_program.disableAttributeArray(color)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

# class WaveformsViewGL(GLWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setMinimumWidth(100)
#         self.setMinimumHeight(100)
