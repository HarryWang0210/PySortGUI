import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import seaborn as sns
from OpenGL.GL import *  # noqa
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (QColor, QGuiApplication, QMatrix4x4, QOpenGLBuffer,
                         QOpenGLShader, QOpenGLShaderProgram, QVector3D)
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget

from pysortgui.DataStructure.datav3 import (ContinuousData, DiscreteData,
                                            SpikeSorterData)
from pysortgui.Widgets.WidgetsInterface import WidgetsInterface

logger = logging.getLogger(__name__)


class WaveformsView(gl.GLViewWidget, WidgetsInterface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.window_title = "Waveforms View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.GLOBAL_WAVS_LIMIT = 100000

        self.redraw_wavs = False

        self.data_object = None  # SpikeSorterData object
        # self.data_scale = 1.0
        # self.spikes = None
        # self.has_spikes = False
        self.thr = 0.0
        # self.has_thr = False
        self.color_palette_list = sns.color_palette('bright', 64)
        self.plot_visible = False
        self.widget_visible = False

        self._x_boundary: tuple[int, int] = (0, 0)
        self._x_range: tuple[int, int] = (0, 1)
        # self._y_boundary: tuple[int, int] = (0, 0)
        self._y_range: tuple[int, int] = (-1000, 1000)
        self._y_scale = 1

        self.current_wavs_mask = []
        self.current_showing_units = []

        self.current_spike_object = None
        self.manual_mode = False

        self.initPlotItem()

    def initPlotItem(self):
        """
        Initialize plotWidget and plotItems.
        """
        background_color = (0.35, 0.35, 0.35)
        background_color = QColor(*[int(c * 255) for c in background_color])
        self.setBackgroundColor(background_color)

        self.setCameraPosition(distance=50, azimuth=-90, elevation=90)  # 固定视角
        self.setCameraParams(fov=90)

        self.waveform_item = WaveformGLItem()
        self.waveform_item.setGLOptions('opaque')  # not to mix color
        self.addItem(self.waveform_item)

        self.thr_item = gl.GLLinePlotItem(width=2, mode='lines')
        self.thr_item.setVisible(False)
        self.addItem(self.thr_item)

        # self.select_point_item = SelectWaveformGLItem()
        self.select_point_item = gl.GLLinePlotItem(
            width=2, color='w', mode='line_strip')
        self.select_point_item.setGLOptions('opaque')  # not to mix color

        # pg.PlotCurveItem(
        #     pen=pg.mkPen('w', width=2), clickable=False)
        # self.select_point_item.setZValue(1)
        self.select_point_item.setVisible(False)
        self.addItem(self.select_point_item)

        # self.manual_curve_item = gl.GLLinePlotItem(width=2, color='r', mode='lines')
        # # self.manual_curve_item.setZValue(1)
        # self.manual_curve_item.setVisible(False)
        # self.addItem(self.manual_curve_item)

        self.pan = self.disable_action
        self.orbit = self.disable_action
        self.mouseDragEvent = self.disable_action
        self.wheelEvent = self.graphMouseWheelEvent
        self.mousePressEvent = self.disable_action
        self.mouseMoveEvent = self.disable_action
        self.mouseReleaseEvent = self.disable_action

        # self.plot_item.getViewBox().wheelEvent = self.graphMouseWheelEvent
        # self.plot_item.scene().mousePressEvent = self.graphMousePressEvent
        # self.plot_item.scene().mouseMoveEvent = self.graphMouseMoveEvent
        # self.plot_item.scene().mouseReleaseEvent = self.graphMouseReleaseEvent

    def widgetVisibilityChanged(self, visible: bool):
        self.widget_visible = visible
        self.updatePlot()

    def data_file_name_changed(self, data):
        self.data_object = data
        self.plot_visible = False
        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.current_spike_object = new_spike_object

        # self.has_spikes = True
        self.redraw_wavs = True
        self.plot_visible = True
        if self.current_spike_object is None:
            self.plot_visible = False
            self.updatePlot()
            return
        data_scale = np.max(np.abs(self.current_spike_object.waveforms)) / 2
        self._x_range = (0, self.current_spike_object.waveforms.shape[1] - 1)
        self._y_scale = (
            (self._x_range[1] - self._x_range[0]) / 2) / data_scale
        self.setCameraPosition(pos=pg.Vector((self._x_range[1] - self._x_range[0]) / 2, 0, 0),
                               distance=(self._x_range[1] - self._x_range[0]) / 2)

    def showing_units_changed(self, showing_unit_IDs):
        self.current_showing_units = showing_unit_IDs
        self.current_wavs_mask = np.isin(self.current_spike_object.unit_IDs,
                                         self.current_showing_units)
        self.redraw_wavs = True
        self.updatePlot()

    def activate_manual_mode(self, state):
        self.manual_mode = state

    def select_point(self, data):
        selected, wav_index = data

        if selected:
            current_showing_data = self.current_spike_object._waveforms[self.current_wavs_mask]
            y = current_showing_data[wav_index, :]
            x = np.arange(len(y))
            z = np.ones(len(y)) * .01*2
            pos = np.vstack([x, y, z]).transpose()

            self.select_point_item.setData(pos=pos)

        if self.select_point_item.visible() == selected:
            return

        self.select_point_item.setVisible(selected)

    def updatePlot(self):
        visible = self.plot_visible and self.widget_visible
        if visible and not self.current_spike_object is None:
            if self.redraw_wavs:
                # self.removeWaveformItems()
                self.drawWaveforms()
                self.redraw_wavs = False
            # if self.has_thr:
            self.drawThreshold()
        self.waveform_item.setVisible(visible)
        self.thr_item.setVisible(visible)

        # self.plot_item.getViewBox().setXRange(*self._x_range, padding=0)
        # self.plot_item.getViewBox().setYRange(*self._y_range, padding=0)

        # for waveforms_item in self.waveforms_item_list:
        #     waveforms_item.setVisible(
        #         visible and not self.current_spike_object is None)

        # self.thr_item.setVisible(visible)

    def drawThreshold(self):
        x = self._x_range
        y = np.ones(len(x)) * self.current_spike_object.threshold
        z = np.ones(len(x)) * .01
        pos = np.vstack([x, y, z]).transpose()

        self.thr_item.setData(pos=pos)

    def drawWaveforms(self):
        # create elements
        waveforms = self.current_spike_object._waveforms[self.current_wavs_mask]
        unit_IDs = self.current_spike_object._unit_IDs[self.current_wavs_mask]
        xlen = waveforms.shape[1]
        x_element = np.arange(xlen)

        if len(unit_IDs) > self.GLOBAL_WAVS_LIMIT:
            ds_index = self.downsamplingWaveforms(waveforms, unit_IDs)
            waveforms = waveforms[ds_index]
            unit_IDs = unit_IDs[ds_index]

        elif len(unit_IDs) < 1:
            self.waveform_item.setData(pos=None)
            return

        resort_index = np.argsort(unit_IDs)
        waveforms = waveforms[resort_index]
        unit_IDs = unit_IDs[resort_index]

        x = np.tile(x_element, waveforms.shape[0])
        y = np.ravel(waveforms)
        z = np.zeros(len(y))

        vertexes = np.vstack([x, y, z]).transpose()

        color = self.getColor(unit_IDs)
        color = np.repeat(color, len(x_element), axis=0)

        self.waveform_item.setData(pos=vertexes,
                                   color=color,
                                   wav_size=xlen)

    def getColor(self, unit_IDs):
        n = len(unit_IDs)
        color = np.zeros((n, 3))

        unit_color_map = dict(zip(self.current_spike_object.unit_header['ID'], np.arange(
            self.current_spike_object.unit_header.shape[0], dtype=int)))

        def color_map(ID):
            return self.color_palette_list[unit_color_map.get(int(ID))]

        color = np.array(np.vectorize(color_map)(unit_IDs)).T

        color = np.hstack((color, np.ones((n, 1))))

        return color

    def downsamplingWaveforms(self, waveforms, unit_IDs):
        length = waveforms.shape[0]
        msg = "reducing number of waveforms to plot"
        logger.info(msg)
        max_set = []
        min_set = []
        # waveforms = self.data_spike_chan.Waveforms
        for unit in np.unique(unit_IDs):
            unit_mask = unit_IDs == unit
            waveforms_unit = waveforms[unit_mask, :]
            max_set_u = np.argmax(waveforms_unit, axis=0)
            min_set_u = np.argmin(waveforms_unit, axis=0)
            inv = np.where(unit_mask)[0]
            max_set.append(inv[max_set_u])
            min_set.append(inv[min_set_u])

        max_set = np.concatenate(max_set)
        min_set = np.concatenate(min_set)

        rand_set = np.random.permutation(length)[:self.GLOBAL_WAVS_LIMIT]

        ds_index = np.unique(np.hstack((max_set, rand_set, min_set)))

        return ds_index

    def viewMatrix(self):
        """Overwrite viewMatrix"""
        tr = QMatrix4x4()
        tr.translate(0.0, 0.0, -self.opts['distance'])
        if self.opts['rotationMethod'] == 'quaternion':
            tr.rotate(self.opts['rotation'])
        else:
            # default rotation method
            tr.rotate(self.opts['elevation']-90, 1, 0, 0)
            tr.rotate(self.opts['azimuth']+90, 0, 0, -1)
        center = self.opts['center']
        tr.scale(1, self._y_scale, 1)
        tr.translate(-center.x(), -center.y(), -center.z())
        return tr

    def graphMouseWheelEvent(self, event):
        """Overwrite PlotItem.getViewBox().wheelEvent."""
        modifiers = event.modifiers()
        if (modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
            """scale y axis."""
            delta = -event.angleDelta().x()
            if delta == 0:
                delta = -event.angleDelta().y()
            self._y_scale *= 0.999**delta
            self.update()

        # modifiers = QGuiApplication.keyboardModifiers()
        # if (modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
        #     """scale y axis."""
        #     delta = int(event.delta() / 120)
        #     # current_range = self.plot_item.getViewBox().state['viewRange']
        #     data_scale = int(self._y_range[1] / (1 + delta / 10))
        #     self._y_range = (-data_scale, data_scale)

        #     # self.redraw_data = True
        #     # self.redraw_bg = True
        #     # self.redraw_events = True
        #     # self.redraw_spikes = True
        #     self.updatePlot()

    def disable_action(self, *args, **kwargs):
        pass

    def graphMousePressEvent(self, event):
        """Overwrite PlotItem.scene().mousePressEvent."""
        self.manual_curve_item.setVisible(True)

        pos = event.scenePos()
        mouse_view = self.getViewBox().mapSceneToView(pos)
        x = mouse_view.x()
        y = mouse_view.y()

        self.manual_curve_item.setData([x, x], [y, y])

    def graphMouseMoveEvent(self, event):
        """Overwrite PlotItem.scene().mouseMoveEvent."""
        if self.manual_mode:
            pos = event.scenePos()
            mouse_view = self.getViewBox().mapSceneToView(pos)
            x = mouse_view.x()
            y = mouse_view.y()

            x_data, y_data = self.manual_curve_item.getData()

            x_data = np.append(x_data, x)
            y_data = np.append(y_data, y)

            self.manual_curve_item.setData(x_data, y_data)

    def graphMouseReleaseEvent(self, event):
        """Overwrite PlotItem.scene().mouseReleaseEvent."""
        self.manual_curve_item.setVisible(False)


class WaveformGLItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, **kwds):
        super().__init__()
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.pos = np.zeros((36, 2))
        self.color = np.zeros((36, 4))
        self.wav_size = 36
        self.has_selected_wav = False
        self.selected_wav_pos = None

        self.setData(**kwds)
        # print(color.dtype)
        # self.unit_IDs = unit_IDs
        self.width = 1.
        self.start = np.arange(stop=self.pos.shape[0], step=self.wav_size)
        self.count = np.ones(
            self.pos.shape[0] // self.wav_size) * self.wav_size

    def setData(self, **kwds):
        args = ['pos', 'color', 'wav_size']
        for k in kwds.keys():
            if k not in args:
                raise ValueError(
                    'Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        for arg in args:
            if arg in kwds:
                value = kwds[arg]
                if arg == 'pos':
                    if value is None:
                        pass
                    elif not isinstance(value, np.ndarray):
                        value = np.array(value).astype(np.int32)
                elif arg == 'color':
                    if not isinstance(value, np.ndarray):
                        value = np.array(value)

                setattr(self, arg, value)

        if self.pos is None:
            self.pos = np.zeros((36, 2))
            self.color = np.zeros((36, 4))
            self.wav_size = 36
        self.start = np.arange(stop=self.pos.shape[0], step=self.wav_size)
        self.count = np.ones(
            self.pos.shape[0] // self.wav_size) * self.wav_size
        self.update()

    # def setSelectWav(self, selected, pos):
    #     self.has_selected_wav = selected
    #     self.selected_wav_pos = pos
    #     self.update()

    def paint(self):
        import time

        start = time.perf_counter()
        if self.pos is None:
            return
        self.setupGLState()
        glEnableClientState(GL_VERTEX_ARRAY)

        try:
            glVertexPointer(
                self.pos.shape[1], GL_FLOAT, 0, self.pos)
            if isinstance(self.color, np.ndarray):
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(self.color.shape[1], GL_FLOAT, 0, self.color)

            glLineWidth(self.width)

            glMultiDrawArrays(GL_LINE_STRIP, self.start,
                              self.count, len(self.start))

            #     glVertexPointer(
            #         self.selected_wav_pos.shape[1], GL_FLOAT, 0, self.selected_wav_pos)
            #     glLineWidth(self.width)
            #     glDrawArrays(GL_LINE_STRIP, 0, self.wav_size)

        finally:
            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
        logger.info(f'plot wav {time.perf_counter() - start}')


# class SelectWaveformGLItem(gl.GLGraphicsItem.GLGraphicsItem):
#     def __init__(self, **kwds):
#         super().__init__()
#         glopts = kwds.pop('glOptions', 'additive')
#         self.setGLOptions(glopts)
#         self.pos = np.zeros((36, 2))
#         self.color = np.zeros((36, 4))

#         self.setData_(**kwds)
#         # print(color.dtype)
#         # self.unit_IDs = unit_IDs
#         self.width = 2.

#     def setData_(self, **kwds):
#         args = ['pos', 'color']
#         for k in kwds.keys():
#             if k not in args:
#                 raise ValueError(
#                     'Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
#         for arg in args:
#             if arg in kwds:
#                 value = kwds[arg]
#                 if arg == 'pos':
#                     if value is None:
#                         pass
#                     elif not isinstance(value, np.ndarray):
#                         value = np.array(value).astype(np.int32)
#                 elif arg == 'color':
#                     if not isinstance(value, np.ndarray):
#                         value = np.array(value)

#                 setattr(self, arg, value)

#         if self.pos is None:
#             self.pos = np.zeros((36, 2))
#             self.color = np.zeros((36, 4))
#         self.update()

#     def paint(self):
#         if self.pos is None:
#             return
#         self.setupGLState()
#         glEnableClientState(GL_VERTEX_ARRAY)

#         try:
#             glVertexPointer(
#                 self.pos.shape[1], GL_FLOAT, 0, self.pos)
#             if isinstance(self.color, np.ndarray):
#                 glEnableClientState(GL_COLOR_ARRAY)
#                 glColorPointer(self.color.shape[1], GL_FLOAT, 0, self.color)

#             glLineWidth(self.width)

#             glDrawArrays(GL_LINE_STRIP, 0, self.pos.shape[0])

#         finally:
#             glDisableClientState(GL_COLOR_ARRAY)
#             glDisableClientState(GL_VERTEX_ARRAY)
#         logger.info('plot select')
