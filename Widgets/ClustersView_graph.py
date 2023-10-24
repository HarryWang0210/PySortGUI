from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPainter
import pyqtgraph as pg
import pyqtgraph.opengl as gl
# import copy
import numpy as np
import seaborn as sns
import time
from DataStructure.data import SpikeSorterData
from Widgets.WidgetsInterface import WidgetsInterface


class ClustersView(gl.GLViewWidget, WidgetsInterface):
    signal_data_file_name_changed = QtCore.pyqtSignal(SpikeSorterData)
    signal_spike_chan_changed = QtCore.pyqtSignal(object)
    signal_selected_units_changed = QtCore.pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Clusters View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.data_object = None
        self.color_palette_list = sns.color_palette(None, 64)

        self.visible = False  # overall visible
        self.num_waveforms = 0
        self.waveform_units_visible = []

        self.spikes = None
        self.has_waveforms = False
        self.point_color = None
        self.pca = None
        self.initPlotItem()

    def initPlotItem(self):
        background_color = (0.35, 0.35, 0.35)
        background_color = QColor(*[int(c * 255) for c in background_color])
        self.setBackgroundColor(background_color)

        self.setCameraPosition(distance=2,  elevation=45, azimuth=45)

        # 添加XYZ轴
        self.axis_label_item_list = []
        axis_length = 1
        axis_pos = np.array([[0, 0, 0],
                             [axis_length, 0, 0],
                             [0, 0, 0],
                             [0, axis_length, 0],
                             [0, 0, 0],
                             [0, 0, axis_length]])
        axis_color = np.array([[1, 0, 0, 1],
                               [1, 0, 0, 1],
                               [0, 1, 0, 1],
                               [0, 1, 0, 1],
                               [0, 0, 1, 1],
                               [0, 0, 1, 1]])
        self.axis_manual_curve_item = gl.GLLinePlotItem(
            pos=axis_pos, color=axis_color,  width=2, mode='lines')
        self.addItem(self.axis_manual_curve_item)

        axis_text = ["PCA1", "PCA2", "PCA3"]
        for i in range(3):
            axis_label_item = gl.GLTextItem(text=axis_text[i],
                                            color=(axis_color[i*2] * 255).astype(np.int32))
            self.axis_label_item_list.append(axis_label_item)
            self.addItem(self.axis_label_item_list[i])

        self.axis_label_item_list[0].setData(pos=(axis_length * 1.1, 0, 0))
        self.axis_label_item_list[1].setData(pos=(0, axis_length * 1.1, 0))
        self.axis_label_item_list[2].setData(pos=(0, 0, axis_length * 1.1))

        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('opaque')  # not to mix color
        self.addItem(self.scatter)

        self.manual_curve_item = GLPainterItem(color=(255, 0, 0))
        self.addItem(self.manual_curve_item)

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        self.updatePlot()

    def spike_chan_changed(self, meta_data):
        self.computePCA(meta_data["ID"], meta_data["Label"])
        self.visible = True
        self.waveform_units_visible = [True] * self.num_waveforms
        self.updatePlot()

    def selected_units_changed(self, selected_rows):
        self.waveform_units_visible = np.isin(
            self.spikes["unitID"], list(selected_rows))
        self.updatePlot()

    def computePCA(self, chan_ID, label):
        spikes = self.data_object.getSpikes(chan_ID, label)
        if spikes["unitInfo"] is None:
            self.spikes = None
            self.has_waveforms = False
        else:
            self.spikes = spikes
            self.has_waveforms = True
            self.num_waveforms = self.spikes["waveforms"].shape[0]
            self.pca = self.data_object.waveforms_pca(self.spikes["waveforms"])
            self.point_color = self.getColor()

    def updatePlot(self):
        if self.visible and self.has_waveforms:
            self.scatter.setData(pos=self.pca[self.waveform_units_visible],
                                 size=3,
                                 color=self.point_color[self.waveform_units_visible])
        self.scatter.setVisible(self.visible and self.has_waveforms)

    def getColor(self):
        n = self.num_waveforms
        color = np.zeros((n, 3))

        for i in range(n):
            color[i, :] = self.color_palette_list[int(
                self.spikes["unitID"][i])]
        color = np.hstack((color, np.ones((n, 1))))
        return color

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos

        self.draw_mode = True
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                """select point"""
                pass
            else:
                if self.draw_mode:
                    self.manual_curve_item.setVisible(self.draw_mode)
                    self.manual_curve_item.setData(
                        pos=[[self.mousePos.x(), self.mousePos.y()]])

        # view_w = self.width()
        # view_h = self.height()
        # m = self.projectionMatrix() * self.viewMatrix()
        # m = np.array(m.data(), dtype=np.float32).reshape((4, 4))
        # one_mat = np.ones((self.points.shape[0], 1))
        # points = np.concatenate((self.points, one_mat), axis=1)
        # new = np.matmul(points, m)
        # new[:, :3] = new[:, :3] / new[:, 3].reshape(-1, 1)
        # new = new[:, :3]
        # projected_array = np.zeros((new.shape[0], 2))
        # projected_array[:, 0] = (new[:, 0] + 1) / 2
        # projected_array[:, 1] = (-new[:, 1] + 1) / 2
        # self.projected_array = copy.deepcopy(projected_array)
        # self.projected_array[:, 0] = self.projected_array[:, 0] * view_w
        # self.projected_array[:, 1] = self.projected_array[:, 1] * view_h

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                """select point"""
                pass
            else:
                line_data = self.manual_curve_item.getData()
                line_data = np.append(
                    line_data, [[self.mousePos.x(), self.mousePos.y()]], axis=0)
                self.manual_curve_item.setData(pos=line_data)
        elif ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            self.orbit(-diff.x(), diff.y())

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                """select point"""
                pass
            else:
                line_data = self.manual_curve_item.getData()
                line_data = np.append(
                    line_data, [line_data[0]], axis=0)
                self.manual_curve_item.setData(pos=line_data)
                self.draw_mode = False
                self.manual_curve_item.setVisible(self.draw_mode)


class GLPainterItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, parentItem=None, **kwds):
        """All keyword arguments are passed to setData()"""
        super().__init__(parentItem=parentItem)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)

        self.pos = np.array([[0.0, 0.0]]).astype(np.int32)
        self.color = (255, 255, 255)
        self.type = 'polyline'

        self.setData(**kwds)

    def setData(self, **kwds):
        """
        Update the data displayed by this item. All arguments are optional;
        for example it is allowed to update text while leaving colors unchanged, etc.

        ====================  ==================================================
        **Arguments:**
        ------------------------------------------------------------------------
        pos                   (3,) array of floats specifying text location.
        color                 QColor or array of ints [R,G,B] or [R,G,B,A]. (Default: Qt.white)
        type                  Type to draw. (Default: 'polyline')
        ====================  ==================================================
        """
        args = ['pos', 'color', 'type']
        for k in kwds.keys():
            if k not in args:
                raise ValueError(
                    'Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        for arg in args:
            if arg in kwds:
                value = kwds[arg]
                if arg == 'pos':
                    value = np.array(value).astype(np.int32)
                #     if isinstance(value, np.ndarray):
                #         if value.shape != (3,):
                #             raise ValueError('"pos.shape" must be (3,).')
                #     elif isinstance(value, (tuple, list)):
                #         if len(value) != 3:
                #             raise ValueError('"len(pos)" must be 3.')
                elif arg == 'color':
                    value = QColor(*value)

                setattr(self, arg, value)
        self.update()

    def getData(self):
        return self.pos

    def paint(self):
        self.setupGLState()

        painter = QPainter(self.view())
        painter.setPen(self.color)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)

        if self.type == 'polyline':
            polyline = [QPoint(*point) for point in self.pos]
            painter.drawPolyline(*polyline)
        painter.end()
