from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QPainter
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import *  # noqa
import numpy as np
from scipy.spatial import KDTree
from matplotlib.path import Path
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
        self.spikes = None
        self.has_spikes = False
        self.color_palette_list = sns.color_palette(None, 64)
        self.visible = False  # overall visible\

        # from UnitOperateTools widget
        self.draw_mode = False
        self.feature_on_selection = False
        self.axis_label = ["PCA1", "PCA2", "PCA3"]

        # change only when waveforms change
        self.num_wavs = 0  # N
        self.all_wavs_pca = []  # backup pca on all waveforms
        self.current_wav_units = []  # waveform units (N,), int
        self.current_wav_colors = []  # waveform colors (N, 3), float

        # data use on showing
        self.current_wavs_mask_list = []  # visible waveforms (N,), bool
        self.current_pca = []  # current showing pca
        self.current_showing_data = []

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

        self.axis_label = ["PCA1", "PCA2", "PCA3"]
        for i in range(3):
            axis_label_item = gl.GLTextItem(text=self.axis_label[i],
                                            color=(axis_color[i*2] * 255).astype(np.int32))
            self.axis_label_item_list.append(axis_label_item)
            self.addItem(self.axis_label_item_list[i])

        self.axis_label_item_list[0].setData(pos=(axis_length * 1.1, 0, 0))
        self.axis_label_item_list[1].setData(pos=(0, axis_length * 1.1, 0))
        self.axis_label_item_list[2].setData(pos=(0, 0, axis_length * 1.1))

        self.scatter_item = gl.GLScatterPlotItem()
        self.scatter_item.setGLOptions('opaque')  # not to mix color
        self.addItem(self.scatter_item)

        self.nearest_point_item = gl.GLScatterPlotItem()
        self.nearest_point_item.setGLOptions('opaque')  # not to mix color
        self.addItem(self.nearest_point_item)

        self.test_point_item = gl.GLScatterPlotItem()
        self.test_point_item.setGLOptions('opaque')  # not to mix color
        self.addItem(self.test_point_item)

        self.manual_curve_item = GLPainterItem(color=(255, 0, 0))
        self.addItem(self.manual_curve_item)

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        self.updatePlot()

    def spike_chan_changed(self, meta_data):
        spikes = self.data_object.getSpikes(
            meta_data["ID"], meta_data["Label"])
        if spikes["unitInfo"] is None:
            self.has_spikes = False
            self.spikes = None
            self.visible = False
        else:
            self.has_spikes = True
            self.spikes = spikes

            self.visible = True
            self.num_wavs = self.spikes["waveforms"].shape[0]
            self.all_wavs_pca = self.computePCA(self.spikes["waveforms"])

            self.current_wav_units = self.spikes["unitID"]
            self.current_wav_colors = self.getColor(self.current_wav_units)

            self.current_wavs_mask_list = [True] * self.num_wavs
            self.current_pca = self.all_wavs_pca[self.current_wavs_mask_list]
            self.setCurrentShowingData()

        self.updatePlot()

    def selected_units_changed(self, selected_rows):
        self.current_wavs_mask_list = np.isin(
            self.current_wav_units, list(selected_rows))
        if self.feature_on_selection:
            self.current_pca = self.computePCA(
                self.spikes["waveforms"][self.current_wavs_mask_list])
        else:
            self.current_pca = self.all_wavs_pca[self.current_wavs_mask_list]
        self.setCurrentShowingData()
        self.updatePlot()

    def updatePlot(self):
        if self.visible and self.has_spikes:
            self.scatter_item.setData(pos=self.current_showing_data,
                                      size=3,
                                      color=self.current_wav_colors[self.current_wavs_mask_list])
        self.scatter_item.setVisible(self.visible and self.has_spikes)

    def setCurrentShowingData(self):
        # TODO:
        showing_data = np.zeros((np.sum(self.current_wavs_mask_list), 3))
        for i in range(3):
            if self.axis_label[i] == 'PCA1':
                showing_data[:, i] = self.current_pca[:, 0]
            elif self.axis_label[i] == 'PCA2':
                showing_data[:, i] = self.current_pca[:, 1]
            elif self.axis_label[i] == 'PCA3':
                showing_data[:, i] = self.current_pca[:, 2]
            elif self.axis_label[i] == 'time':
                pass
            elif self.axis_label[i] == 'slice':
                pass

        self.current_showing_data = showing_data

    def computePCA(self, wav_data):
        return self.data_object.waveforms_pca(wav_data)

    def getColor(self, unit_data):
        n = len(unit_data)
        color = np.zeros((n, 3))

        for i in range(n):
            color[i, :] = self.color_palette_list[int(unit_data[i])]
        color = np.hstack((color, np.ones((n, 1))))
        return color

    def __project(self, obj_pos):
        # modify from pyqtgraph.opengl.items.GLTextItem
        # FIXME: work when obj_pos shape=(1, 3)
        modelview = np.array(self.viewMatrix().data()
                             ).reshape((4, 4))  # (4, 4)
        projection = np.array(self.projectionMatrix().data()
                              ).reshape((4, 4))  # (4, 4)
        viewport = [0, 0, self.width(), self.height()]
        obj_vec = np.hstack((obj_pos, np.ones((obj_pos.shape[0], 1))))

        view_vec = np.matmul(modelview.T, obj_vec.T)
        proj_vec = np.matmul(projection.T, view_vec).T

        zero_filter = proj_vec[:, 3] != 0.0

        result = np.zeros((proj_vec.shape[0], 2))

        nozero_proj_vec = proj_vec[zero_filter]
        nozero_proj_vec = (nozero_proj_vec[:, 0:3].T / nozero_proj_vec[:, 3]).T

        result[zero_filter] = np.array([viewport[0] + (1.0 + nozero_proj_vec[:, 0]) * viewport[2] / 2,
                                        viewport[3] - (viewport[1] + (1.0 + nozero_proj_vec[:, 1]) * viewport[3] / 2)]).T
        return result

    def findNearestNeighbor(self, pos):
        projected_data = self.__project(self.current_showing_data)
        tree = KDTree(projected_data)
        nearest_point_index = tree.query(pos)[1]
        return nearest_point_index

    def findPointInRegion(self, verteices):
        projected_data = self.__project(self.current_showing_data)

        # first filter: reduce point count
        xmin, ymin = np.min(projected_data, axis=0)
        xmax, ymax = np.max(projected_data, axis=0)

        lower_mask = (projected_data > [xmin, ymin]).all(axis=1)
        upper_mask = (projected_data < [xmax, ymax]).all(axis=1)
        in_rect_points = projected_data[lower_mask & upper_mask]
        in_rect_points_index = np.where(lower_mask & upper_mask)[0]
        print(in_rect_points)
        # secind filter: find the points in polygon
        region = Path(verteices)
        in_region_points_index = in_rect_points_index[
            region.contains_points(in_rect_points)]
        return in_region_points_index

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos

        self.draw_mode = True
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                """select point"""
                self.nearest_point_item.setVisible(True)
                nearest_point_index = self.findNearestNeighbor(
                    np.array([self.mousePos.x(), self.mousePos.y()]))
                self.nearest_point_item.setData(pos=self.current_showing_data[nearest_point_index, :].reshape((-1, 3)),
                                                size=10,
                                                color=[1, 1, 1, 1])
            else:
                if self.draw_mode:
                    self.manual_curve_item.setVisible(self.draw_mode)
                    self.manual_curve_item.setData(
                        pos=[[self.mousePos.x(), self.mousePos.y()]])

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                """select point"""
                nearest_index = self.findNearestNeighbor(
                    np.array([self.mousePos.x(), self.mousePos.y()]))
                self.nearest_point_item.setData(pos=self.current_showing_data[nearest_index, :].reshape((-1, 3)),
                                                size=10,
                                                color=[1, 1, 1, 1])
            else:
                line_data = self.manual_curve_item.getData()
                line_data = np.append(
                    line_data, [[self.mousePos.x(), self.mousePos.y()]], axis=0)
                self.manual_curve_item.setData(pos=line_data)
        elif ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            self.orbit(-diff.x(), diff.y())

    def mouseReleaseEvent(self, ev):
        # FIXME: 有些點不會被圈到，但是可以被當成最近點(project沒問題)
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.nearest_point_item.setVisible(False)

            line_data = self.manual_curve_item.getData()
            line_data = np.append(
                line_data, [line_data[0]], axis=0)
            self.manual_curve_item.setData(pos=line_data)
            self.draw_mode = False

            in_region_points_index = self.findPointInRegion(line_data)
            self.test_point_item.setData(pos=self.current_showing_data[in_region_points_index, :].reshape((-1, 3)),
                                         size=10,
                                         color=[1, 1, 1, 1])
            # self.manual_curve_item.setVisible(self.draw_mode)


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
