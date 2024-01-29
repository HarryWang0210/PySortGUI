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
from Widgets.WidgetsInterface import WidgetsInterface
from sklearn.preprocessing import MaxAbsScaler
from DataStructure.datav3 import SpikeSorterData, ContinuousData, DiscreteData

import logging
logger = logging.getLogger(__name__)


class ClustersView(gl.GLViewWidget, WidgetsInterface):
    signal_manual_waveforms = QtCore.pyqtSignal(object)
    signal_select_point = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Clusters View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        # self.data_object = None
        # self.spikes = None
        # self.has_spikes = False
        self.color_palette_list = sns.color_palette('bright', 64)
        self.visible = False  # overall visible

        # from UnitOperateTools widget
        self.manual_mode = False
        self.feature_on_selection = False
        self.axis_label = ["PCA1", "PCA2", "PCA3"]

        # change only when waveforms change
        # self.num_wavs = 0  # N
        # self.current_wav_units = []  # waveform units (N,), int
        # self.current_wav_colors = []  # waveform colors (N, 3), float

        # data use on showing
        self.current_wavs_mask = []  # visible waveforms (N,), bool
        self.current_pca = []  # current showing pca
        self.current_showing_data = []

        self.current_spike_object = None
        self.current_showing_units = []
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
        self.axis_lines_item = gl.GLLinePlotItem(
            pos=axis_pos, color=axis_color,  width=2, mode='lines')
        self.addItem(self.axis_lines_item)

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

        self.manual_curve_item = GLPainterItem(color=(255, 0, 0))
        self.addItem(self.manual_curve_item)
        self.manual_curve_item.setVisible(False)

    def data_file_name_changed(self, data):
        # self.data_object = data
        self.visible = False
        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.current_spike_object = new_spike_object

        # self.has_spikes = True
        self.visible = True
        if self.current_spike_object is None:
            # self.has_spikes = False
            self.visible = False
            return
        # self.has_spikes = not self.current_spike_object is None
        # self.updatePlot()

    # def spike_chan_changed(self, current_chan_info):
    #     spikes = current_chan_info
    #     if spikes["unitID"] is None:
    #         self.has_spikes = False
    #         self.spikes = None
    #         self.visible = False
    #     else:
    #         self.has_spikes = True
    #         self.spikes = spikes

    #         self.visible = True

    #         # self.current_wav_colors = self.getColor(self.current_wav_units)

    #         # self.current_pca = self.all_wavs_pca[self.current_wavs_mask]
    #         # self.setCurrentShowingData()

    #     # self.updatePlot()

    def showing_units_changed(self, showing_unit_IDs):
        self.current_showing_units = showing_unit_IDs
        self.updatePlot()

    # def showing_spikes_data_changed(self, spikes_data):
    #     if self.has_spikes:
    #         self.current_wav_units = spikes_data['current_wav_units']
    #         self.current_wavs_mask = np.isin(spikes_data['current_wav_units'],
    #                                          spikes_data['current_showing_units'])

    #         self.current_wav_colors = self.getColor(self.current_wav_units)

    #         self.setCurrentPCA()
    #         self.setCurrentShowingData()
    #     self.updatePlot()

    def manual_mode_state_changed(self, state):
        self.manual_mode = state

    def features_changed(self, features):
        self.axis_label = features.copy()
        for i in range(3):
            self.axis_label_item_list[i].setData(text=self.axis_label[i])
        # self.setCurrentShowingData()
        self.updatePlot()

    def feature_on_selection_state_changed(self, state):
        self.feature_on_selection = state
        # self.setCurrentPCA()
        # self.setCurrentShowingData()
        self.updatePlot()

    def updatePlot(self):
        if self.visible and not self.current_spike_object is None:
            self.drawScatter()
        self.scatter_item.setVisible(
            self.visible and not self.current_spike_object is None)

    def drawScatter(self):
        self.current_wavs_mask = np.isin(self.current_spike_object.unit_IDs,
                                         self.current_showing_units)
        self.setCurrentPCA()
        self.current_showing_data = self.setCurrentShowingData()
        wav_colors = self.getColor(self.current_spike_object.unit_IDs)
        self.scatter_item.setData(pos=self.current_showing_data,
                                  size=3,
                                  color=wav_colors)

    def setCurrentPCA(self):
        if self.feature_on_selection:
            self.current_pca = self.current_spike_object.waveformsPCA(selected_unit_IDs=self.current_showing_units,
                                                                      n_components=3,
                                                                      ignore_invalid=False)
            self.current_pca = MaxAbsScaler().fit_transform(self.current_pca)

        else:
            self.current_pca = self.current_spike_object.waveformsPCA(selected_unit_IDs=None,
                                                                      n_components=3,
                                                                      ignore_invalid=False)
            self.current_pca = MaxAbsScaler().fit_transform(self.current_pca)
            self.current_pca = self.current_pca[self.current_wavs_mask]

    def setCurrentShowingData(self):
        # TODO: time, slice
        showing_data = np.zeros((np.sum(self.current_wavs_mask), 3))
        for i in range(3):
            if self.axis_label[i] == 'PCA1':
                showing_data[:, i] = self.current_pca[:, 0]
            elif self.axis_label[i] == 'PCA2':
                showing_data[:, i] = self.current_pca[:, 1]
            elif self.axis_label[i] == 'PCA3':
                showing_data[:, i] = self.current_pca[:, 2]
            elif self.axis_label[i] == 'time':
                ts = self.current_spike_object.timestamps

                transformed_ts = (MaxAbsScaler().fit_transform(
                    ts.reshape(-1, 1))).reshape(-1)

                showing_data[:, i] = transformed_ts[self.current_wavs_mask]
            elif self.axis_label[i] == 'amplitude':
                print("TODO: amplitude")
            elif self.axis_label[i] == 'slice':
                print("TODO: slice")

        return showing_data
        # self.current_showing_data = showing_data

    # def computePCA(self, wav_data):
    #     pca = self.data_object.waveforms_pca(wav_data)
    #     transformed_data = MaxAbsScaler().fit_transform(pca)
    #     return transformed_data

    def getColor(self, unit_IDs):
        logger.debug('getColor')
        n = len(unit_IDs)
        color = np.zeros((n, 3))

        for i in range(n):
            color[i, :] = self.color_palette_list[int(unit_IDs[i])]
        color = np.hstack((color, np.ones((n, 1))))

        return color[self.current_wavs_mask]

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
        xmin, ymin = np.min(verteices, axis=0)
        xmax, ymax = np.max(verteices, axis=0)

        lower_mask = (projected_data > [xmin, ymin]).all(axis=1)
        upper_mask = (projected_data < [xmax, ymax]).all(axis=1)
        in_rect_points = projected_data[lower_mask & upper_mask]
        in_rect_points_index = np.where(lower_mask & upper_mask)[0]

        # secind filter: find the points in polygon
        region = Path(verteices)
        in_region_points_index = in_rect_points_index[
            region.contains_points(in_rect_points)]
        return in_region_points_index

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier):
                """select point"""
                self.nearest_point_item.setVisible(True)
                nearest_point_index = self.findNearestNeighbor(
                    np.array([self.mousePos.x(), self.mousePos.y()]))
                self.nearest_point_item.setData(pos=self.current_showing_data[nearest_point_index, :].reshape((-1, 3)),
                                                size=10,
                                                color=[1, 1, 1, 1])

                self.signal_select_point.emit((True, nearest_point_index))

            elif self.manual_mode:
                self.manual_curve_item.setVisible(True)
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
                nearest_point_index = self.findNearestNeighbor(
                    np.array([self.mousePos.x(), self.mousePos.y()]))
                self.nearest_point_item.setData(pos=self.current_showing_data[nearest_point_index, :].reshape((-1, 3)),
                                                size=10,
                                                color=[1, 1, 1, 1])
                self.signal_select_point.emit((True, nearest_point_index))

            elif self.manual_mode:
                line_data = self.manual_curve_item.getData()
                line_data = np.append(
                    line_data, [[self.mousePos.x(), self.mousePos.y()]], axis=0)
                self.manual_curve_item.setData(pos=line_data)
        elif ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            self.orbit(-diff.x(), diff.y())

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.nearest_point_item.setVisible(False)
            self.signal_select_point.emit((False, 0))

            if self.manual_mode:
                line_data = self.manual_curve_item.getData()
                line_data = np.append(
                    line_data, [line_data[0]], axis=0)
                self.manual_curve_item.setData(pos=line_data)
                self.manual_curve_item.setVisible(False)

                in_region_points_index = self.findPointInRegion(line_data)
                global_index = np.where(self.current_wavs_mask)[0][
                    in_region_points_index]

                if len(global_index) == 0:
                    return
                self.signal_manual_waveforms.emit(global_index)


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
