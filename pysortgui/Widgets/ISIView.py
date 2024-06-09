import logging

import numpy as np
import pyqtgraph as pg
import seaborn as sns
from PyQt5.QtGui import QColor, QFont

# from OpenGL.GL import *
# from PyQt5.QtWidgets import QOpenGLWidget
from DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from Widgets.WidgetsInterface import WidgetsInterface

logger = logging.getLogger(__name__)


class ISIView(pg.PlotWidget, WidgetsInterface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "ISI View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.color_palette_list = sns.color_palette('bright', 64)
        self.plot_visible = False  # overall visible
        self.widget_visible = False

        self.box_size = 1  # size of a barplot
        self.border_width = 0.2  # space between barplot
        self.max_time = 0.1  # maximum time in sec of isi distribution
        self.bin_size = 0.001  # time in sec of isi distribution bin

        # array-like, cache the time axis of computed isi distribution result
        self.time_axis: np.ndarray = None
        # a dictionary cache all computed isi distribution result
        self.isi_distrib_dict: dict[tuple[int, int], np.ndarray] = dict()
        self.isi_thr: float = 0

        # array-like, store the x loc of every barplot start
        self.x_start: np.ndarray = None
        # array-like, store the y loc of every barplot start
        self.y_start: np.ndarray = None

        self.current_spike_object: DiscreteData = None  # spike object
        self.current_showing_units = []  # array-like, store the ID of showing units

        self.initPlotItem()

    def initPlotItem(self):
        """
        Initialize plotWidget and plotItems.
        """
        self.plot_item = self.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.setClipToView(True)

        # setup background
        background_color = (0.35, 0.35, 0.35)
        background_color = QColor(*[int(c * 255) for c in background_color])
        self.setBackground(background_color)

        x_axis = self.getAxis('bottom')
        y_axis = self.getAxis('left')
        font = QFont('Arial')
        font.setPixelSize(30)
        x_axis.setHeight(35)
        x_axis.setStyle(tickFont=font)
        x_axis.setTextPen(pg.mkPen(color='w'))
        y_axis.setStyle(tickFont=font)
        y_axis.setTextPen(pg.mkPen(color='w'))
        self.initAxis()

    def initAxis(self):
        x_axis = self.getAxis('bottom')
        y_axis = self.getAxis('left')
        x_axis.setTicks([[(0, '')]])
        y_axis.setTicks([[(0, '')]])

    def data_file_name_changed(self, data):
        self.data_object = data
        self.plot_visible = False
        self.initAxis()
        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        if new_spike_object is self.current_spike_object:
            return
        self.current_spike_object = new_spike_object

        self.plot_visible = True
        # array-like, cache the time axis of computed isi distribution result
        self.time_axis = None
        # a dictionary cache all computed isi distribution result
        self.isi_distrib_dict.clear()
        if self.current_spike_object is None:
            self.plot_visible = False
            self.initAxis()
            self.updatePlot()
            return

    def showing_units_changed(self, showing_unit_IDs: list):
        self.current_showing_units = sorted(showing_unit_IDs)
        self.barplot_item_list = []
        self.updatePlot()
        self.plot_item.enableAutoRange()

    def isi_threshold_changed(self, isi_thr: float):
        self.isi_thr = isi_thr
        self.drawAreaUnderISIThreshold(self.isi_thr)

    def widgetVisibilityChanged(self, active: bool):
        self.widget_visible = active
        self.updatePlot()

    def updatePlot(self):
        self.clear()
        visible = self.plot_visible and self.widget_visible
        if visible:
            self.drawISI(self.current_showing_units)

    def drawISI(self, unit_ID_list: list):
        unit_color_map = dict(zip(self.current_spike_object.unit_header['ID'], np.arange(
            self.current_spike_object.unit_header.shape[0], dtype=int)))

        self.x_start = np.arange(len(unit_ID_list)) * \
            (self.box_size+self.border_width)
        self.y_start = np.arange(len(unit_ID_list)) * \
            (self.box_size+self.border_width)

        for i in range(len(unit_ID_list)):
            ID_y = unit_ID_list[i]
            y_offset = self.y_start[i]

            color = self.color_palette_list[unit_color_map[int(ID_y)]]
            color = (np.array(color) * 255).astype(int)

            for j in range(len(unit_ID_list)):
                if i > j:
                    # ignore upper triangular
                    continue
                ID_x = unit_ID_list[j]
                x_offset = self.x_start[j]

                # try use caches
                y = self.isi_distrib_dict.get((ID_y, ID_x))
                x = self.time_axis

                if y is None:
                    # no caches, compute isi
                    x, y = self.computeISI(ID_y, ID_x)
                    self.isi_distrib_dict[(ID_y, ID_x)] = y
                    if self.time_axis is None:
                        self.time_axis = x

                # if (ID_y, ID_x) in self.isi_distrib_dict.keys():
                #     # use caches
                #     x = self.time_axis
                #     y = self.isi_distrib_dict[(ID_y, ID_x)]
                # else:
                #     # no caches, compute isi
                #     x, y = self.computeISI(ID_y, ID_x)
                #     self.isi_distrib_dict[(ID_y, ID_x)] = y
                #     if self.time_axis is None:
                #         self.time_axis = x

                x_values = x.copy()
                y_values = y.copy()

                x_values = x_values * self.box_size / self.max_time

                with np.errstate(divide='ignore', invalid='ignore'):
                    y_values = y_values * self.box_size / np.max(y_values)

                bar = pg.BarGraphItem(x0=x_values + x_offset, width=self.box_size/(self.max_time/self.bin_size),
                                      y0=y_offset, height=y_values,
                                      pen=(0, 0, 0, 0),
                                      brush=color)

                self.barplot_item_list.append(bar)
                self.addItem(bar)

        self.drawAreaUnderISIThreshold(self.isi_thr)

        x_mid = self.x_start + self.box_size / 2
        y_mid = self.y_start + self.box_size / 2

        x_ticks = list(zip(x_mid, [str(unit_ID) for unit_ID in unit_ID_list]))
        y_ticks = list(zip(y_mid, [str(unit_ID) for unit_ID in unit_ID_list]))
        x_axis = self.getAxis('bottom')
        y_axis = self.getAxis('left')
        x_axis.setTicks([x_ticks])
        y_axis.setTicks([y_ticks])

    def drawAreaUnderISIThreshold(self, isi_thr: float):
        if self.time_axis is None:
            return
        under_thr_mask = self.time_axis < isi_thr
        for barplot_item in self.barplot_item_list:
            barplot_color = barplot_item.opts['brush']
            new_color = [(224, 224, 224) if under else barplot_color
                         for under in under_thr_mask]
            barplot_item.setOpts(brushes=new_color)

    def computeISI(self, unit1: int, unit2: int):
        x, y = self.current_spike_object.ISI(list({unit1, unit2}),
                                             t_max=self.max_time,
                                             bin_size=self.bin_size)
        return x, y

    def graphMouseWheelEvent(self, event):
        """Overwrite PlotItem.getViewBox().wheelEvent."""
        pass

    def graphMousePressEvent(self, event):
        """Overwrite PlotItem.scene().mousePressEvent."""
        pass

    def graphMouseMoveEvent(self, event):
        """Overwrite PlotItem.scene().mouseMoveEvent."""
        pass

    def graphMouseReleaseEvent(self, event):
        """Overwrite PlotItem.scene().mouseReleaseEvent."""
        pass
