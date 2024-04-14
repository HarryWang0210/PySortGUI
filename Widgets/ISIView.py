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
        self.visible = False  # overall visible

        self.box_size = 1
        self.border_width = 0.2
        self.max_time = 0.1
        self.bin_size = 0.001
        self.x_start: np.ndarray = None
        self.y_start: np.ndarray = None

        self.current_wavs_mask = []
        self.current_showing_units = []

        self.current_spike_object = None

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

    def data_file_name_changed(self, data):
        self.data_object = data
        self.visible = False
        self.updatePlot()

    def showing_spike_data_changed(self, new_spike_object: DiscreteData | None):
        self.current_spike_object = new_spike_object

        self.visible = True
        if self.current_spike_object is None:
            self.visible = False
            self.updatePlot()
            return

    def showing_units_changed(self, showing_unit_IDs):
        self.current_showing_units = sorted(showing_unit_IDs)
        self.clear()
        self.updatePlot()

    def updatePlot(self):
        if self.visible:
            self.drawISI(self.current_showing_units)

    def drawISI(self, unit_ID_list: list):
        mask = None

        self.x_start = np.arange(len(unit_ID_list)) * \
            (self.box_size+self.border_width)
        self.y_start = np.arange(len(unit_ID_list)) * \
            (self.box_size+self.border_width)

        for i in range(len(unit_ID_list)):
            ID_y = unit_ID_list[i]
            y_offset = self.y_start[i]
            for j in range(len(unit_ID_list)):
                if i > j:
                    continue

                ID_x = unit_ID_list[j]
                x_offset = self.x_start[j]
                logger.debug(f'{ID_y}, {ID_x}')
                x, y = self.computeISI(ID_y, ID_x)
                if mask is None:
                    mask = x < 0.005
                logger.debug(len(y))
                x = x * self.box_size / self.max_time
                y = y * self.box_size / np.max(y)
                bar1 = pg.BarGraphItem(x0=x + x_offset, width=self.box_size/(self.max_time/self.bin_size),
                                       y0=y_offset, height=y,
                                       pen=(0, 0, 0, 0),
                                       brushes=[(0, 0, 0) if i else (224, 224, 224) for i in mask])
                self.addItem(bar1)

        x_mid = self.x_start + self.box_size / 2
        y_mid = self.y_start + self.box_size / 2

        x_ticks = list(zip(x_mid, [str(unit_ID) for unit_ID in unit_ID_list]))
        y_ticks = list(zip(y_mid, [str(unit_ID) for unit_ID in unit_ID_list]))
        x_axis = self.getAxis('bottom')
        y_axis = self.getAxis('left')
        x_axis.setTicks([x_ticks])
        y_axis.setTicks([y_ticks])

        font = QFont()
        font.setPixelSize(40)
        x_axis.setStyle(tickFont=font)
        x_axis.setHeight(45)
        y_axis.setStyle(tickFont=font)
        self.plot_item.enableAutoRange()

    def computeISI(self, unit1: int, unit2: int):
        x, y = self.current_spike_object.ISI(list({unit1, unit2}),
                                             t_max=self.max_time,
                                             bin_size=self.bin_size)
        # x = x * self.box_size / self.max_time
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
