import logging

import numpy as np
import pyqtgraph as pg
import seaborn as sns
from PyQt5.QtGui import QColor

# from OpenGL.GL import *
# from PyQt5.QtWidgets import QOpenGLWidget
from pysortgui.DataStructure.datav3 import ContinuousData, DiscreteData, SpikeSorterData
from pysortgui.Widgets.WidgetsInterface import WidgetsInterface

logger = logging.getLogger(__name__)


class ISIView(pg.GraphicsLayoutWidget, WidgetsInterface):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "ISI View"
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)

        self.color_palette_list = sns.color_palette('bright', 64)
        self.visible = False  # overall visible

        self.current_wavs_mask = []

        self.current_showing_units = []

        self.current_spike_object = None

        self.initPlotItem()

    def initPlotItem(self):
        """
        Initialize plotWidget and plotItems.
        """
        # self.plot_item = self.getPlotItem()
        # self.plot_item.clear()
        # self.plot_item.setMenuEnabled(False)
        # setup background
        background_color = (0.35, 0.35, 0.35)
        background_color = QColor(*[int(c * 255) for c in background_color])
        self.setBackground(background_color)
        bar1 = pg.BarGraphItem(x=range(5), height=[1, 5, 2, 4, 3], width=0.5)
        bar2 = pg.BarGraphItem(x=range(5), height=[10, 5, 2, 4, 3], width=0.5)

        p1 = self.addPlot(row=0, col=0)
        p2 = self.addPlot(row=1, col=0)

        p1.addItem(bar1)
        p2.addItem(bar2)
        # self.clear()

        # self.getViewBox().wheelEvent = self.graphMouseWheelEvent
        # self.scene().mousePressEvent = self.graphMousePressEvent
        # self.scene().mouseMoveEvent = self.graphMouseMoveEvent
        # self.scene().mouseReleaseEvent = self.graphMouseReleaseEvent
        # hide auto range button
        # self.plot_item.hideButtons()

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
        self.current_showing_units = showing_unit_IDs
        self.current_wavs_mask = np.isin(self.current_spike_object.unit_IDs,
                                         self.current_showing_units)
        self.updatePlot()

    def updatePlot(self):
        pass

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
