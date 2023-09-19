from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import numpy as np
import tables


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 创建一个QWidget作为主窗口的中心部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        # 设置窗口属性
        self.setWindowTitle("PyQtGraph in QMainWindow")
        self.setGeometry(100, 100, 800, 600)
        # 创建一个垂直布局
        layout = QVBoxLayout(central_widget)

        # 创建一个PlotWidget
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        # 创建一个按钮来切换直线的可见性
        self.toggle_button = QPushButton("Toggle Line Visibility")
        self.toggle_button.clicked.connect(self.toggle_line_visibility)
        layout.addWidget(self.toggle_button)

        self.button = QPushButton("Downsample Data")
        self.button.clicked.connect(self.set_downsampling)
        layout.addWidget(self.button)

        self.plot_item = self.plot.getPlotItem()
        self.plot_item.getViewBox().wheelEvent = self.customMouseWheel
        self.plot_item.scene().mousePressEvent = self.gmousePressEvent
        self.plot_item.scene().mouseMoveEvent = self.gmouseMoveEvent
        self.plot_item.scene().mouseReleaseEvent = self.gmouseReleaseEvent
        # self.plot_item.getViewBox().setXRange(0, 1000, padding=0)

        background_color = QtGui.QColor(
            *[int(c * 255) for c in (0.35, 0.35, 0.35)])  # 使用红色(RGB值为255, 0, 0)
        self.plot.setBackground(background_color)
        # self.plot.hideButtons()

        # 获取 x 轴和 y 轴对象
        x_axis = self.plot.getAxis('bottom')
        y_axis = self.plot.getAxis('left')

        # 将 x 轴和 y 轴的可见性设置为 False
        # x_axis.setPen(None)
        # x_axis.setStyle(showValues=False)
        # y_axis.setPen(None)
        # y_axis.setStyle(showValues=False)

        self.raw_item = pg.PlotDataItem(pen='w')
        self.plot.addItem(self.raw_item)

        # # 生成一些示例数据
        y1 = np.arange(10)
        y2 = np.flip(y1)
        y3 = np.ones(10) * 5

        y = np.concatenate((y1, y2))
        y = np.concatenate((y, y3))

        x = np.tile(np.arange(10), 3)
        connect = np.ones(len(y1)-1)
        connect = np.append(connect, 0)
        connect = np.tile(connect, 3).astype(np.int32)

        # 创建一个PlotDataItem，并添加数据
        self.data_item = self.plot.plot(
            x, y, pen='w', connect=connect)
        # self.data_item.setData(x=[0, 1, 2], y=[1, 0, 1], connect="auto")
        # 创建y=0的直线，默认不可见
        self.line = pg.InfiniteLine(angle=0, movable=True, pen='g')
        self.line.setVisible(False)
        self.plot.addItem(self.line)

        # 存储手动绘制的线条
        self.manual_lines = []
        # 初始化手动绘制模式状态
        self.draw_mode = False

    def setData(self, filename="data/MX6-22_2020-06-17_17-07-48_no_ref.h5"):
        chan_ID = 0
        with tables.open_file(filename, mode="r") as file:
            if "/Raws" in file.root:
                Raws = file.get_node("/Raws")
                with tables.open_file(Raws(mode='r')._v_file.filename, mode="r") as rawsfile:
                    data = rawsfile.get_node(
                        "/Raws/raw" + str(chan_ID).zfill(3))
                    self.data = data[:2000]
            else:
                raise

    def set_downsampling(self):
        # 设置 downsample 参数
        self.data_item.setDownsampling(
            auto=True,  method='peak')  # 使用 'peak' 模式进行数据降采样

    def customMouseWheel(self, event):
        modifiers = QtGui.QGuiApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.ShiftModifier:
            delta = int(event.delta()/120)  # 获取鼠标滚轮滚动的距离
            current_range = self.plot_item.getViewBox().state['viewRange']
            new_range = [current_range[0][0] +
                         delta, current_range[0][1] + delta]
            self.plot_item.getViewBox().setXRange(*new_range, padding=0)

        elif (modifiers == (QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)):
            delta = int(event.delta()/120)  # 获取鼠标滚轮滚动的距离
            current_range = self.plot_item.getViewBox().state['viewRange']
            new_range = [current_range[1][0] +
                         delta, current_range[1][1] + delta]
            self.plot_item.getViewBox().setYRange(*new_range, padding=0)
        else:
            # 如果没有按下Shift键，执行默认的滚轮操作
            pass

    def toggle_line_visibility(self):
        # 切换直线的可见性
        self.line.setVisible(not self.line.isVisible())

    def toggle_draw_mode(self):
        # 切换手动绘制模式状态
        self.draw_mode = not self.draw_mode

        if self.draw_mode:
            self.draw_button.setText("Draw Mode: On")
        else:
            self.draw_button.setText("Draw Mode: Off")
            # 清除手动绘制的线条
            self.manual_lines = []
            self.manual_curve.setData(None)

    def gmousePressEvent(self, event):
        self.draw_mode = True
        # # 在手动绘制模式下，当鼠标按下时创建一个新的线条

        # 创建一个PlotCurveItem用于手动绘制线条
        pen = pg.mkPen('r', width=2)  # 创建线条笔刷
        self.manual_curve = pg.PlotCurveItem(pen=pen, clickable=False)
        self.plot.addItem(self.manual_curve)
        # self.manual_lines.append(self.manual_curve)

        # # 获取鼠标按下的位置
        pos = event.scenePos()
        view = self.plot.getViewBox()
        mouse_view = view.mapSceneToView(pos)
        x = mouse_view.x()
        y = mouse_view.y()
        # 设置线条的起始点
        self.manual_curve.setData([x, x], [y, y])
        print("press: ", x, y)

    def gmouseMoveEvent(self, event):
        if self.draw_mode:
            # 在手动绘制模式下，当鼠标移动时添加线条的点
            pos = event.scenePos()
            view = self.plot.getViewBox()
            mouse_view = view.mapSceneToView(pos)
            x = mouse_view.x()
            y = mouse_view.y()

            # 获取线条的当前点坐标
            x_data, y_data = self.manual_curve.getData()

            # 添加新的点
            x_data = np.append(x_data, x)
            y_data = np.append(y_data, y)

            # 更新线条的数据
            self.manual_curve.setData(x_data, y_data)
            print("move: ", x, y)

    def gmouseReleaseEvent(self, event):
        self.draw_mode = False
        print(self.manual_curve.getData())
        self.manual_curve.setData(np.array([]), np.array([]))


def main():
    app = QApplication(sys.argv)
    main_window = MyMainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
# import pyqtgraph.examples
# pyqtgraph.examples.run()
