import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QPushButton, QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.amplitude = 1.0
        self.frequency = 1.0
        self.phase = 0.0
        self.curve_visible = False

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.curve_visible:
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_LINE_STRIP)

            x_values = np.linspace(-10, 10, 1000)
            y_values = self.amplitude * \
                np.sin(self.frequency * x_values + self.phase)

            for x, y in zip(x_values, y_values):
                glVertex3f(x, y, 0.0)

            glEnd()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-10, 10, -2, 2, -1, 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Parametric Folded Curve")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.opengl_widget = OpenGLWidget(self)
        layout.addWidget(self.opengl_widget)

        amplitude_slider = QSlider(Qt.Horizontal)
        amplitude_slider.setMinimum(1)
        amplitude_slider.setMaximum(10)
        amplitude_slider.setValue(5)
        amplitude_slider.valueChanged.connect(self.update_amplitude)
        layout.addWidget(amplitude_slider)

        frequency_slider = QSlider(Qt.Horizontal)
        frequency_slider.setMinimum(1)
        frequency_slider.setMaximum(10)
        frequency_slider.setValue(5)
        frequency_slider.valueChanged.connect(self.update_frequency)
        layout.addWidget(frequency_slider)

        phase_slider = QSlider(Qt.Horizontal)
        phase_slider.setMinimum(0)
        phase_slider.setMaximum(360)
        phase_slider.setValue(0)
        phase_slider.valueChanged.connect(self.update_phase)
        layout.addWidget(phase_slider)

        self.toggle_curve_button = QPushButton("Show Curve", self)
        self.toggle_curve_button.setCheckable(True)
        self.toggle_curve_button.setChecked(False)
        self.toggle_curve_button.toggled.connect(self.toggle_curve)
        layout.addWidget(self.toggle_curve_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def update_amplitude(self, value):
        self.opengl_widget.amplitude = value / 2.0
        self.opengl_widget.update()

    def update_frequency(self, value):
        self.opengl_widget.frequency = value / 2.0
        self.opengl_widget.update()

    def update_phase(self, value):
        self.opengl_widget.phase = np.radians(value)
        self.opengl_widget.update()

    def toggle_curve(self, state):
        self.opengl_widget.curve_visible = state
        self.opengl_widget.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
