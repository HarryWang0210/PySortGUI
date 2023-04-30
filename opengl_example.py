import sys
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt
from OpenGL.GL import *

class MyOpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.x_offset = 0
        self.y_offset = 0

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        # Set the viewport to the current window size
        glViewport(0, 0, self.width(), self.height())

        # Set the scissor region to the entire window size
        glScissor(0, 0, self.width(), self.height())
        # glEnable(GL_SCISSOR_TEST)

        # Draw the entire scene
        self.draw_scene()

        # Set the scissor region to the sliding window size
        glScissor(self.x_offset, self.y_offset, 200, self.height())
        glEnable(GL_SCISSOR_TEST)

        # Draw the sliding window
        self.draw_sliding_window()

    def draw_scene(self):
        # Draw the entire scene here
        # Set the color to red
        glColor3f(1.0, 0.0, 0.0)

        # Draw a rectangle
        glBegin(GL_QUADS)
        glVertex2f(-0.5, -0.5)
        glVertex2f(0.5, -0.5)
        glVertex2f(0.5, 0.5)
        glVertex2f(-0.5, 0.5)
        glEnd()
        pass

    def draw_sliding_window(self):
        # Draw the sliding window contents here
        # Set the color to green
        glColor3f(0.0, 1.0, 0.0)

        # Draw a rectangle
        glBegin(GL_QUADS)
        glVertex2f(-0.5, -0.5)
        glVertex2f(0.5, -0.5)
        glVertex2f(0.5, 0.5)
        glVertex2f(-0.5, 0.5)
        glEnd()
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.x_offset -= 10
            self.update()
        elif event.key() == Qt.Key_Right:
            self.x_offset += 10
            self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = MyOpenGLWidget()
    widget.show()
    sys.exit(app.exec_())