import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QOpenGLWidget, QMainWindow, QPushButton
from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, QMatrix4x4, QPainter, QColor, QFont, QTransform
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np


class OpenGLWidget(QOpenGLWidget):
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glLoadIdentity()

        self.zoom_factor = 1.0  # 初始縮放因子
        # 编译着色器
        vertex_shader_source = """
        attribute vec3 position;
        attribute vec3 color;
        uniform mat4 model_matrix;
        varying vec3 fragColor;

        void main() {
            gl_Position = model_matrix * vec4(position, 1.0);
            fragColor = color;
        }
        """
        fragment_shader_source = """
        varying vec3 fragColor;

        void main() {
            gl_FragColor = vec4(fragColor, 1.0);
        }
        """

        vertex_shader = QOpenGLShader(QOpenGLShader.Vertex)
        vertex_shader.compileSourceCode(vertex_shader_source)

        fragment_shader = QOpenGLShader(QOpenGLShader.Fragment)
        fragment_shader.compileSourceCode(fragment_shader_source)

        # 创建着色器程序并链接
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShader(vertex_shader)
        self.shader_program.addShader(fragment_shader)
        self.shader_program.link()
        self.model_matrix = QMatrix4x4()

        # 创建并绑定VBO
        self.vbo = glGenBuffers(2)
        # self.histogram_vbo = glGenBuffers(1)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 使用修改后的着色器程序
        self.shader_program.bind()
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])
        self.model_matrix.setToIdentity()
        self.model_matrix.scale(self.zoom_factor, self.zoom_factor)
        self.shader_program.setUniformValue("model_matrix", self.model_matrix)

        # 定义顶点数据，包括位置和颜色信息
        vertices_data = np.array([
            # 直线（绿色）
            -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # 顶点1，x, y, r, g, b
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # 顶点2，x, y, r, g, b

            # 折线（白色）
            -0.5, 0.5, 0.0, 1.0, 1.0, 1.0,  # 顶点3，x, y, r, g, b
            0.0, -0.5, 0.0, 1.0, 1.0, 1.0,  # 顶点4，x, y, r, g, b
            0.5, 0.5, 0.0, 1.0, 1.0, 1.0  # 顶点5，x, y, r, g, b
        ], dtype=np.float32)

        # 将顶点数据上传到VBO
        glBufferData(GL_ARRAY_BUFFER, vertices_data.nbytes,
                     vertices_data, GL_STATIC_DRAW)

        # 设置顶点属性指针
        position = self.shader_program.attributeLocation("position")
        color = self.shader_program.attributeLocation("color")
        self.shader_program.enableAttributeArray(position)
        self.shader_program.enableAttributeArray(color)
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE,
                              6 * 4, ctypes.c_void_p(3 * 4))

        # 绘制折线（白色）
        glDrawArrays(GL_LINE_STRIP, 2, 3)

        # 检查是否绘制绿色直线
        if self.draw_green_line:
            # 绘制直线（绿色）
            glDrawArrays(GL_LINES, 0, 2)

        self.shader_program.disableAttributeArray(position)
        self.shader_program.disableAttributeArray(color)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.drawHistogram()

        print(self.model_matrix.data())
        # self.render_text_opengl("Hello, OpenGL!", -0.9, 0.9)
        # 使用QPainter在OpenGL上绘制文本
        # self.render_text_qpainter("Hello, OpenGL!", -0.9, 0.9)

        # 解綁
        self.shader_program.release()

    # 添加用于绘制直方图的方法
    def drawHistogram(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])

        # 定义直方图的顶点数据，包括位置和颜色信息
        histogram_vertices_data = np.array([
            0.0, -1.0, 1.0, 0.0, 0.0,  # 直方图的顶点数据，每个柱状条的位置和颜色
            # 例如：x1, y1, r1, g1, b1, x2, y2, r2, g2, b2, ...
            0.0, 1.0, 1.0, 0.0, 0.0,
            0.3, -1.0, 1.0, 0.0, 0.0,  # 根据你的直方图数据和需求设置
            0.3, 1.0, 1.0, 0.0, 0.0  # ...
        ], dtype=np.float32)

        # 将直方图的顶点数据上传到VBO
        glBufferData(GL_ARRAY_BUFFER, histogram_vertices_data.nbytes,
                     histogram_vertices_data, GL_STATIC_DRAW)

        # 设置顶点属性指针（假设你的着色器有相应的属性）
        position = self.shader_program.attributeLocation("position")
        color = self.shader_program.attributeLocation("color")
        self.shader_program.enableAttributeArray(position)
        self.shader_program.enableAttributeArray(color)
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 5 * 4, None)
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE,
                              5 * 4, ctypes.c_void_p(2 * 4))

        # 绘制直方图（例如使用 glDrawArrays）
        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(histogram_vertices_data) // 5)

        # 解绑
        self.shader_program.disableAttributeArray(position)
        self.shader_program.disableAttributeArray(color)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_text_qpainter(self, text, x, y):
        painter = QPainter(self)
        # painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(255, 255, 255))  # 设置文本颜色
        painter.setFont(QFont("Arial", 12))  # 设置字体和大小
        painter.drawText(int(self.width() * (x + 1) / 2),
                         int(self.height() * (1 - y) / 2), text)
        painter.end()

    def __init__(self):
        super().__init__()
        self.draw_green_line = True  # 控制是否绘制绿色直线
        self.shader_program = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('OpenGL Lines')
        self.setGeometry(100, 100, 400, 400)

        # 添加按钮来切换绿色直线的显示
        toggle_button = QPushButton("Toggle Green Line", self)
        toggle_button.setGeometry(10, 10, 150, 30)
        toggle_button.clicked.connect(self.toggleGreenLine)

        zoom_in_button = QPushButton("Zoom In", self)
        zoom_in_button.setGeometry(10, 10, 75, 30)
        zoom_in_button.clicked.connect(self.zoomIn)

        zoom_out_button = QPushButton("Zoom Out", self)
        zoom_out_button.setGeometry(95, 10, 75, 30)
        zoom_out_button.clicked.connect(self.zoomOut)

    def toggleGreenLine(self):
        self.draw_green_line = not self.draw_green_line
        self.update()

    def zoomIn(self):
        # 放大畫面
        self.zoom_factor *= 1.1
        self.update()

    def zoomOut(self):
        # 縮小畫面
        self.zoom_factor /= 1.1
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    gl_widget = OpenGLWidget()
    window.setCentralWidget(gl_widget)
    window.setWindowTitle("OpenGL Lines")
    window.setGeometry(100, 100, 400, 400)
    window.show()
    sys.exit(app.exec_())
