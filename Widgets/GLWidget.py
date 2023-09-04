from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QOpenGLShader, QOpenGLShaderProgram, QMatrix4x4, QPainter, QColor, QFont
from PyQt5.QtCore import Qt
from OpenGL.GL import *
import numpy as np


class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(100)
        self.setMinimumHeight(100)
        self.visible = False

        self.x_scale = 1.0
        self.y_scale = 1.0
        self.z_scale = 1.0

        self.x_transform = 0.0
        self.y_transform = 0.0
        self.z_transform = 0.0

        self.x_rotate = 0.0
        self.y_rotate = 0.0
        self.z_rotate = 0.0

    def initializeGL(self):
        glClearColor(0.35, 0.35, 0.35, 0.8)  # background color
        glEnable(GL_DEPTH_TEST)

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

        # Create shader
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShader(vertex_shader)
        self.shader_program.addShader(fragment_shader)
        self.shader_program.link()
        self.model_matrix = QMatrix4x4()

        # create vbo

    def create_vbo(self, n):
        # # create vbo
        # self.position_vbo = glGenBuffers(1)
        # self.color_vbo = glGenBuffers(1)
        self.vbo = glGenBuffers(n)

    # def pre_paintGL(self, vbo):
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     # 使用修改后的着色器程序
    #     self.shader_program.bind()
    #     glBindBuffer(GL_ARRAY_BUFFER, vbo)
    #     self.model_matrix.setToIdentity()
    #     self.model_matrix.scale(self.zoom_factor, self.zoom_factor)
    #     self.shader_program.setUniformValue("model_matrix", self.model_matrix)
