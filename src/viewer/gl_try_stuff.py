from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import random
import numpy as np
from numba import cuda as cuda
import math

# import pycuda.autoinit
# import pycuda.driver as cuda
# from pycuda.compiler import SourceModule


@cuda.jit
def increment_a_2D_array(array):
    x, y = cuda.grid(2)
    if x < array.shape[0] and y < array.shape[1]:
        array[x, y] += 1


a = np.zeros((4, 4), dtype=np.float32)
a_gpu = cuda.to_device(a)                       # push 4x4 image to GPU.

# Use numba to call kernel:
threadsperblock = (4, 4)
blockspergrid_x = math.ceil(a.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(a.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
increment_a_2D_array[blockspergrid, threadsperblock](a_gpu)

a_res = a_gpu.copy_to_host()
passed = ((a + 1) == a_res).all()                 # compare to CPU calculation.
print("Test passed: %s" % (passed))


def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_POINTS)
    x = [0.0, 640.0, 320.0]
    y = [0.0, 0.0, 480.0]
    curx = 0
    cury = 320
    glVertex2f(curx, cury)
    for i in range(0, 2000):
        idx = random.randint(0, 2)
        curx = (curx + x[idx]) / 2.0
        cury = (cury + y[idx]) / 2.0
        glVertex2f(curx, cury)
    glEnd()
    glFlush()


glutInit()
glutInitWindowSize(640, 480)
glutCreateWindow("Sierpinski")
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
glutDisplayFunc(display)
glClearColor(1.0, 1.0, 1.0, 0.0)
glColor3f(0.0, 0.0, 0.0)
glPointSize(1.0)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluOrtho2D(0.0, 640.0, 0.0, 480.0)
glutMainLoop()
