import sys
import numpy as np
from dataclasses import dataclass

# Open GL imports:
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from OpenGL.GL.ARB.pixel_buffer_object import *

# Pycuda stuff:
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
import pycuda.tools as cuda_tools
from pycuda.compiler import SourceModule

cuda_driver.init()
global context
context = cuda_tools.make_default_context()
device = context.get_device()

# roll, pitch, yaw = [0.0] * 3
#
# # Global pointers
# cuda_module = None
# invert = None


@dataclass
class Square():
    position: list
    width: int

    def draw(self):

        x_left = self.position[0] - self.width/2
        x_right = self.position[0] + self.width/2
        y_top = self.position[1] + self.width/2
        y_bottom = self.position[1] - self.width/2

        glBegin(GL_QUADS)               # Begin the sketch
        glVertex2f(x_left, y_bottom)    # Coordinates for the bottom left point
        glVertex2f(x_right, y_bottom)   # Coordinates for the bottom right point
        glVertex2f(x_right, y_top)      # Coordinates for the top right point
        glVertex2f(x_left, y_top)       # Coordinates for the top left point
        glEnd()                         # Mark the end of drawing


class Application:
    def __init__(self):
        self.w, self.h = 500, 500
        self.the_square = Square([150, 150], 100)

        # Buffer pointers:
        self.source_pbo = None
        self.dest_pbo = None
        self.pycuda_dest_pbo = None
        self.pycuda_source_pbo = None

        # Texture:
        self.output_texture = None

        # CUDA process call:
        self.invert_cuda = None

    @staticmethod
    def graceful_exit():
        global context
        cuda_tools.clear_context_caches()
        context.pop()
        context = None
        sys.exit(0)

    def input_handler(self, key, x, y):
        if key == b'\x1b':
            self.graceful_exit()
        elif key == b'w':
            self.the_square.position[1] += 10
        elif key == b'a':
            self.the_square.position[0] -= 10
        elif key == b's':
            self.the_square.position[1] -= 10
        elif key == b'd':
            self.the_square.position[0] += 10

    def create_PBOs(self):
        data = np.zeros((self.w * self.h, 4), dtype=np.uint8)

        # Source
        self.source_pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.source_pbo)
        glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.pycuda_source_pbo = cuda_gl.BufferObject(int(self.source_pbo))

        # Dest
        self.dest_pbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.dest_pbo)
        glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.pycuda_dest_pbo = cuda_gl.BufferObject(int(self.dest_pbo))

        print(f'PBOs created: {self.pycuda_source_pbo}')

    def destroy_PBOs(self):
        for pbo in [self.source_pbo, self.dest_pbo, self.pycuda_source_pbo, self.pycuda_dest_pbo]:
            glBindBuffer(GL_ARRAY_BUFFER, int(pbo))
            glDeleteBuffers(1, int(pbo))
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        self.source_pbo = None
        self.dest_pbo = None
        self.pycuda_source_pbo = None
        self.pycuda_dest_pbo = None

    def create_texture(self):
        self.output_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.output_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        print(f'Texture created: {self.output_texture}')

    def process(self):
        grid_dimensions = (self.w // 16, self.h // 16)

        source_mapping = self.pycuda_source_pbo.map()
        dest_mapping = self.pycuda_dest_pbo.map()

        self.invert_cuda.prepared_call(grid_dimensions, (16, 16, 1),
                                       source_mapping.device_ptr(),
                                       dest_mapping.device_ptr())

        cuda_driver.Context.synchronize()

        source_mapping.unmap()
        dest_mapping.unmap()

    def process_image(self):
        self.pycuda_source_pbo.unregister()

        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, int(self.source_pbo))

        # read data into pbo. note: use BGRA format for optimal performance
        glReadPixels(
            0,  # start x
            0,  # start y
            self.w,  # end   x
            self.h,  # end   y
            GL_BGRA,  # format
            GL_UNSIGNED_BYTE,  # output type
            ctypes.c_void_p(0))

        pycuda_source_pbo = cuda_gl.BufferObject(int(self.source_pbo))

        self.process()

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, int(self.dest_pbo))
        glBindTexture(GL_TEXTURE_2D, self.output_texture)

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        self.w, self.h,
                        GL_BGRA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

    def show_screen(self):
        global context

        # 1) Render scene:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glViewport(0, 0, self.w, self.h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0.0, self.w, 0.0, self.h, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glColor3f(0.2, 0.2, 0.5)
        self.the_square.draw()

        # 2) CUDA Process:
        self.process_image()

        # 3) Display image:
        glEnable(GL_TEXTURE_2D)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glViewport(0, 0, self.w, self.h)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1.0, -1.0, 0.5)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, -1.0, 0.5)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 1.0, 0.5)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1.0, 1.0, 0.5)
        glEnd()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)
        glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)

    def main(self):

        # Create window:
        glutInit()
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.w, self.h)
        window = glutCreateWindow("PyCuda GL Interop Example")

        # Set the loop functions:
        glutDisplayFunc(self.show_screen)
        glutIdleFunc(self.show_screen)
        glutKeyboardFunc(self.input_handler)
        self.create_texture()

        # Cuda OpenGL Interop:

        cuda_module = SourceModule("""
        __global__ void invert(unsigned char *source, unsigned char *dest)
        {
          int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
          int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
          int threads_in_block = blockDim.x * blockDim.y;
          //Since the image is RGBA we multiply the index 4.
          //We'll only use the first 3 (RGB) channels though
          int idx              = 4 * (threads_in_block * block_num + thread_num);
          dest[idx  ] = 255 - source[idx  ];
          dest[idx+1] = 255 - source[idx+1];
          dest[idx+2] = 255 - source[idx+2];
        }
        """)

        self.invert_cuda = cuda_module.get_function("invert")
        self.invert_cuda.prepare("PP")

        self.create_PBOs()

        glutMainLoop()

        return 0


if __name__ == "__main__":
    app = Application()
    app.main()
