import time
import arcade
from src.main import generate_rays, generate_scene, rotation_y, rotation_z
from src.cuda_ray_tracing import *
import numpy as np
import PIL.Image
import PIL.ImageOps
import threading

UPDATE_RATE = 1./60


class Renderer:
    def __init__(self, width, height):
        # Resolution settings:
        self.width = width
        self.height = height

        # Shader Settings:
        self.ambient = 0.1
        self.lambert = 0.6
        self.reflect = 0.5
        self.ref_depth = 4

        # GPU thread settings:
        self.threadsperblock = (16, 16)

        # Generate Scene:
        spheres_host, lights_host, planes_host = generate_scene()
        rays_host = np.zeros((6, width, height), dtype=np.float32)

        self.spheres = cuda.to_device(spheres_host)
        self.lights = cuda.to_device(lights_host)
        self.planes = cuda.to_device(planes_host)

        # Make an empty pixel array:
        self.A = cuda.to_device(np.zeros((3, width, height), dtype=np.uint8))
        blockspergrid_x = int(np.ceil(self.A.shape[1] / self.threadsperblock[0]))
        blockspergrid_y = int(np.ceil(self.A.shape[2] / self.threadsperblock[1]))
        self.blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Set up the camera:
        self.camera_euler = [0, -20, 0]
        self.camera_position = [-4, 0, 4]

        camera_origin_host, camera_rotation_host, pixel_locations_host = \
            generate_rays(width, height, camera_rotation=self.camera_euler, camera_position=self.camera_position)
        self.origin = cuda.to_device(camera_origin_host)
        self.camera_R = cuda.to_device(camera_rotation_host)
        self.pixel_locations = cuda.to_device(pixel_locations_host)

        self.rays = cuda.to_device(rays_host)

        # Compile the functions:
        self.ray_dir()
        self.render()

        self.stream = cuda.stream()

        self.image = None
        self.updated = False
        self.texture = None

    def update_camera_position(self):
        self.origin = cuda.to_device(self.camera_position)
        self.ray_dir()

    def update_camera_rotation(self):

        RZ = rotation_z(self.camera_euler[2])
        RY = rotation_y(self.camera_euler[1])

        ROT = np.matmul(RZ, RY)

        self.camera_R = cuda.to_device(ROT)
        self.ray_dir()

    def ray_dir(self):
        ray_dir_kernel[self.blockspergrid, self.threadsperblock](self.pixel_locations, self.rays, self.origin, self.camera_R)

    def render(self):
        render_kernel[self.blockspergrid, self.threadsperblock](self.A, self.rays, self.spheres, self.lights, self.planes,
                                                                self.ambient, self.lambert, self.reflect, self.ref_depth)

    def get_render(self):
        x = self.A.copy_to_host(stream=self.stream)
        y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

        for i in range(3):
            y[:, :, i] = x[i, :, :]

        image = PIL.Image.fromarray(y.astype(np.uint8), mode='RGB')
        image = image.rotate(270)
        image = PIL.ImageOps.mirror(image)

        return image

    def render_loop(self, arcade_window):
        last_time = time.time()
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt < UPDATE_RATE:
                time.sleep(UPDATE_RATE-dt)

            self.render()
            self.image = self.get_render()
            self.texture = arcade.Texture('result', self.image)


class RenderWindow(arcade.Window):
    def __init__(self, width, height, renderer):
        super().__init__(width=width, height=height)

        self.renderer = renderer
        self.set_update_rate(1./30)
        self._buffer = None

        self.FPS = 0.0

        self.started = False

    def on_draw(self):
        # Draw the background texture
        if self.started:
            # print('do i get here?')
            from pyglet.gl import GLException
            try:
                self._buffer.draw(center_x=self.width//2, center_y=self.height//2,
                                  width=self.width, height=self.height)
            except GLException:
                print('whoopsie')

            arcade.draw_text(text='FPS: '+'{:.1f}'.format(self.FPS), start_x=30, start_y=self.height-35,
                             color=arcade.color.WHITE, font_size=22)
        else:
            arcade.draw_rectangle_filled(center_x=self.width//2, center_y=self.height//2,
                                         width=self.width, height=self.height, color=arcade.color.BLUE)

    def set_buffer(self, texture):
        self._buffer = texture
        self.started = True

    def update(self, dt):
        self.FPS = (1 / dt)

        self.set_buffer(self.renderer.texture)

    def on_key_press(self, symbol, mod):

        camera_speed = 0.1
        camera_rot_speed = 5

        if symbol == arcade.key.W:
            self.renderer.camera_position[0] += camera_speed
            self.renderer.update_camera_position()
        elif symbol == arcade.key.S:
            self.renderer.camera_position[0] += -camera_speed
            self.renderer.update_camera_position()
        elif symbol == arcade.key.A:
            self.renderer.camera_position[1] += camera_speed
            self.renderer.update_camera_position()
        elif symbol == arcade.key.D:
            self.renderer.camera_position[1] += -camera_speed
            self.renderer.update_camera_position()
        elif symbol == arcade.key.LSHIFT:
            self.renderer.camera_position[2] += camera_speed
            self.renderer.update_camera_position()
        elif symbol == arcade.key.LCTRL:
            self.renderer.camera_position[2] += -camera_speed
            self.renderer.update_camera_position()

        # ROTATE LEFT
        elif symbol == arcade.key.Q:
            self.renderer.camera_euler[2] += camera_rot_speed
            self.renderer.update_camera_rotation()

        # ROTATE RIGHT
        elif symbol == arcade.key.E:
            self.renderer.camera_euler[2] += -camera_rot_speed
            self.renderer.update_camera_rotation()

        # PITCH DOWN
        elif symbol == arcade.key.K:
            self.renderer.camera_euler[1] += -camera_rot_speed
            self.renderer.update_camera_rotation()

        # PITCH UP
        elif symbol == arcade.key.M:
            self.renderer.camera_euler[1] += camera_rot_speed
            self.renderer.update_camera_rotation()


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Ray tracing renderer:
    renderer = Renderer(width=w, height=h)

    # Arcade GUI window
    window = RenderWindow(width=w, height=h, renderer=renderer)

    # Loop for rendering on a separate thread
    thread_get = threading.Thread(target=renderer.render_loop, args=(window, ), daemon=True)

    # Start the renderer Loop thread
    thread_get.start()

    # Start the GUI window
    arcade.run()

    return 0


if __name__ == "__main__":
    main()
