import time

import arcade
from main import generate_rays, generate_scene
from cuda_ray_tracing import *
import numpy as np
import PIL.Image
import PIL.ImageOps


class Scene:
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

    def update_camera(self):
        # Set up the camera:

        camera_origin_host, camera_rotation_host, pixel_locations_host = \
            generate_rays(self.width, self.height, camera_rotation=self.camera_euler, camera_position=self.camera_position)

        self.origin = cuda.to_device(camera_origin_host)
        self.camera_R = cuda.to_device(camera_rotation_host)
        self.pixel_locations = cuda.to_device(pixel_locations_host)

        self.ray_dir()

    def ray_dir(self):

        ray_dir_kernel[self.blockspergrid, self.threadsperblock](self.pixel_locations,
                                                                 self.rays,
                                                                 self.origin,
                                                                 self.camera_R)

    def render(self):
        render_kernel[self.blockspergrid, self.threadsperblock](self.A,
                                                                self.rays,
                                                                self.spheres,
                                                                self.lights,
                                                                self.planes,
                                                                self.ambient,
                                                                self.lambert,
                                                                self.reflect,
                                                                self.ref_depth)

    def get_render(self):
        start = time.time()
        x = self.A.copy_to_host(stream=self.stream)
        end = time.time()
        y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

        for i in range(3):
            y[:, :, i] = x[i, :, :]

        image = PIL.Image.fromarray(y.astype(np.uint8), mode='RGB')
        image = image.rotate(270)
        image = PIL.ImageOps.mirror(image)

        # print(1000*(end-start), 'ms')

        return image


class RenderWindow(arcade.Window):
    def __init__(self, width, height, scene):
        super().__init__(width=width, height=height)

        self.scene = scene
        self.set_update_rate(1./30)
        self.buffer = None

        self.last_time = time.time()

    def on_draw(self):
        # Draw the background texture
        if self.buffer is not None:
            self.buffer.draw(center_x=self.width//2, center_y=self.height//2,
                             width=self.width, height=self.height)
        now = time.time()
        print(1000*(now - self.last_time), 'ms')
        self.last_time = now

    def update(self, dt):
        # Move camera:
        # self.scene.camera_position[0] += 1
        # self.scene.camera_euler[2] += 1

        # Update rays and render:
        # self.scene.update_camera()
        self.scene.render()

        # Load result
        image = self.scene.get_render()
        self.buffer = arcade.Texture('render_result', image)


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    scene = Scene(width=w, height=h)
    window = RenderWindow(width=w, height=h, scene=scene)

    arcade.run()

    return 0


main()
