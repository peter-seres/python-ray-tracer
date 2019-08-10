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
        x = self.A.copy_to_host()

        mode = "RGB"
        size = x.shape[2], x.shape[1]
        obj = x.tobytes()
        rawmode = mode

        # print(size)

        image = PIL.Image.frombuffer(mode, size, obj, "raw", rawmode, 0, 1)
        return image

    def iter_pixel_array(self, A):
        for i in range(A.shape[1]):
            for j in range(A.shape[2]):
                yield i, j, A[:, i, j]

    def save_png(self, name):

        result = self.get_render()
        # Save the image to a .png file:
        im = PIL.Image.new("RGB", (result.shape[1], result.shape[2]), (255, 255, 255))
        for x, y, color in self.iter_pixel_array(result):
            im.putpixel((x, y), tuple(color))
        im.save(name)


class RenderWindow(arcade.Window):
    def __init__(self, width, height, scene):
        super().__init__(width=width, height=height)

        self.scene = scene
        self.set_update_rate(1./30)
        self.buffer = None

    def on_draw(self):

        # Draw the background texture
        if self.buffer is not None:
            arcade.draw_texture_rectangle(self.width // 2, self.height // 2,
                                          self.width, self.height, self.buffer)

    def my_load_texture(self):
        image = self.scene.get_render()
        result = arcade.Texture('render_result', image)

        return result

    def update(self, dt):
        # Move camera:
        self.scene.camera_euler[2] += 1

        # Update rays and render:
        self.scene.update_camera()
        self.scene.render()

        # Load result
        self.buffer = self.my_load_texture()


def main():
    # Resolution settings:
    w = 800
    h = 600

    scene = Scene(width=w, height=h)
    window = RenderWindow(width=w, height=h, scene=scene)

    arcade.run()

    return 0


main()
