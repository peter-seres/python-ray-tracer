from dataclasses import dataclass
import numpy as np
from PIL import Image
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functools import wraps
from time import time


def timed(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        end = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, end-start))
        return result
    return wrap


@dataclass
class Sphere:
    origin: np.ndarray
    radius: float
    color: np.ndarray


@dataclass
class Ray:
    origin: np.ndarray
    dir: np.ndarray


@dataclass
class Camera:
    origin: np.ndarray
    field_of_view: float

    # todo: Allow camera coordinate system to be rotated
    cam_i: np.ndarray = np.array([1, 0, 0])
    cam_j: np.ndarray = np.array([0, -1, 0])
    cam_k: np.ndarray = np.array([0, 0, 1])
    R = np.array([cam_i, cam_j, cam_k])

    def generate_pixel_locations(self, width, height):
        # Generate pixel locations in camera-body coordinate-frame:
        x = 1 / np.tan(np.radians(self.field_of_view)/2)    # Distances of pixels from camera origin
        yy = np.linspace(-1, 1, width)                      # Distances of pixels along width
        zz = np.linspace(-1, 1, height)                     # Distances of pixels along height

        # Loop through the pixel locations and to generate a ray for pixel location j, k
        for j, y in enumerate(yy):
            for k, z in enumerate(zz):
                pix_loc_local = [x, y, z]
                pix_loc_global = self.origin + np.matmul(self.R, np.transpose(pix_loc_local))

                yield pix_loc_global, j, k

    def generate_rays(self, width, height):
        for pix_loc_global, j, k in self.generate_pixel_locations(width, height):

            ray_direction = (pix_loc_global - self.origin)
            ray_direction /= np.linalg.norm(ray_direction)

            ray = Ray(origin=self.origin, dir=ray_direction)

            yield ray, j, k

    def enum_coordinate_system(self):
        for vec in [self.cam_i, self.cam_j, self.cam_k]:
            yield self.origin[0], self.origin[1], self.origin[2], vec[0], vec[1], vec[2]

    def generate_corners(self):
        # Generate pixel locations in camera-body coordinate-frame:
        x = 1 / np.tan(np.radians(self.field_of_view)/2)    # Distances of pixels from camera origin
        yy = np.linspace(-1, 1, 2)                          # Distances of pixels along width
        zz = np.linspace(-1, 1, 2)                          # Distances of pixels along height

        for y in yy:
            for z in zz:
                pix_loc_local = [x, y, z]
                yield self.origin + np.matmul(self.R, np.transpose(pix_loc_local))


class Scene:
    def __init__(self):
        self.objects = []
        self.camera = Camera(origin=np.array([0, 0, 0]), field_of_view=45)

    def add_object(self, obj):
        self.objects.append(obj)

    @timed
    def render(self, width, height):
        pixel_array = np.zeros(shape=(width, height, 3), dtype=np.uint8)

        for ray, j, k in self.camera.generate_rays(width, height):

            dist = np.inf
            closes_obj_index = 0
            for obj_index, obj in enumerate(self.objects):
                dist_ray = intersect_ray_sphere(ray, obj)

                if dist_ray < dist:
                    dist = dist_ray
                    closes_obj_index = obj_index

            if dist == np.inf:
                color = np.array([0, 0, 0])
            else:
                color = self.objects[closes_obj_index].color

            pixel_array[j, k] = color

        return pixel_array

    def plot(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        # Plot camera coordinate system:
        for x, y, z, i, j, k in self.camera.enum_coordinate_system():
            ax.quiver(x, y, z, i, j, k, length=0.2)

        # Plot spheres in the scene:
        for obj in self.objects:
            ax.scatter(obj.origin[0], obj.origin[1], obj.origin[2], s=5)

        # # Plot pixel locations:
        for corner in self.camera.generate_corners():
            ax.scatter(corner[0], corner[1], corner[2], s=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


def intersect_ray_sphere(ray, sphere):
    a = np.dot(ray.dir, ray.dir)
    b = 2 * np.dot(ray.dir, ray.origin - sphere.origin)
    c = np.dot(ray.origin - sphere.origin, ray.origin - sphere.origin) - sphere.radius * sphere.radius

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        distSqrt = np.sqrt(discriminant)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)

        if t1 >= 0:
            return t1 if t0 < 0 else t0
    else:
        return np.inf


def main():
    # Resolution settings:
    w = 800
    h = 600

    # Create scene with camera:
    scene = Scene()

    # Add a sphere object in the middle:
    sphere = Sphere(origin=np.array([10, 0, 0]), radius=1, color=np.array([200, 85, 85]))
    scene.add_object(sphere)

    # Render the image:
    pixel_array = scene.render(w, h)

    # Save the image to a .png file:
    Image.fromarray(np.rot90(pixel_array)).save('output.png')

    # Plot scene setup:
    scene.plot()


if __name__ == '__main__':
    main()
