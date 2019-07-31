import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class SphereCP:
    origin: cp.ndarray
    radius: float
    color: cp.ndarray


@dataclass
class RayCP:
    origin: cp.ndarray
    dir: cp.ndarray


@dataclass
class Sphere:
    origin: np.ndarray
    radius: float
    color: np.ndarray


@dataclass
class Ray:
    origin: np.ndarray
    dir: np.ndarray


def intersect_ray_sphere_cp(ray, sphere):
    a = cp.dot(ray.dir, ray.dir)
    b = 2 * cp.dot(ray.dir, ray.origin - sphere.origin)
    c = cp.dot(ray.origin - sphere.origin, ray.origin - ray.origin) - sphere.radius * sphere.radius

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        return True
    else:
        return False


def intersect_ray_sphere_np(ray, sphere):
    a = np.dot(ray.dir, ray.dir)
    b = 2 * np.dot(ray.dir, ray.origin - sphere.origin)
    c = np.dot(ray.origin - sphere.origin, ray.origin - ray.origin) - sphere.radius * sphere.radius

    discriminant = b * b - 4 * a * c

    if discriminant >= 0:
        return True
    else:
        return False


def generate_rays(width, height, origin=None, field_of_view=45):
    """ General direction is [1, 0, 0] for now. Pointing to x axis. """

    fig = plt.figure()
    ax = Axes3D(fig)

    if origin is None:
        origin = np.array([0, 0, 0])

    cam_i = np.array([1, 0, 0])
    cam_j = np.array([0, -1, 0])
    cam_k = np.array([0, 0, 1])

    R = np.array([cam_i, cam_j, cam_k])

    # Generate pixel locations in body coordinate-frame:
    x = 1 / np.tan(np.radians(field_of_view) / 2)  # Distance of pixels from origin
    yy = np.linspace(-1, 1, width)  # Distances of pixels along width
    zz = np.linspace(-1, 1, height)  # Distances ff pixels along height

    ax.quiver(origin[0], origin[1], origin[2], cam_i[0], cam_i[1], cam_i[2], length=0.5, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], cam_j[0], cam_j[1], cam_j[2], length=0.5, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], cam_k[0], cam_k[1], cam_k[2], length=0.5, normalize=True)

    for j, y in enumerate(yy):
        for k, z in enumerate(zz):
            pix_loc_local = [x, y, z]
            # print(type(x), type(y), type(z))
            pix_loc_global = np.matmul(R, np.transpose(pix_loc_local))

            ray_direction = pix_loc_global - origin
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            ray = Ray(origin=np.asarray(origin), dir=np.asarray(ray_direction))

            yield j, k, ray


def main():
    # Image resolution
    w = 300     # image width (pixels)
    h = 300     # image height (pixels)

    # The scene contains a and a sphere:
    sphere = Sphere(origin=np.array([100, 0, 0]), radius=2, color=np.array([255, 0, 0]))

    # Cast rays and look for intersection:
    pixel_array = np.zeros(shape=(w, h, 3), dtype=np.uint8)

    start = time.time()
    for j, k, ray in generate_rays(w, h):
        intersect = intersect_ray_sphere_np(ray, sphere)

        if intersect:
            pixel_array[j, k] = sphere.color
        else:
            pixel_array[j, k] = np.array([50, 50, 50])
    end = time.time()

    result = pixel_array
    # result = cp.asnumpy(pixel_array)

    # Save the data to a png file:
    image = Image.fromarray(np.rot90(result))
    image.save('HDresult.png')

    print(f'Ray tracing took {end-start} seconds with {w*h} rays')


main()
