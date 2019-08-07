import time
from PIL import Image
from numba import cuda
import numpy as np
import math
from main import Camera
from scene_objects import Sphere


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin, ray_direction, sphere_origin, sphere_radius):

    # L vector: from ray origin to sphere center
    L0 = sphere_origin[0] - ray_origin[0]
    L1 = sphere_origin[1] - ray_origin[1]
    L2 = sphere_origin[2] - ray_origin[2]

    # R vector: ray direction
    R0 = ray_direction[0]
    R1 = ray_direction[1]
    R2 = ray_direction[2]

    # t_ca = np.dot(L, ray_direction)
    t_ca = L0 * R0 + L1 * R1 + L2 * R2

    if t_ca < 0:
        return -999.0

    d = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2 - t_ca**2)

    if d < 0 or d > sphere_radius:
        return -999.0

    t_hc = math.sqrt(sphere_radius ** 2 - d ** 2)

    t_0 = t_ca - t_hc
    t_1 = t_ca + t_hc

    if t_0 < 0:
        t_0 = t_1
        if t_0 < 0:
            return -999.0

    return t_0


@cuda.jit(device=True)
def get_intersection(ray_origin, ray_dir, spheres):

    intersect_index = 404
    intersect_distance = 404.0

    for sphere_index in range(spheres.shape[1]):
        sphere_origin = spheres[0:3, sphere_index]
        sphere_radius = spheres[3, sphere_index]

        dist = intersect_ray_sphere(ray_origin, ray_dir, sphere_origin, sphere_radius)

        if dist != -999.0 and (intersect_index == 404 or dist < intersect_distance):
            intersect_index = sphere_index
            intersect_distance = dist

    return intersect_index, intersect_distance


@cuda.jit(device=True)
def clip(intensity):
    intensity = int(round(intensity))
    return min(max(0, intensity), 255)


@cuda.jit(device=True)
def trace(ray_origin, ray_dir, spheres, lights):
    """ Trace the ray and return the R, G, B values"""

    # Start with black
    R = 0.0
    G = 0.0
    B = 0.0

    R_obj = 0.0
    G_obj = 0.0
    B_obj = 0.0

    dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, 0], spheres[3, 0])

    if dist > 0:
        R_obj = spheres[4, 0]
        G_obj = spheres[5, 0]
        B_obj = spheres[6, 0]

    R = R + R_obj
    G = G + G_obj
    B = B + B_obj

    return clip(R), clip(G), clip(B)


@cuda.jit
def render_kernel(pixel_array, rays, spheres, lights):

    # Location of pixel
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds and run the ray trace for this pixel:
    if x <= pixel_array.shape[1] and y <= pixel_array.shape[2]:

        ray_origin = rays[0:3, x, y]
        ray_dir = rays[3:6, x, y]

        pixel_array[0, x, y], pixel_array[1, x, y], pixel_array[2, x, y] = trace(ray_origin, ray_dir, spheres, lights)


def main():
    # Resolution settings:
    w = 500
    h = 500

    # Generate rays:
    c = Camera(field_of_view=45)
    rays_host = np.zeros((6, w, h), dtype=np.float32)
    rays_host[3:6, :, :] = c.create_rays(w, h)

    # Generate sphere matrix:
    spheres_host = np.zeros((7, 1), dtype=np.float32)

    spheres_host[0:3, 0] = np.array([5.0, 0, 0])
    spheres_host[3, 0] = 1.0
    spheres_host[4:7, 0] = np.array([255, 78, 72])

    # Make an empty pixel array:
    A = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))
    rays = cuda.to_device(rays_host)
    spheres = cuda.to_device(spheres_host)
    lights = cuda.to_device(np.zeros((3, 1), dtype=np.float32))

    # Setup the cuda kernel grid:
    threadsperblock = (32, 32)
    blockspergrid_x = int(np.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Render using the kernel
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, lights)
    end = time.time()

    print(f'Render+compile time: {1000*(end-start)} ms')

    # Save the image
    pixel_array = np.zeros(shape=(A.shape[1], A.shape[2], 3), dtype=np.uint8)
    for RGB in range(3):
        pixel_array[:, :, RGB] = A[RGB, :, :]
    Image.fromarray(np.rot90(pixel_array)).save('cuda_output.png')

    return 0


if __name__ == '__main__':
    main()
