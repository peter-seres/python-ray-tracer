import time
from PIL import Image
from numba import cuda
import numpy as np
import math
from main import Camera


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin, ray_direction, sphere_origin, sphere_radius):
    L0 = sphere_origin[0] - ray_origin[0]
    L1 = sphere_origin[1] - ray_origin[1]
    L2 = sphere_origin[2] - ray_origin[2]

    R0 = ray_direction[0]
    R1 = ray_direction[1]
    R2 = ray_direction[2]

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


# @cuda.jit(device=True)
# def clip_all(color):
#     R = min(max(0, int(round(color[0]))), 255)
#     G = min(max(0, int(round(color[1]))), 255)
#     B = min(max(0, int(round(color[2]))), 255)
#     return R, G, B


@cuda.jit(device=True)
def clip(color):
    return min(max(0, int(round(color))), 255)
#
#
# @cuda.jit(device=True)
# def add(tuple1, tuple2):
#     return tuple1[0] + tuple2[0], tuple1[1] + tuple2[1], tuple1[2] + tuple2[2]


@cuda.jit(device=True)
def trace(ray_origin, ray_dir, spheres):
    """ Trace the ray and return the R, G, B values"""
    # Start with black
    R = 0.0
    G = 0.0
    B = 0.0

    R_obj = 0.0
    G_obj = 0.0
    B_obj = 0.0

    intersect_dist = 999.0
    for idx in range(spheres.shape[1]):
        dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])

        if intersect_dist > dist > 0:
            R_obj = spheres[4, idx]
            G_obj = spheres[5, idx]
            B_obj = spheres[6, idx]

    R = R + R_obj
    G = G + G_obj
    B = B + B_obj

    # Get point of intersection
    # Get point->light vector
    # Get intersection normal vector
    # Get reflection direction vector

    # Check shadow: if not in shadow:
    #       Do lambert shading

    lambert_intensity = 0.0

    R = R + R_obj * lambert_intensity
    G = G + G_obj * lambert_intensity
    B = B + B_obj * lambert_intensity

    # Calculate reflected light by calling trace again recursively

    return clip(R), clip(B), clip(G)


@cuda.jit()
def render_kernel(pixel_array, rays, spheres):

    # Location of pixel
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds and run the ray trace for this pixel:
    if x <= pixel_array.shape[1] and y <= pixel_array.shape[2]:

        ray_origin = rays[0:3, x, y]
        ray_dir = rays[3:6, x, y]

        pixel_array[0, x, y], pixel_array[1, x, y], pixel_array[2, x, y] = trace(ray_origin, ray_dir, spheres)


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Generate rays:
    print('Generating Rays...')
    c = Camera(field_of_view=45)
    rays_host = np.zeros((6, w, h), dtype=np.float32)
    rays_host[3:6, :, :] = c.create_rays(w, h)

    # Define sphere positions, radiii and colors
    spheres_host = np.zeros((7, 2), dtype=np.float32)

    spheres_host[0:3, 0]    = np.array([10.0, 0.0, -0.5])
    spheres_host[3, 0]      = 2.5
    spheres_host[4:7, 0]    = np.array([255, 0, 0])

    spheres_host[0:3, 1]    = np.array([5.0, 0.5, 0.0])
    spheres_host[3, 1]      = 1.0
    spheres_host[4:7, 1]    = np.array([0, 0, 255])

    # Make an empty pixel array:
    A = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))
    rays = cuda.to_device(rays_host)
    spheres = cuda.to_device(spheres_host)

    # Setup the cuda kernel grid:
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Compile it:
    print('Compiling...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres)
    end = time.time()
    print(f'Compile time: {1000*(end-start)} ms')

    # Render it:
    print('Rendering...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres)
    end = time.time()
    print(f'Render time: {1000*(end-start)} ms')

    # Save the image
    name = 'cuda_output.png'
    print(f'Saving image to {name}')
    pixel_array = np.zeros(shape=(A.shape[1], A.shape[2], 3), dtype=np.uint8)
    for RGB in range(3):
        pixel_array[:, :, RGB] = A[RGB, :, :]
    Image.fromarray(np.rot90(pixel_array)).save(name)

    return 0


if __name__ == '__main__':
    main()
