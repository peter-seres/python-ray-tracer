import time
from PIL import Image
from numba import cuda
import numpy as np
import math
from main import Camera


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin, ray_direction):
    sphere_radius = 2.0

    # L vector: from ray origin to sphere center
    L0 = 10.0 - ray_origin[0]
    L1 = 0.0 - ray_origin[1]
    L2 = 0.0 - ray_origin[2]

    # R vector: ray direction
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


@cuda.jit(device=True)
def clip(intensity):
    intensity = int(round(intensity))
    return min(max(0, intensity), 255)


@cuda.jit(device=True)
def trace(ray_origin, ray_dir):
    """ Trace the ray and return the R, G, B values"""
    # Start with black
    R = 0.0
    G = 0.0
    B = 0.0

    R_obj = 0.0
    G_obj = 0.0
    B_obj = 0.0

    dist = intersect_ray_sphere(ray_origin, ray_dir)

    if dist > 0:
        R_obj = 255
        G_obj = 50
        B_obj = 50

    R = R + R_obj
    G = G + G_obj
    B = B + B_obj

    return clip(R), clip(G), clip(B)


@cuda.jit()
def render_kernel(pixel_array, rays):

    # Location of pixel
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds and run the ray trace for this pixel:
    if x <= pixel_array.shape[1] and y <= pixel_array.shape[2]:

        ray_origin = rays[0:3, x, y]
        ray_dir = rays[3:6, x, y]

        pixel_array[0, x, y], pixel_array[1, x, y], pixel_array[2, x, y] = trace(ray_origin, ray_dir)


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Generate rays:
    print('Generating Rays...')
    c = Camera(field_of_view=45)
    rays_host = np.zeros((6, w, h), dtype=np.float32)
    rays_host[3:6, :, :] = c.create_rays(w, h)

    # Make an empty pixel array:
    print('Allocating to GPU RAM...')
    A = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))
    rays = cuda.to_device(rays_host)

    # Setup the cuda kernel grid:
    threadsperblock = (32, 32)
    blockspergrid_x = int(np.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Compile it:
    print('Compiling...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays)
    end = time.time()
    print(f'Compile time: {1000*(end-start)} ms')

    # Render it:
    print('Rendering...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays)
    end = time.time()
    print(f'Render time: {1000*(end-start)} ms')

    # Save the image
    print('Saving...')
    pixel_array = np.zeros(shape=(A.shape[1], A.shape[2], 3), dtype=np.uint8)
    for RGB in range(3):
        pixel_array[:, :, RGB] = A[RGB, :, :]
    Image.fromarray(np.rot90(pixel_array)).save('cuda_output.png')

    return 0


if __name__ == '__main__':
    main()
