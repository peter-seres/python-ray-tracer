import numpy as np
from cuda_ray_tracing import *
import time
from camera import Camera
from PIL import Image


def generate_scene(w, h):
    # Define sphere positions, radiii and colors
    spheres = np.zeros((7, 2), dtype=np.float32)

    spheres[0:3, 0]    = np.array([10.0, 0.0, -0.5])
    spheres[3, 0]      = 2.5
    spheres[4:7, 0]    = np.array([255, 0, 0])

    spheres[0:3, 1]    = np.array([5.0, 0.5, 0.0])
    spheres[3, 1]      = 1.0
    spheres[4:7, 1]    = np.array([0, 0, 255])

    # Define the light positions:
    lights = np.zeros((3, 1), dtype=np.float32)
    lights[0:3, 0] = np.array([0.0, -1.0, 0.5])
    return spheres, lights


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Generate scene:
    spheres_host, lights_host = generate_scene(w, h)
    spheres = cuda.to_device(spheres_host)
    lights = cuda.to_device(lights_host)

    # Generate rays:
    camera = Camera(field_of_view=45)
    pixel_locations_host = camera.create_pixel_locations(w, h)

    # Empty rays array to be filled in by the ray-direction kernel
    rays_host = np.zeros((6, w, h), dtype=np.float32)

    # Data needed to get the ray direction kernel running
    origin = cuda.to_device(camera.origin)
    camera_rotation = cuda.to_device(camera.R)
    pixel_locations = cuda.to_device(pixel_locations_host)
    rays = cuda.to_device(rays_host)

    # Make an empty pixel array:
    A = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))

    # Setup the cuda kernel grid:
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Rays it:
    print('Generating rays array')
    start = time.time()
    ray_dir_kernel[blockspergrid, threadsperblock](pixel_locations, rays, origin, camera_rotation)
    end = time.time()
    print(f'Compile+generate time: {1000*(end-start)} ms')

    # Compile it:
    print('Compiling renderer')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, lights)
    end = time.time()
    print(f'Compile time: {1000*(end-start)} ms')

    # Render it:
    print('Rendering...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, lights)
    end = time.time()
    print(f'Render time: {1000*(end-start)} ms')

    # Save the image to a .png file:
    name = 'output.png'
    print(f'Saving image to {name}')
    pixel_array = np.zeros(shape=(A.shape[1], A.shape[2], 3), dtype=np.uint8)
    for RGB in range(3):
        pixel_array[:, :, RGB] = A[RGB, :, :]
    Image.fromarray(np.rot90(pixel_array)).save(name)

    return 0


if __name__ == '__main__':
    main()
