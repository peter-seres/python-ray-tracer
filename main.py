import numpy as np
from cuda_ray_tracing import *
import time
from PIL import Image


def generate_scene():

    # todo: easier user input to add more spheres, and lights

    sphere1 = {'origin': [5., -1., 1.], 'radius': 1.0, 'color': [255, 70, 70]}

    sphere2 = {'origin': [10., 2.0, 0], 'radius': 1.0, 'color': [70, 255, 70]}
    sphere3 = {'origin': [10., -2.0, 2.0], 'radius': 1.0, 'color': [70, 70, 255]}

    light1 = {'origin': [0.0, -2.0, 1.0]}

    sphere_list = [sphere1]
    light_list = [light1]

    # Build the sphere data array
    spheres = np.zeros((7, len(sphere_list)), dtype=np.float32)
    for i, s in enumerate(sphere_list):
        spheres[0:3, i]    = np.array(s['origin'])
        spheres[3, i]      = s['radius']
        spheres[4:7, i]    = np.array(s['color'])

    # Build the light data array
    lights = np.zeros((3, len(light_list)), dtype=np.float32)
    for i, light in enumerate(light_list):
        # print(i)
        lights[0:3, i]    = np.array(np.array(light['origin']))

    return spheres, lights


def generate_rays(width, height, field_of_view=45, camera_position=None):
    if camera_position is None:
        camera_position = [0, 0, 0]                                                 # Default at origin
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])                               # Rotation matrix
    yy, zz = np.mgrid[-1:1:complex(0, width), -1:1:complex(0, height)]              # Create pixel grid
    xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(field_of_view) / 2))      # Distance of grid from origin
    pixel_locations = np.array([xx, yy, zz])

    return camera_position, R, pixel_locations


def main():
    # Resolution settings:
    w = 1000
    h = 1000

    # Generate scene:
    spheres_host, lights_host = generate_scene()
    spheres = cuda.to_device(spheres_host)
    lights = cuda.to_device(lights_host)

    # Generate rays:
    camera_origin_host, camera_rotation_host, pixel_locations_host = generate_rays(w, h)

    # Empty rays array to be filled in by the ray-direction kernel
    rays_host = np.zeros((6, w, h), dtype=np.float32)

    # Data needed to get the ray direction kernel running
    origin = cuda.to_device(camera_origin_host)
    camera_rotation = cuda.to_device(camera_rotation_host)
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
    # pixel_array = A.copy_to_host()

    pixel_array = np.empty(shape=(w, h, 3), dtype=np.uint8)

    for RGB in range(3):
        pixel_array[:, :, RGB] = A[RGB, :, :]

    Image.fromarray(pixel_array).save(name)

    return pixel_array


if __name__ == '__main__':
    x = main()
