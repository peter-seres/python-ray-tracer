import numpy as np
from cuda_ray_tracing import *
import time
from PIL import Image

RED = [255, 70, 70]
GREEN = [70, 255, 70]
BLUE = [70, 70, 255]
YELLOW = [255, 255, 70]


def generate_scene():

    sphere1 = {'origin': [6., 1., 1.], 'radius': 0.8, 'color': RED}
    sphere2 = {'origin': [3., -1., 1.], 'radius': 0.8, 'color': GREEN}
    sphere3 = {'origin': [4., -1., -1.], 'radius': 0.8, 'color': BLUE}
    sphere4 = {'origin': [6., 1., -1.], 'radius': 0.8, 'color': YELLOW}

    light1 = {'origin': [0.0, -4.0, 2.0]}

    sphere_list = [sphere1, sphere2, sphere3, sphere4]
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
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])                                 # Rotation matrix
    AR = width/height
    yy, zz = np.mgrid[AR:-AR:complex(0, width), 1:-1:complex(0, height)]              # Create pixel grid
    xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(field_of_view) / 2))      # Distance of grid from origin
    pixel_locations = np.array([xx, yy, zz])

    return camera_position, R, pixel_locations


def iter_pixel_array(A):
    for x in range(A.shape[1]):
        for y in range(A.shape[2]):
            yield x, y, A[:, x, y]


def main():
    # Resolution settings:
    w = 1920
    h = 1080

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
    N = 1000
    print(f'Rendering {N} times...')
    start = time.time()
    for i in range(N):
        render_kernel[blockspergrid, threadsperblock](A, rays, spheres, lights)
    end = time.time()
    print(f'Render time: {1000*(end-start)} ms')

    result = np.array(A).astype(np.uint8)

    # Save the image to a .png file:
    name = 'output.png'
    print(f'Saving image to {name}')
    im = Image.new("RGB", (result.shape[1], result.shape[2]), (255, 255, 255))
    for x, y, color in iter_pixel_array(result):
        im.putpixel((x, y), tuple(color))
    im.save(name)


if __name__ == '__main__':
    main()
