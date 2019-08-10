import numpy as np
from cuda_ray_tracing import *
import time
from PIL import Image

RED = [255, 70, 70]
GREEN = [70, 255, 70]
BLUE = [70, 70, 255]
YELLOW = [255, 255, 70]
GREY = [125, 125, 125]


def generate_scene():

    # Light source: (only 1 for now)
    light = {'origin': [0.0, -3.0, 2.0]}

    # Spheres:
    # sphere1 = {'origin': [8., 1.8, 1.], 'radius': 0.8, 'color': RED}
    # sphere2 = {'origin': [5., -1., 1.], 'radius': 0.8, 'color': GREEN}
    # sphere3 = {'origin': [6., -1., -1.], 'radius': 0.8, 'color': BLUE}
    # sphere4 = {'origin': [8., 1., -1.], 'radius': 0.8, 'color': YELLOW}
    # sphere5 = {'origin': [8.5, 2.5, -1.], 'radius': 0.8, 'color': BLUE}

    sphere1 = {'origin': [6.0, 0.0, 0.5], 'radius': 0.99, 'color': RED}
    sphere2 = {'origin': [-2., 0., 0.5], 'radius': 0.99, 'color': GREEN}
    sphere3 = {'origin': [-3., -1.5, 0.5], 'radius': 0.99, 'color': BLUE}

    # Polygons:
    plane1 = {'origin': [5, 0, -0.5], 'normal': [0, 0, 1], 'color': GREY}

    # sphere_list = [sphere1, sphere2, sphere3, sphere4, sphere5]
    sphere_list = [sphere1, sphere2, sphere3]
    light_list = [light]
    plane_list = [plane1]

    # Build the sphere data array
    spheres = np.zeros((7, len(sphere_list)), dtype=np.float32)
    for i, s in enumerate(sphere_list):
        spheres[0:3, i]    = np.array(s['origin'])
        spheres[3, i]      = s['radius']
        spheres[4:7, i]    = np.array(s['color'])

    # Build the sphere data array
    lights = np.zeros((3, len(light_list)), dtype=np.float32)
    for i, lig in enumerate(light_list):
        lights[0:3, i]    = np.array(lig['origin'])

    # Build the polygon data array
    planes = np.zeros((9, len(plane_list)), dtype=np.float32)
    for i, p in enumerate(plane_list):
        planes[0:3, i]    = np.array(p['origin'])
        planes[3:6, i]    = np.array(p['normal']) / np.linalg.norm(np.array(p['normal']))
        planes[6:, i]    = np.array(p['color'])

    return spheres, lights, planes


def generate_rays(width, height, field_of_view=45, camera_position=None):
    if camera_position is None:
        camera_position = [0, 0, 0]                                                 # Default at origin
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])                                 # Rotation matrix
    AR = width/height
    yy, zz = np.mgrid[AR:-AR:complex(0, width), 1:-1:complex(0, height)]            # Create pixel grid
    xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(field_of_view) / 2))      # Distance of grid from origin
    pixel_locations = np.array([xx, yy, zz])

    return camera_position, R, pixel_locations


def iter_pixel_array(A):
    for x in range(A.shape[1]):
        for y in range(A.shape[2]):
            yield x, y, A[:, x, y]


def main(do_render_timing_test=False):
    # Resolution settings:
    w = 1000
    h = 1000

    # Generate scene:
    spheres_host, light_host, planes_host = generate_scene()
    spheres = cuda.to_device(spheres_host)
    light = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)

    # Generate rays:
    camera_origin_host, camera_rotation_host, pixel_locations_host = generate_rays(w, h)

    # Empty rays array to be filled in by the ray-direction kernel
    rays_host = np.zeros((6, w, h), dtype=np.float32)

    # Send data needed to get the ray direction kernel running to the device:
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
    print('Generating ray directions')
    start = time.time()
    ray_dir_kernel[blockspergrid, threadsperblock](pixel_locations, rays, origin, camera_rotation)
    end = time.time()
    print(f'Compile+generate time: {1000*(end-start)} ms')

    # Compile + render it:
    print('Compiling and running render')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, True)
    end = time.time()
    print(f'Compile + render time: {1000*(end-start)} ms')

    # Render it:
    if do_render_timing_test:
        N = 1000
        print(f'Rendering {N} times...')
        start = time.time()
        for i in range(N):
            render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes)
        end = time.time()
        print(f'Render time: {1000*(end-start)} ms')

    # Get the pixel array from GPU memory.
    result = A.copy_to_host()

    # Save the image to a .png file:
    name = 'output.png'
    print(f'Saving image to {name}')
    im = Image.new("RGB", (result.shape[1], result.shape[2]), (255, 255, 255))
    for x, y, color in iter_pixel_array(result):
        im.putpixel((x, y), tuple(color))
    im.save(name)


if __name__ == '__main__':
    main(do_render_timing_test=False)
