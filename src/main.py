import time
import numpy as np
from numba import cuda
from src.cuda_ray_tracing import render_kernel, ray_dir_kernel
from PIL import Image, ImageOps


RED = [255, 70, 70]
GREEN = [70, 255, 70]
BLUE = [70, 70, 255]
YELLOW = [255, 255, 70]
GREY = [125, 125, 125]
MAGENTA = [139, 0, 139]


def custom_scene() -> (list, list, list):
    # Light sources:
    light1 = {'origin': [2.5, -2.0, 3.0]}
    light2 = {'origin': [2.5,  2.0, 3.0]}
    light_list = [light1, light2]

    # Horizontal plane:
    plane1 = {'origin': [5, 0, 0], 'normal': [0, 0, 1], 'color': GREY}
    plane_list = [plane1]

    # sphere list

    sphere1 = sphere = {'origin': [2.2, 0.3, 1.0], 'radius': 1.0, 'color': RED}
    sphere2 = sphere = {'origin': [0.6, 0.7, 0.4], 'radius': 0.4, 'color': BLUE}
    sphere3 = sphere = {'origin': [0.6, -0.8, 0.5], 'radius': 0.5, 'color': YELLOW}
    sphere4 = sphere = {'origin': [-1.2, 0.2, 0.5], 'radius': 0.5, 'color': MAGENTA}
    sphere5 = sphere = {'origin': [-1.7, -0.5, 0.3], 'radius': 0.3, 'color': GREEN}
    sphere6 = sphere = {'origin': [-2.0, 1.31, 1.3], 'radius': 1.3, 'color': RED}

    sphere_list = [sphere1, sphere2, sphere3, sphere4, sphere5, sphere6]

    return sphere_list, light_list, plane_list


def generate_scene(sphere_list: list, light_list: list, plane_list: list) -> (np.ndarray, np.ndarray, np.ndarray):
    """ Takes scene object dictionaries and converts them to a numpy array format. """

    # Build the sphere data array
    spheres = np.zeros((7, len(sphere_list)), dtype=np.float32)
    for i, s in enumerate(sphere_list):
        spheres[0:3, i]    = np.array(s['origin'])
        spheres[3, i]      = s['radius']
        spheres[4:7, i]    = np.array(s['color'])

    # Build the light data array
    lights = np.zeros((3, len(light_list)), dtype=np.float32)
    for i, lig in enumerate(light_list):
        lights[0:3, i]    = np.array(lig['origin'])

    # Build the plane data array
    planes = np.zeros((9, len(plane_list)), dtype=np.float32)
    for i, p in enumerate(plane_list):
        planes[0:3, i]    = np.array(p['origin'])
        planes[3:6, i]    = np.array(p['normal']) / np.linalg.norm(np.array(p['normal']))
        planes[6:, i]    = np.array(p['color'])

    return spheres, lights, planes


def rotation_z(psi: float) -> np.ndarray:
    """ Return the Rotation matrix around the Z-axis for psi in degrees. """

    psi = np.deg2rad(psi)
    R_rot = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
    return R_rot


def rotation_y(theta: float) -> np.ndarray:
    """ Return the Rotation matrix around the Y-axis for theta in degrees. """

    theta = np.deg2rad(theta)
    R_rot = np.array([[np.cos(theta), 0, -np.sin(theta)],
                      [0,             1,               0],
                      [np.sin(theta), 0, np.cos(theta)]])
    return R_rot


def rotation_x(phi: float) -> np.ndarray:
    """ Return the Rotation matrix around the X-axis for phi in degrees. """

    phi = np.deg2rad(phi)
    R_rot = np.array([[1, 0, 0],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi), np.cos(phi)]])
    return R_rot


def generate_rays(width: int, height: int, camera_rotation: list = None, field_of_view: int = 45, camera_position: list = None) -> (list, np.ndarray, np.ndarray):
    """ Generates the camera position, the camera attitude rotation matrix and the pixel locations on the camera plane."""

    if camera_position is None:
        camera_position = [0, 0, 0]                                                 # Default at origin

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])                                 # Rotation matrix

    if camera_rotation is None:
        camera_rotation = [0, 0, 0]

    Ry = rotation_y(camera_rotation[1])
    Rz = rotation_z(camera_rotation[2])

    R = np.matmul(Rz, Ry)

    AR = width/height
    yy, zz = np.mgrid[AR:-AR:complex(0, width), 1:-1:complex(0, height)]            # Create pixel grid
    xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(field_of_view) / 2))      # Distance of grid from origin
    pixel_locations = np.array([xx, yy, zz])

    return camera_position, R, pixel_locations


def iter_pixel_array(A):
    """ Generator to iterate through each pixel. """

    for x in range(A.shape[1]):
        for y in range(A.shape[2]):
            yield x, y, A[:, x, y]


def main(do_render_timing_test=False):

    """ Entry point:

    1) Set up render parameters: resolution, shader parameters
    2) Set up scene: spheres, planes and lights
    3) Set up camera rays.

    """

    # 1) Render and shader settings:
    w = 1000
    h = 1000
    ambient_int, lambert_int, reflection_int = 0.1, 0.6, 0.5

    # 2) Generate scene:
    sphere_list, light_list, plane_list = custom_scene()

    # Generate the numpy arrays:
    spheres_host, light_host, planes_host = generate_scene(sphere_list, light_list, plane_list)

    # Send the numpy arrays to GPU memory:
    spheres = cuda.to_device(spheres_host)
    light = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)

    # 3) Set up camera and rays
    camera_rotation = [0, -20, 0]
    camera_position = [-2, 0, 2.0]
    camera_origin_host, camera_rotation_host, pixel_locations_host = \
        generate_rays(w, h, camera_rotation=camera_rotation, camera_position=camera_position)

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
    print('Generating ray directions:')
    start = time.time()
    ray_dir_kernel[blockspergrid, threadsperblock](pixel_locations, rays, origin, camera_rotation)
    end = time.time()
    print(f'Compile + generate time: {1000*(end-start)} ms')

    # Compile + render it:
    print('Compiling...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, ambient_int, lambert_int, reflection_int, 1)
    end = time.time()
    print(f'Compile + run time: {1000*(end-start)} ms')

    # Render it: (run it once more to measure render only time
    if do_render_timing_test:

        print(f'Rendering...')
        start = time.time()
        render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, ambient_int, lambert_int,
                                                      reflection_int, 10)
        end = time.time()
        print(f'Render time: {1000*(end-start)} ms')

    # Get the pixel array from GPU memory.
    result = A.copy_to_host()
    image = get_render(result)
    image.save('../output/render.png')

    return 0


# noinspection DuplicatedCode
def get_render(x: np.ndarray) -> Image:

    """ Takes numpy array x and converts it to RGB image. """
    y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

    # Rearrange the coordinates:
    for i in range(3):
        y[:, :, i] = x[i, :, :]

    image = Image.fromarray(y.astype(np.uint8), mode='RGB')
    image = image.rotate(270)
    image = ImageOps.mirror(image)

    return image


if __name__ == '__main__':
    main(do_render_timing_test=False)
