import time
import numpy as np
from numba import cuda
from ray_tracing import render_kernel, ray_dir_kernel
from PIL import Image, ImageOps
from scene.colors import *
from scene.rotation import euler_rotation
from scene.scene import Sphere, Light, Plane, Scene


def new_scene() -> (np.ndarray, np.ndarray, np.ndarray):

    # Light sources:
    lights = [Light([2.5, -2.0, 3.0])]

    # Sphere objects:
    sphere_1 = Sphere([2.2, 0.3, 1.0], 1.0, RED)
    sphere_2 = Sphere([0.6, 0.7, 0.4], 0.4, BLUE)
    sphere_3 = Sphere([0.6, -0.8, 0.5], 0.5, YELLOW)
    sphere_4 = Sphere([-1.2, 0.2, 0.5], 0.5, MAGENTA)
    sphere_5 = Sphere([-1.7, -0.5, 0.3], 0.3, GREEN)
    sphere_6 = Sphere([-2.0, 1.31, 1.3], 1.3, RED)
    spheres = [sphere_1, sphere_2, sphere_3, sphere_4, sphere_5, sphere_6]

    # Plane objects:
    planes = [Plane([5, 0, 0], [0, 0, 1], GREY)]

    # Generate the scene data:
    scene = Scene(lights, spheres, planes)

    return scene.generate_scene()


def generate_rays(width: int, height: int, camera_rotation: list = None, field_of_view: int = 45, camera_position: list = None) -> (list, np.ndarray, np.ndarray):
    """ Generates the camera position, the camera attitude rotation matrix and the pixel locations on the camera plane."""

    if camera_position is None:
        camera_position = [0, 0, 0]

    if camera_rotation is None:
        camera_rotation = [0, 0, 0]

    R = euler_rotation(camera_rotation[0], camera_rotation[1], camera_rotation[2])

    AR = width/height
    yy, zz = np.mgrid[AR:-AR:complex(0, width), 1:-1:complex(0, height)]            # Create pixel grid
    xx = np.ones(shape=yy.shape) * (1 / np.tan(np.radians(field_of_view) / 2))      # Distance of grid from origin
    pixel_locations = np.array([xx, yy, zz])

    return camera_position, R, pixel_locations


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
    # sphere_list, light_list, plane_list = custom_scene()

    # Generate the numpy arrays:
    # spheres_host, light_host, planes_host = generate_scene(sphere_list, light_list, plane_list)
q
    spheres_host, light_host, planes_host = new_scene()

    # print(type(spheres_host), type(light_host), type(planes_host))
    # print(spheres_host.shape, light_host.shape, planes_host.shape)

    # Send the numpy arrays to GPU memory:
    spheres = cuda.to_device(spheres_host)
    light = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)

    # 3) Set up camera and rays
    camera_rotation = [0, -20, 0]
    camera_position = [-2, 0, 2.0]
    camera_origin_host, camera_rotation_host, pixel_locations_host = \
        generate_rays(w, h, camera_rotation=camera_rotation, camera_position=camera_position)
    rays_host = np.zeros((6, w, h), dtype=np.float32)

    # 4) Memory Allocation on the GPU:

    origin = cuda.to_device(camera_origin_host)
    camera_rotation = cuda.to_device(camera_rotation_host)
    pixel_locations = cuda.to_device(pixel_locations_host)
    rays = cuda.to_device(rays_host)

    # Empty pixel array:
    A = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))

    # 5) Setup the cuda kernel grid:
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Calculate ray directions:
    print('Generating ray directions:')
    start = time.time()
    ray_dir_kernel[blockspergrid, threadsperblock](pixel_locations, rays, origin, camera_rotation)
    end = time.time()
    print(f'Compile + generate time: {1000*(end-start)} ms')

    # JIT Compile + render it:
    print('Compiling...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, ambient_int, lambert_int, reflection_int, 1)
    end = time.time()
    print(f'Compile + run time: {1000*(end-start)} ms')

    # Render it: (run it once more to measure render only time)
    if do_render_timing_test:

        print(f'Rendering...')
        start = time.time()
        render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, 0.04, lambert_int,
                                                      0.9, 19)
        end = time.time()
        print(f'Render time: {1000*(end-start)} ms')

    # Get the pixel array from GPU memory.
    result = A.copy_to_host()
    image = get_render(result)
    image.save('../output/test_180yaw.png')

    return 0


def get_render(x: np.ndarray) -> Image:
    """ Takes numpy array x and converts it to RGB image. """

    y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

    # Rearrange the coordinates:
    for i in range(3):
        y[:, :, i] = x[i, :, :]

    im = Image.fromarray(y.astype(np.uint8), mode='RGB')
    im = ImageOps.mirror(im.rotate(270))

    return im


if __name__ == '__main__':
    main(do_render_timing_test=False)
