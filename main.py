import numpy as np
from cuda_ray_tracing import *
import time
import PIL.Image
import PIL.ImageOps

RED = [255, 70, 70]
GREEN = [70, 255, 70]
BLUE = [70, 70, 255]
YELLOW = [255, 255, 70]
GREY = [125, 125, 125]
MAGENTA = [139, 0, 139]


def generate_scene():

    # Light source: (only 1 for now)
    light1 = {'origin': [2.5, -2.0, 3.0]}
    light2 = {'origin': [2.5,  2.0, 3.0]}

    # Spheres:
    sphere1 = {'origin': [5., 0, 1.], 'radius': 1, 'color': RED}
    # sphere2 = {'origin': [5., -1., 1.], 'radius': 0.8, 'color': GREEN}
    # sphere3 = {'origin': [6., -1., -1.], 'radius': 0.8, 'color': BLUE}
    # sphere4 = {'origin': [8., 1., -1.], 'radius': 0.8, 'color': YELLOW}
    # sphere5 = {'origin': [8.5, 2.5, -1.], 'radius': 0.8, 'color': BLUE}

    # sphere1 = {'origin': [4.0, 0.0, 0.3], 'radius': 0.995, 'color': RED}
    # sphere2 = {'origin': [-1, 0., 0.3], 'radius': 0.995, 'color': GREEN}
    # sphere3 = {'origin': [-2., 2, 0.6], 'radius': 1.1, 'color': BLUE}
    # sphere4 = {'origin': [4.0, -2.3, 0.5], 'radius': 1.2, 'color': GREEN}
    sphere5 = {'origin': [3, 1.3, 0.4], 'radius': 0.4, 'color': MAGENTA}

    # Polygons:
    plane1 = {'origin': [5, 0, 0], 'normal': [0, 0, 1], 'color': GREY}

    # sphere_list = [sphere1, sphere2, sphere3, sphere4, sphere5]
    sphere_list = [sphere1, sphere5]
    light_list = [light1, light2]
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


def rotation_z(psi):
    """ Return the Rotation matrix around the Z-axis for psi in degrees. """
    psi = np.deg2rad(psi)
    R_rot = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
    return R_rot


def rotation_y(theta):
    """ Return the Rotation matrix around the Y-axis for theta in degrees. """
    theta = np.deg2rad(theta)
    R_rot = np.array([[np.cos(theta), 0, -np.sin(theta)],
                      [0,             1,               0],
                      [np.sin(theta), 0, np.cos(theta)]])
    return R_rot


def rotation_x(phi):
    """ Return the Rotation matrix around the X-axis for phi in degrees. """
    phi = np.deg2rad(phi)
    R_rot = np.array([[1, 0, 0],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi), np.cos(phi)]])
    return R_rot


def generate_rays(width, height, camera_rotation=None, field_of_view=45, camera_position=None):
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
    for x in range(A.shape[1]):
        for y in range(A.shape[2]):
            yield x, y, A[:, x, y]


def main(do_render_timing_test=False):
    # Resolution settings:
    w = 1000
    h = 1000

    ambient_int, lambert_int, reflection_int = 0.1, 0.6, 0.5

    # Generate scene:
    spheres_host, light_host, planes_host = generate_scene()
    spheres = cuda.to_device(spheres_host)
    light = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)

    # Generate rays:
    # camera_rotation = [0, -10, -180]
    # camera_position = [8.0, 0, 1.0]
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
    print('Generating ray directions')
    start = time.time()
    ray_dir_kernel[blockspergrid, threadsperblock](pixel_locations, rays, origin, camera_rotation)
    end = time.time()
    print(f'Compile+generate time: {1000*(end-start)} ms')

    # Compile + render it:
    print('Compiling and running render')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, ambient_int, lambert_int, reflection_int, 4)
    end = time.time()
    print(f'Compile + render time: {1000*(end-start)} ms')

    # Render it: (run it once more to measure render only time
    if do_render_timing_test:
        print(f'Rendering...')
        start = time.time()
        render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, ambient_int, lambert_int, reflection_int, 4)
        end = time.time()
        print(f'Render time: {1000*(end-start)} ms')

    # Get the pixel array from GPU memory.
    result = A.copy_to_host()

    image, x, y = get_render(result)
    image.save('get_render_output.png')

    # Save the image to a .png file:
    # name = 'output.png'
    # print(f'Saving image to {name}')
    # im = Image.new("RGB", (result.shape[1], result.shape[2]), (255, 255, 255))
    # for x, y, color in iter_pixel_array(result):
    #     im.putpixel((x, y), tuple(color))
    # im.save(name)

    return x, y


def get_render(x):

    y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

    for i in range(3):
        y[:, :, i] = x[i, :, :]

    image = PIL.Image.fromarray(y.astype(np.uint8), mode='RGB')
    image = image.rotate(270)
    image = PIL.ImageOps.mirror(image)

    return image, x, y


if __name__ == '__main__':
    x, y = main(do_render_timing_test=False)
