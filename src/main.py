import time
import numpy as np
from numba import cuda
from ray_tracing import render_kernel, ray_dir_kernel
from PIL import Image, ImageOps
from scene import Scene, Camera


def get_render(x: np.ndarray) -> Image:
    """ Takes numpy array x and converts it to RGB image. """

    y = np.zeros(shape=(x.shape[1], x.shape[2], 3))

    # Rearrange the coordinates:
    for i in range(3):
        y[:, :, i] = x[i, :, :]

    im = Image.fromarray(y.astype(np.uint8), mode='RGB')
    im = ImageOps.mirror(im.rotate(270))

    return im


def main():
    # 1) Render and shader settings:
    w, h = 1000, 1000
    ambient_int, lambert_int, reflection_int, reflection_depth = 0.1, 0.6, 0.5, 5

    # 2) Generate scene:
    scene = Scene.default_scene()
    spheres_host, light_host, planes_host = scene.generate_scene()

    # Send the data arrays to GPU memory:
    spheres = cuda.to_device(spheres_host)
    light = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)

    # 3) Set up camera and rays
    camera = Camera(resolution=(w, h), position=[-2, 0, 2.0], euler=[0, -20, 0])

    # Send the camera data to GPU memory:
    camera_origin = cuda.to_device(camera.position)
    camera_rotation = cuda.to_device(camera.rotation)
    pixel_locations = cuda.to_device(camera.generate_pixel_locations())

    # 4) Memory Allocation on the GPU:
    rays = cuda.to_device(np.zeros((6, w, h), dtype=np.float32))
    A = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))

    # 5) Setup the cuda kernel grid:
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(A.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Calculate ray directions:
    print('Running ray generation kernel')
    start = time.time()
    ray_dir_kernel[blockspergrid, threadsperblock](pixel_locations, rays, camera_origin, camera_rotation)
    end = time.time()
    print(f'Compile + generate time: {1000 * (end - start):,.1f} ms')

    # JIT Compile + render it:
    print('Running render kernel...')
    start = time.time()
    render_kernel[blockspergrid, threadsperblock](A, rays, spheres, light, planes, ambient_int, lambert_int,
                                                  reflection_int, reflection_depth)
    end = time.time()
    print(f'Compile + run time: {1000 * (end - start):,.1f} ms')

    # Get the pixel array from GPU memory.
    result = A.copy_to_host()
    image = get_render(result)
    image.save('../output/test_180yaw.png')

    return 0


if __name__ == '__main__':
    main()
