import numpy as np
from numba import cuda
from ray_tracing import render
from scene import Scene, Camera
from viewer import convert_array_to_image


def main():
    # 1) Render and shader settings:
    w, h = 1220, 1220
    amb, lamb, refl, refl_depth = 0.1, 0.6, 0.5, 5

    # 2) Generate scene:
    scene = Scene.default_scene()
    spheres_host, light_host, planes_host = scene.generate_scene()

    # Send the data arrays to GPU memory:
    spheres = cuda.to_device(spheres_host)
    lights = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)

    # 3) Set up camera and rays
    camera = Camera(resolution=(w, h), position=[-2, 0, 2.0], euler=[0, -20, 0])

    # Send the camera data to GPU memory:
    camera_origin = cuda.to_device(camera.position)
    camera_rotation = cuda.to_device(camera.rotation)
    pixel_loc = cuda.to_device(camera.generate_pixel_locations())

    # 4) Memory Allocation for the result:
    result = cuda.to_device(np.zeros((3, w, h), dtype=np.uint8))

    # 5) Setup the cuda kernel grid:
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(result.shape[1] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(result.shape[2] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # 6) Call JIT compiled renderer:
    render[blockspergrid, threadsperblock](pixel_loc, result, camera_origin, camera_rotation,
                                           spheres, lights, planes, amb, lamb, refl, refl_depth)

    # 7) Present the result as a .png
    result = result.copy_to_host()
    image = convert_array_to_image(result)
    image.save('../output/test_sampled.png')

    return 0


if __name__ == '__main__':
    main()
