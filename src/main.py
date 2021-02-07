import numpy as np
from numba import cuda
from ray_tracing import render
from scene import Camera
from viewer import convert_array_to_image
from scene.scene_setups import default_scene, triangle_test_scene


def main():
    # 1) Render and shader settings:
    w, h = 300, 300
    amb, lamb, refl, refl_depth = 0.2, 0.6, 0.3, 1
    aliasing = False

    # 2) Generate scene:
    scene = triangle_test_scene()
    spheres_host, light_host, planes_host, triangles_host = scene.generate_scene()

    # Send the data arrays to GPU memory:
    spheres = cuda.to_device(spheres_host)
    lights = cuda.to_device(light_host)
    planes = cuda.to_device(planes_host)
    triangles = cuda.to_device(triangles_host)

    # 3) Set up camera and rays
    camera = Camera(resolution=(w, h), position=[-7, 0, 5.0], euler=[0, -15, 0])

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
    print("startinng kernel")
    render[blockspergrid, threadsperblock](pixel_loc, result, camera_origin, camera_rotation,
                                           spheres, lights, planes, triangles, amb, lamb, refl, refl_depth, aliasing)

    # import time
    # st = time.time()
    # render[blockspergrid, threadsperblock](pixel_loc, result, camera_origin, camera_rotation,
    #                                        spheres, lights, planes, amb, lamb, refl, refl_depth+2, aliasing)
    # et = time.time()
    # print(f"time: {1000 * (et - st):,.1f} ms")

    # 7) Present the result as a .png
    print("render done")
    result = result.copy_to_host()
    image = convert_array_to_image(result)
    image.save('../output/test_triangle.png')

    return 0


if __name__ == '__main__':
    main()
