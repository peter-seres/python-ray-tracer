from .scene import Sphere, Light, Plane, Triangle, Scene
from .colors import *


def default_scene() -> Scene:
    lights = [Light([2.5, -2.0, 3.0]),
              Light([2.5, 2.0, 3.0]),
              Light([5.0, 0.1, 6.0])]

    spheres = [Sphere([2.2, 0.3, 1.0], 1.0, RED),
               Sphere([0.6, 0.7, 0.4], 0.4, BLUE),
               Sphere([0.6, -0.8, 0.5], 0.5, YELLOW),
               Sphere([-1.2, 0.2, 0.5], 0.5, MAGENTA),
               Sphere([-1.7, -0.5, 0.3], 0.3, GREEN),
               Sphere([-2.0, 1.31, 1.3], 1.3, RED)]

    planes = [Plane([5, 0, 0], [0, 0, 1], GREY)]

    return Scene(lights, spheres, planes)


def triangle_test_scene() -> Scene:
    lights = [Light([2.5, -2.0, 3.0]),
              Light([2.5, 2.0, 3.0])]

    spheres = [Sphere([2.2, 0.3, 1.0], 1.0, RED),
               Sphere([0.6, 0.7, 0.4], 0.4, BLUE)]

    planes = [Plane([5, 0, 0], [0, 0, 1], GREY)]

    triangles = [Triangle([1.0, -1.0, 0.2], [4.0, -1.0, 0.5], [1.5, 0.5, 0.2], GREEN)]

    return Scene(lights, spheres, planes, triangles)
