import time

from PIL import Image
import numpy as np
from camera import Camera


class Sphere:
    def __init__(self, position, radius, color):
        self.origin = np.array(position)
        self.r = radius
        self.color = np.array(color)

    def intersect_ray(self, ray):

        a = np.dot(ray.dir, ray.dir)
        b = 2 * np.dot(ray.dir, ray.origin - self.origin)
        c = np.dot(ray.origin - self.origin, ray.origin - self.origin) - self.r * self.r

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return np.array([0, 0, 0])
        else:
            return self.color


def main():
    # Image resolution
    w = 500     # image width (pixels)
    h = 480     # image height (pixels)

    # The scene contains a camera and a sphere:
    camera = Camera(w, h, origin=[0, 0, 0], direction=[1, 0, 1], field_of_view=45)
    sphere1 = Sphere(position=[50, 0, 0], radius=5, color=[255, 255, 123])

    # Generate pixels:
    start = time.time()
    pixel_array = camera.cast_rays(sphere1)
    end = time.time()
    print(f'Ray tracing took {end-start} seconds with {500*480} rays and 1 object.')

    # Save the data to a png file:
    image = Image.fromarray(pixel_array)
    image.save('look_up.png')


if __name__ == '__main__':
    main()
