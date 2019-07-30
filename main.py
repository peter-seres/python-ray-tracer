from PIL import Image
import numpy as np
import math


class Sphere:
    def __init__(self, position, radius, color):
        self.origin = position
        self.r = radius
        self.color = color

    def intersect_ray(self, ray):
        a = np.dot(ray.dir, ray.dir)
        b = 2 * np.dot(ray.dir, ray.origin - self.origin)
        c = np.dot(ray.origin - self.origin, ray.origin - self.origin) - self.r * self.r

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return np.array([0, 0, 0])
        else:
            return self.color


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.dir = direction

def enum_pixel_array(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            yield i, j


class Camera:
    def __init__(self, position=np.array([0, 0, 0]), direction=np.array([1, 0, 0]), field_of_view=45, width=500, height=600):

        self.origin = position
        self.dir = direction
        self.fov = field_of_view

        assert width % 2 == 0
        assert height % 2 == 0

        for i in range(-int(width/2), int(width/2)):
            for j in range(-int(height / 2), int(height / 2)):
                x = i * math.radians(field_of_view) / width
                y = j * math.radians(field_of_view) / height


def main():
    PIXEL_WIDTH = 0.5
    PIXEL_HEIGHT = 0.5

    # The scene contains a camera and a few spheres:
    camera = Camera(position=np.array([0, 0, 0]), direction=np.array([1, 0, 0]))
    sphere1 = Sphere(position=np.array([50, 2, 0]), radius=4, color=np.array([200, 100, 123]))
    # sphere2 = Sphere(position=np.array([50, -10, 0]), radius=5, color=np.array([200, 45, 45]))

    w = 500     # image width (pixels)
    h = 480     # image height (pixels)
    ch = 3      # number of channels (RGB)
    pixel_array = np.zeros(shape=(w, h, ch), dtype=np.uint8)

    for i, j in enum_pixel_array(pixel_array):
        ray = Ray(origin=np.array([0, i*PIXEL_WIDTH, j*PIXEL_HEIGHT]), direction=camera.dir)
        pixel_array[i, j] = sphere1.intersect_ray(ray)

    image = Image.fromarray(pixel_array)
    image.save('result.png')


if __name__ == '__main__':
    main()
