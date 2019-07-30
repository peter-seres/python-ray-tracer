import numpy as np


class Sphere:
    def __init__(self, position, radius, color):
        self.origin = np.array(position)
        self.r = radius
        self.color = np.array(color)
        self.distance_from_origin = np.linalg.norm(self.origin)

    def intersect_ray(self, ray):

        a = np.dot(ray.dir, ray.dir)
        b = 2 * np.dot(ray.dir, ray.origin - self.origin)
        c = np.dot(ray.origin - self.origin, ray.origin - self.origin) - self.r * self.r

        discriminant = b * b - 4 * a * c

        if discriminant >= 0:

            distSqrt = np.sqrt(discriminant)
            q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0

            t0 = q / a
            t1 = c / q
            t0, t1 = min(t0, t1), max(t0, t1)

            if t1 >= 0:
                return t1 if t0 < 0 else t0
            # P = None

        else:
            # dist = np.linalg.norm(ray.origin - P)
            return np.inf

    def dist(self):
        return self.distance_from_origin

    def distance_from(self, coord):
        return np.linalg.norm(self.origin - coord)

    def __str__(self):
        return f'Sphere at location: {self.origin}'
