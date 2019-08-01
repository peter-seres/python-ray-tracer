import numpy as np


class SceneObject(object):
    def __init__(self, origin):
        self.origin = origin

    def intersect_ray(self, ray_origin, ray_direction):
        raise NotImplementedError


class Sphere(SceneObject):
    def __init__(self, origin, radius, color, k_lambert=0.5, k_specular=0.5):
        super().__init__(origin)

        self.radius = radius
        self.color = color

        self.k_lambert = 0.7
        self.k_specular = 0.5

    def intersect_ray_old(self, ray_origin, ray_dir):
        """ Returns the distance to the sphere. Returns None if does not intersect. """

        sphere_to_ray = ray_origin - self.origin

        # ignore rays too close:
        # eps = 0.01
        # if abs(np.linalg.norm(sphere_to_ray) - self.radius) < eps:
        #     return None

        a = np.dot(ray_dir, ray_dir)
        b = 2 * np.dot(ray_dir, sphere_to_ray)
        c = np.dot(sphere_to_ray, sphere_to_ray) - self.radius * self.radius

        discriminant = b * b - 4 * a * c

        if discriminant >= 0:
            dist = (-b - np.sqrt(discriminant)) / 2
            if dist > 0:
                return dist

    def intersect_ray(self, ray_origin, ray_direction):
        ray_direction /= np.linalg.norm(ray_direction)
        L = self.origin - ray_origin

        t_ca = np.dot(L, ray_direction)

        if t_ca < 0:
            return None

        d = np.sqrt(np.dot(L, L) - t_ca**2)

        if d < 0 or d > self.radius:
            return None

        t_hc = np.sqrt(self.radius ** 2 - d ** 2)

        t_0 = t_ca - t_hc
        t_1 = t_ca + t_hc

        if t_0 < 0:
            t_0 = t_1
            if t_0 < 0:
                return None

        return t_0

    def normal(self, P):
        N = P - self.origin
        return N / np.linalg.norm(N)

