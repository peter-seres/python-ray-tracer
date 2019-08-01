import numpy as np


class SceneObject(object):
    def __init__(self, origin, color, k_lambert, k_specular):
        self.origin = origin
        self.color = color
        self.k_lambert = k_lambert
        self.k_specular = k_specular

    def intersect_ray(self, ray_origin, ray_dir):
        """ Should return the distance to the sphere. Returns None if does not intersect. """

        raise NotImplementedError

    def get_normal(self, P):
        """ Should return the normal vector at point P. """

        raise NotImplementedError


class Plane(SceneObject):
    def __init__(self, origin, color=None, normal=None, k_lambert=0.5, k_specular=0.5):
        if color is None:
            color = np.array([125, 125, 125])       # Default grey

        if normal is None:
            normal = np.array([0, 0, 1])            # Default horizontal

        super().__init__(origin, color, k_lambert, k_specular)
        self.normal = normal

    def intersect_ray(self, ray_origin, ray_dir, EPS=0.001):
        """ Returns the distance to the plane. Returns None if does not intersect. """

        p_0 = self.origin                               # plane origin
        N = self.normal / np.linalg.norm(self.normal)   # plane normal vector
        L_0 = ray_origin                                # ray origin
        L = ray_dir / np.linalg.norm(ray_dir)           # ray direction vector

        denom = np.dot(L, N)

        if abs(denom) < EPS:
            return None

        polo = (p_0 - L_0)
        dist = np.dot(polo, N) / denom

        if dist > 0:
            return dist
        else:
            return None

    def get_normal(self, P):
        return self.normal / np.linalg.norm(self.normal)


class Sphere(SceneObject):
    def __init__(self, origin, radius, color, k_lambert=0.5, k_specular=0.5):
        super().__init__(origin, color, k_lambert, k_specular)

        self.radius = radius

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

    def get_normal(self, P):
        N = P - self.origin
        return N / np.linalg.norm(N)


class Light:
    def __init__(self, origin):
        self.origin = origin
