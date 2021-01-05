from numba import cuda
from math import sqrt
from .common import normalize, dot, vector_difference


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin: tuple, ray_dir: tuple, sphere_origin: tuple, sphere_radius: float) -> float:
    """ This function takes the ray and sphere data and computes whether there is an intersection. Return -999.9 if
     the ray does not hit the sphere. Returns the distance from the ray origin along the ray direction at which the
     intersection point can be found."""

    # 0) R : Vector of ray direction. Make sure it is normalized:
    R = normalize(ray_dir)

    # 1) L : Vector pointing from Sphere origin to ray origin
    L = vector_difference(sphere_origin, ray_origin)

    # 2) Second order equation terms: a, b, c:
    a = dot(R, R)
    b = 2 * dot(L, R)
    c = dot(L, L) - sphere_radius * sphere_radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0.0:
        return -999.9
    else:
        numerator = -b - sqrt(discriminant)

        if numerator > 0.0:
            return numerator / (2 * a)

        numerator = -b + sqrt(discriminant)

        if numerator > 0.0:
            return numerator / (2 * a)
        else:
            return -999.0


@cuda.jit(device=True)
def intersect_ray_plane(ray_origin: tuple, ray_dir: tuple, plane_origin: tuple, plane_normal: tuple) -> float:
    """ Returns the distance to the plane. Returns -999.9 if does not intersect. """

    # Threshold for parallel planes:
    EPS = 0.001

    # N: Plane normal vector
    N = plane_normal[0:3]

    # Dot product of ray direction and normal vector
    denom = dot(ray_dir, plane_normal)

    # Check if ray is not parallel with plane:
    if abs(denom) < EPS:
        return -999.9

    # LP: Vector from ray to plane center
    LP = vector_difference(ray_origin, plane_origin)

    nominator = dot(LP, N)

    dist = nominator / denom

    if dist > 0:
        return dist
    else:
        return -999.0
