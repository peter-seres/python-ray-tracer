from numba import cuda
from math import sqrt


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin: tuple, ray_dir: tuple, sphere_origin: tuple, sphere_radius: float) -> float:
    """ This function takes the ray and sphere data and computes whether there is an intersection. Return -999.9 if
     the ray does not hit the sphere. Returns the distance from the ray origin along the ray direction at which the
     intersection point can be found."""

    # 0) R : Vector of ray direction. Make sure it is normalized:
    dir_norm = sqrt(ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2])
    R0 = ray_dir[0] / dir_norm
    R1 = ray_dir[1] / dir_norm
    R2 = ray_dir[2] / dir_norm

    # 1) L : Vector pointing from Sphere origin to ray origin
    L0 = ray_origin[0] - sphere_origin[0]
    L1 = ray_origin[1] - sphere_origin[1]
    L2 = ray_origin[2] - sphere_origin[2]

    # 2) Second order equation terms: a, b, c:
    a = R0 * R0 + R1 * R1 + R2 * R2
    b = 2 * (L0 * R0 + L1 * R1 + L2 * R2)
    c = L0 * L0 + L1 * L1 + L2 * L2 - sphere_radius * sphere_radius

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
    N0 = plane_normal[0]
    N1 = plane_normal[1]
    N2 = plane_normal[2]

    # Dot product of ray direction and normal vector
    denom = ray_dir[0] * N0 + ray_dir[1] * N1 + ray_dir[2] * N2

    # Check if ray is not parallel with plane:
    if abs(denom) < EPS:
        return -999.9

    # LP: Vector from ray to plane center
    LP_0 = plane_origin[0] - ray_origin[0]
    LP_1 = plane_origin[1] - ray_origin[1]
    LP_2 = plane_origin[2] - ray_origin[2]

    nominator = LP_0 * N0 + LP_1 * N1 + LP_2 * N2

    dist = nominator / denom

    if dist > 0:
        return dist
    else:
        return -999.0
