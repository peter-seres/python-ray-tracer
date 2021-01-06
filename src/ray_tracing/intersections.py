from numba import cuda
from math import sqrt
from .common import normalize, dot, vector_difference, cross, linear_comb


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


@cuda.jit(device=True)
def intersect_ray_triangle(ray_origin, ray_dir, vertices) -> float:

    P = ray_origin
    D = ray_dir

    # Triangle is defined using 3 vertex positions:
    (V0, V1, V2) = vertices

    # Triangle normal unit vector:
    a = vector_difference(V0, V1)
    b = vector_difference(V0, V2)
    n = cross(a, b)
    N = normalize(n)

    # Dot product between N and any point in the plane:
    ND = dot(N, vector_difference(P, V1))

    # Dot product of ray direction and normal vector
    denom = dot(D, N)

    # Threshold for parallel planes:
    EPS = 0.001
    if abs(denom) < EPS:
        return -999.9

    # Distance to the ray-plane intersection:
    dist = (ND - dot(P, N)) / denom

    # If it's negative the intersection is behind us:
    if dist <= 0:
        return -999.9

    point = linear_comb(P, D, 1.0, dist)

    # Find if it's within the triangle:
    edge_0 = vector_difference(V0, V1)
    edge_1 = vector_difference(V1, V2)
    edge_2 = vector_difference(V2, V0)

    C0 = vector_difference(V0, point)
    C1 = vector_difference(V1, point)
    C2 = vector_difference(V2, point)

    inside_0 = dot(N, cross(edge_0, C0)) > 0
    inside_1 = dot(N, cross(edge_1, C1)) > 0
    inside_2 = dot(N, cross(edge_2, C2)) > 0

    if (inside_0 and inside_1 and inside_2):
        return dist
    else:
        return -999.0
