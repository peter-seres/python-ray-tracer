from numba import cuda
from math import sqrt


@cuda.jit(device=True)
def clip_color(color):
    """ This function takes the resulting RGB float and clips it to an 8bit integer."""

    return min(max(0, int(round(color))), 255)


@cuda.jit(device=True)
def get_sphere_color(index, spheres):
    """ Returns the color tuple of a sphere."""

    R = spheres[4, index]
    G = spheres[5, index]
    B = spheres[6, index]
    return R, G, B


@cuda.jit(device=True)
def get_plane_color(index, planes):
    """ Returns the color tuple of a plane."""

    R = planes[6, index]
    G = planes[7, index]
    B = planes[8, index]
    return R, G, B


@cuda.jit(device=True)
def get_vector_to_light(P, lights, light_index):
    """ Returns the unit vector to a light from point P"""

    L0 = lights[0, light_index] - P[0]
    L1 = lights[1, light_index] - P[1]
    L2 = lights[2, light_index] - P[2]
    norm = sqrt(L0 * L0 + L1 * L1 + L2 * L2)
    L0 = L0 / norm
    L1 = L1 / norm
    L2 = L2 / norm
    return L0, L1, L2


@cuda.jit(device=True)
def get_sphere_normal(P, sphere_index, spheres):
    """ Returns the unit normal vector on the surface of a sphere at point P. """

    N0 = P[0] - spheres[0, sphere_index]
    N1 = P[1] - spheres[1, sphere_index]
    N2 = P[2] - spheres[2, sphere_index]
    norm = sqrt(N0 * N0 + N1 * N1 + N2 * N2)
    N0 = N0 / norm
    N1 = N1 / norm
    N2 = N2 / norm
    return N0, N1, N2


@cuda.jit(device=True)
def get_plane_normal(plane_index, planes):
    """ Returns the unit normal vector on the surface of a normal at point P. """

    N0 = planes[3, plane_index]
    N1 = planes[4, plane_index]
    N2 = planes[5, plane_index]
    norm = sqrt(N0 * N0 + N1 * N1 + N2 * N2)
    N0 = N0 / norm
    N1 = N1 / norm
    N2 = N2 / norm
    return N0, N1, N2


@cuda.jit(device=True)
def get_reflection(ray_dir, normal):
    """ Returns the unit reflection direction vector."""

    D0 = ray_dir[0]
    D1 = ray_dir[1]
    D2 = ray_dir[2]

    N0 = normal[0]
    N1 = normal[1]
    N2 = normal[2]

    D_dot_N = D0 * N0 + D1 * N1 + D2 * N2

    R0 = D0 - 2 * D_dot_N * N0
    R1 = D1 - 2 * D_dot_N * N1
    R2 = D2 - 2 * D_dot_N * N2

    norm = sqrt(R0 * R0 + R1 * R1 + R2 * R2)
    R0 = R0 / norm
    R1 = R1 / norm
    R2 = R2 / norm
    return R0, R1, R2
