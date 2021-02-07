from numba import cuda
from math import sqrt


@cuda.jit(device=True)
def to_tuple3(array):
    return (array[0], array[1], array[2])


@cuda.jit(device=True)
def linear_comb(a, b, c1, c2):
    """
    returns p = c1 * a + c2 * b for 3D vectors
    """
    p0 = c1 * a[0] + c2 * b[0]
    p1 = c1 * a[1] + c2 * b[1]
    p2 = c1 * a[2] + c2 * b[2]
    return (p0, p1, p2)


@cuda.jit(device=True)
def vector_difference(fromm, to):
    f0, f1, f2 = fromm
    t0, t1, t2 = to
    return (t0 - f0, t1 - f1, t2 - f2)


@cuda.jit(device=True)
def normalize(vector):
    (X, Y, Z) = vector
    norm = sqrt(X * X + Y * Y + Z * Z)
    return (X / norm, Y / norm, Z / norm)


@cuda.jit(device=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cuda.jit(device=True)
def matmul(A, x):
    """
    Pack A's rows into a tuple of tuples
    """
    A0, A1, A2 = A
    b0 = dot(A0, x)
    b1 = dot(A1, x)
    b2 = dot(A2, x)
    return (b0, b1, b2)


@cuda.jit(device=True)
def clip_color(color):
    """
    This function takes the resulting RGB float and clips it to an 8bit integer.
    """
    return min(max(0, int(round(color))), 255)


@cuda.jit(device=True)
def clip_color_vector(color3):
    (R, G, B) = color3
    return clip_color(R), clip_color(B), clip_color(G)


@cuda.jit(device=True)
def get_sphere_color(index, spheres):
    """
    Returns the color tuple of a sphere.
    """
    (R, G, B) = spheres[4:7, index]
    return (R, G, B)


@cuda.jit(device=True)
def get_plane_color(index, planes):
    """
    Returns the color tuple of a plane.
    """
    (R, G, B) = planes[6:9, index]
    return (R, G, B)


@cuda.jit(device=True)
def get_vector_to_light(P, lights, light_index):
    """
    Returns the unit vector from point P to a light L.
    """
    L = lights[0:3, light_index]
    P_L = vector_difference(P, L)
    return normalize(P_L)


@cuda.jit(device=True)
def get_sphere_normal(P, sphere_index, spheres):
    """
    Returns the unit normal vector on the surface of a sphere at point P.
    """
    sphere_origin = spheres[0:3, sphere_index]
    N = vector_difference(sphere_origin, P)
    return normalize(N)


@cuda.jit(device=True)
def get_plane_normal(plane_index, planes):
    """
    Returns the unit normal vector on the surface of a normal at point P.
    """
    (N0, N1, N2) = planes[3:6, plane_index]
    return normalize((N0, N1, N2))


@cuda.jit(device=True)
def get_reflection(ray_dir, normal):
    """
    Returns the unit reflection direction vector.
    """
    D_dot_N = dot(ray_dir, normal)
    R = linear_comb(ray_dir, normal, 1.0, -2.0 * D_dot_N)
    return normalize(R)
