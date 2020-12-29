from numba import cuda
from .intersections import intersect_ray_sphere, intersect_ray_plane
from .common import get_sphere_color, get_plane_color, get_sphere_normal, get_plane_normal, get_vector_to_light, \
    get_reflection


@cuda.jit(device=True)
def get_intersection(ray_origin: tuple, ray_dir: tuple, spheres, planes) -> (float, int, int):

    """ Cast a single ray from ray_origin and check whether it hits something.

        Returns the distance along the ray where the intersection happens.
        Returns the index of the object hit.
        Returns obj_type: 0 for a sphere, 1 for a plane and 404 for nothing.
    """

    intersect_dist = 999.0
    obj_index = -999
    obj_type = 404

    # Loop through all spheres:
    for idx in range(spheres.shape[1]):
        dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])

        # If it hits the sphere and dist is closer than the closest one:
        if intersect_dist > dist > 0:
            intersect_dist = dist
            obj_index = idx
            obj_type = 0

    # Loop through all planes:
    for idx in range(planes.shape[1]):
        dist = intersect_ray_plane(ray_origin, ray_dir, planes[0:3, idx], planes[3:6, idx])

        # If it hits the plane and dist is closer than the closest one:
        if intersect_dist > dist > 0:
            intersect_dist = dist
            obj_index = idx
            obj_type = 1

    return intersect_dist, obj_index, obj_type


@cuda.jit(func_or_sig=None, device=True)
def trace(ray_origin: tuple, ray_dir: tuple, spheres, lights, planes, ambient_int: float, lambert_int: float) -> (tuple, tuple, tuple):

    """ Trace the ray and return the R, G, B values"""

    # Start with black
    R, G, B = (0.0, 0.0, 0.0)

    # Declare the color of the object
    R_obj, G_obj, B_obj = (0.0, 0.0, 0.0)

    # >>> 1) Check whether the ray hits any of the spheres or planes:
    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    # If no intersection: return black
    if obj_type == 404:
        return (0., 0., 0.), (404., 404., 404.), (404, 404., 404.)

    # Get point of intersection P:
    P0 = ray_origin[0] + ray_dir[0] * intersect_dist
    P1 = ray_origin[1] + ray_dir[1] * intersect_dist
    P2 = ray_origin[2] + ray_dir[2] * intersect_dist

    # Get the color of the object and the surface normal based on what type of object the ray hit:
    if obj_type == 0:           # (if it's a sphere)

        R_obj, G_obj, B_obj = get_sphere_color(obj_index, spheres)
        N0, N1, N2 = get_sphere_normal((P0, P1, P2), obj_index, spheres)

    elif obj_type == 1:         # (if it's a plane)

        R_obj, G_obj, B_obj = get_plane_color(obj_index, planes)
        N0, N1, N2 = get_plane_normal(obj_index, planes)

    else:                       # (if ray does not intersect)
        return (0., 0., 0.), (404., 404., 404.), (404, 404., 404.)

    # Add ambient light
    R = R + R_obj * ambient_int
    G = G + G_obj * ambient_int
    B = B + B_obj * ambient_int

    # >>> 2) SHADOWS AND LAMBERT SHADING:

    # Shift point P along the normal vector to avoid shadow acne:
    BIAS = 0.0002
    P0 = P0 + BIAS * N0
    P1 = P1 + BIAS * N1
    P2 = P2 + BIAS * N2

    for light_index in range(lights.shape[1]):

        # Get unit vector L from intersection point P to the light:
        L0, L1, L2 = get_vector_to_light((P0, P1, P2), lights, light_index)

        # If there is a line of sight to the light source, do the lambert shading:
        _, _, shadow_type = get_intersection((P0, P1, P2), (L0, L1, L2), spheres, planes)

        if shadow_type == 404:

            lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2

            if lambert_intensity > 0:
                R = R + R_obj * lambert_intensity * lambert_int
                G = G + G_obj * lambert_intensity * lambert_int
                B = B + B_obj * lambert_intensity * lambert_int

    # >>> 3) REFLECTIONS:

    # Reflection direction:
    R0, R1, R2 = get_reflection(ray_dir, (N0, N1, N2))

    # Shift point P along reflection vector to avoid mirror acne:
    P0 = P0 + BIAS * R0
    P1 = P1 + BIAS * R1
    P2 = P2 + BIAS * R2

    color = (R, G, B)
    POINT = (P0, P1, P2)
    REFLECTION_DIR = (R0, R1, R2)

    return color, POINT, REFLECTION_DIR
