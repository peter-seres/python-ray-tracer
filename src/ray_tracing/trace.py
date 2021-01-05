from numba import cuda
from .intersections import intersect_ray_sphere, intersect_ray_plane
from .common import get_sphere_color, get_plane_color, get_sphere_normal, get_plane_normal, get_vector_to_light, \
    get_reflection, dot, linear_comb


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

    """ Trace the ray and return the (R, G, B) values, the point of contact P and reflection direction R. """

    # Start with black
    RGB = (0.0, 0.0, 0.0)

    # >>> 1) Check whether the ray hits any of the spheres or planes:
    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    # If no intersection: return
    if obj_type == 404:
        return RGB, (404., 404., 404.), (404, 404., 404.)

    # Get point of intersection P:
    P = linear_comb(ray_origin, ray_dir, 1.0, intersect_dist)

    # Get the color of the object and the surface normal based on what type of object the ray hit:
    if obj_type == 0:           # (if it's a sphere)

        RGB_obj = get_sphere_color(obj_index, spheres)
        N = get_sphere_normal(P, obj_index, spheres)

    elif obj_type == 1:         # (if it's a plane)

        RGB_obj = get_plane_color(obj_index, planes)
        N = get_plane_normal(obj_index, planes)

    else:                       # (if ray does not intersect)
        return (0., 0., 0.), (404., 404., 404.), (404, 404., 404.)

    # Add ambient light
    RGB = linear_comb(RGB, RGB_obj, 1.0, ambient_int)

    # >>> 2) SHADOWS AND LAMBERT SHADING:

    # Shift point P along the normal vector to avoid shadow acne:
    BIAS = 0.0002
    P = linear_comb(P, N, 1.0, BIAS)

    # Do the lambert shading for each light source:
    for light_index in range(lights.shape[1]):

        # Get unit vector L from intersection point P to the light:
        L = get_vector_to_light(P, lights, light_index)

        # If there is a line of sight to the light source, do the lambert shading:
        _, _, obj_type = get_intersection(P, L, spheres, planes)

        # If there is an object in the way -> skip Lambert shading
        if obj_type != 404:
            continue

        # Lambert intensity depends on shader setting and surface normal
        lambert_intensity = lambert_int * dot(L, N)

        if lambert_intensity > 0:
            RGB = linear_comb(RGB, RGB_obj, 1.0, lambert_intensity)

    # >>> 3) REFLECTIONS:

    # Reflection direction:
    R = get_reflection(ray_dir, N)

    # Shift point P along reflection vector to avoid mirror acne:
    P = linear_comb(P, R, 1.0, BIAS)

    return RGB, P, R


@cuda.jit(device=True)
def sample(ray_origin: tuple, ray_dir: tuple, spheres, lights, planes, ambient_int, lambert_int,
           reflection_int, refl_depth) -> (tuple, tuple, tuple):

    # Run the tracing for this pixel to get R, G, B values
    RGB, POINT, REFLECTION_DIR = trace(ray_origin, ray_dir, spheres, lights, planes, ambient_int, lambert_int)

    # Run the reflection "depth" amount of times:
    for i in range(refl_depth):
        if (POINT[0] == 404. and POINT[1] == 404. and POINT[2] == 404.) or \
                (REFLECTION_DIR[0] == 404. and REFLECTION_DIR[1] == 404. and REFLECTION_DIR[2] == 404.):
            continue

        RGB_refl, POINT, REFLECTION_DIR = trace(POINT, REFLECTION_DIR, spheres, lights, planes, ambient_int, lambert_int)

        # Reflections power loss:
        RGB = linear_comb(RGB, RGB_refl, 1.0, reflection_int ** (i + 1))

    return RGB
