from numba import cuda
import math


@cuda.jit(device=True)
def clip(color):
    """ This function takes the resulting RGB float and clips it to an 8bit integer."""
    return min(max(0, int(round(color))), 255)


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin, ray_dir, sphere_origin, sphere_radius):
    """ This function takes the ray and sphere data and computes whether there is an intersection. Return -999.9 if
     the ray does not hit the sphere. Returns the distance from the ray origin along the ray direction at which the
     intersection point can be found."""

    # 0) R : Vector of ray direction. Make sure it is normalized:
    dir_norm = math.sqrt(ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2])
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
        numerator = -b - math.sqrt(discriminant)

        if numerator > 0.0:
            return numerator / (2 * a)

        numerator = -b + math.sqrt(discriminant)

        if numerator > 0.0:
            return numerator / (2 * a)
        else:
            return -999.0


@cuda.jit(device=True)
def intersect_ray_plane(ray_origin, ray_dir, plane_origin, plane_normal):
    """ Returns the distance to the plane. Returns -999.0 if does not intersect. """

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
        return -999.0

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


@cuda.jit(device=True)
def get_intersection(ray_origin, ray_dir, spheres, planes):
    """ Returns the distance along the ray where the intersection happens.
        Returns the index of the object hit.
        Returns obj_type: 0 for a sphere, 1 for a plane and 404 for nothing. """

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
def get_vector_to_light(P, lights):
    """ Returns the unit vector to a light from point P"""
    light_index = 0
    L0 = lights[0, light_index] - P[0]
    L1 = lights[1, light_index] - P[1]
    L2 = lights[2, light_index] - P[2]
    norm = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2)
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
    norm = math.sqrt(N0 * N0 + N1 * N1 + N2 * N2)
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
    norm = math.sqrt(N0 * N0 + N1 * N1 + N2 * N2)
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

    norm = math.sqrt(R0 * R0 + R1 * R1 + R2 * R2)
    R0 = R0 / norm
    R1 = R1 / norm
    R2 = R2 / norm
    return R0, R1, R2


@cuda.jit(func_or_sig=None, device=True)
def trace(ray_origin, ray_dir, spheres, lights, planes):
    """ Trace the ray and return the R, G, B values"""

    # Ambient light:
    ambient_intensity = 0.3

    # Start with black
    R, G, B = (0.0, 0.0, 0.0)

    # Declare the color of the object
    R_obj, G_obj, B_obj = (0.0, 0.0, 0.0)

    # Check whether the ray hits any of the spheres or planes:
    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    # If no intersection: return black
    if obj_type == 404:
        return 0., 0., 0.

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

    else:
        return 0., 0., 0.

    # Add ambient light
    R = R + R_obj * ambient_intensity
    G = G + G_obj * ambient_intensity
    B = B + B_obj * ambient_intensity

    # >>> 2) SHADOWS AND LAMBERT SHADING:

    # Shift point P along the normal vector to avoid shadow acne:
    BIAS = 0.001
    P0 = P0 + BIAS * N0
    P1 = P1 + BIAS * N1
    P2 = P2 + BIAS * N2

    # Get unit vector L from intersection point P to the light (only one light for now):
    L0, L1, L2 = get_vector_to_light((P0, P1, P2), lights)

    # If there is a line of sight to the light source, do the lambert shading:
    _, _, shadow_type = get_intersection((P0, P1, P2), (L0, L1, L2), spheres, planes)
    if shadow_type == 404:
        lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2
        if lambert_intensity > 0:
            R = R + R_obj * lambert_intensity
            G = G + G_obj * lambert_intensity
            B = B + B_obj * lambert_intensity

    # >>> 3) REFLECTIONS:

    # Reflection direction:
    R0, R1, R2 = get_reflection(ray_dir, (N0, N1, N2))

    # Shift point P along reflection vector to avoid mirror acne:
    P0 = P0 + BIAS * R0
    P1 = P1 + BIAS * R1
    P2 = P2 + BIAS * R2

    # Calculate reflected light by casting an additional ray from intersection point P:
    R_refl, G_refl, B_refl = trace_reflection_1((P0, P1, P2), (R0, R1, R2), spheres, lights, planes)

    reflection_intensity = 0.5
    R = R + R_refl * reflection_intensity
    G = G + G_refl * reflection_intensity
    B = B + B_refl * reflection_intensity

    return R, G, B


@cuda.jit()
def render_kernel(pixel_array, rays, spheres, lights, planes, do_reflection):
    """ This kernel render one pixel by casting a ray from a specific pixel location."""
    # Location of pixel
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds and run the ray trace for this pixel:
    if x <= pixel_array.shape[1] and y <= pixel_array.shape[2]:

        # Get the ray's origin at pixel location
        R0_X = rays[0, x, y]
        R0_Y = rays[1, x, y]
        R0_Z = rays[2, x, y]

        # Get the ray's direction at pixel location
        RD_X = rays[3, x, y]
        RD_Y = rays[4, x, y]
        RD_Z = rays[5, x, y]

        # Run the tracing for this pixel to get R, G, B values
        R, G, B = trace((R0_X, R0_Y, R0_Z), (RD_X, RD_Y, RD_Z), spheres, lights, planes)

        # Save the final color the array:
        pixel_array[0, x, y] = clip(R)
        pixel_array[1, x, y] = clip(G)
        pixel_array[2, x, y] = clip(B)


@cuda.jit
def ray_dir_kernel(pixel_locations, rays, O, R):

    """ Calculates the unit ray direction vector for the pixel at x, y for a camera origin O and camera rotation R.
        The output is then written into a 6 by width by height size 3D array called rays.
    """

    x, y = cuda.grid(2)

    if x <= pixel_locations.shape[1] and y <= pixel_locations.shape[2]:

        PX = pixel_locations[0, x, y]
        PY = pixel_locations[1, x, y]
        PZ = pixel_locations[2, x, y]

        # Ray direction is:
        # todo: this assumes O is at [0, 0, 0] for now.

        RD_X = R[0, 0] * PX + R[0, 1] * PY + R[0, 2] * PZ
        RD_Y = R[1, 0] * PX + R[1, 1] * PY + R[1, 2] * PZ
        RD_Z = R[2, 0] * PX + R[2, 1] * PY + R[2, 2] * PZ

        norm = math.sqrt(RD_X * RD_X + RD_Y * RD_Y + RD_Z * RD_Z)

        # Fill in the rays array:
        rays[0, x, y] = O[0]
        rays[1, x, y] = O[1]
        rays[2, x, y] = O[2]
        rays[3, x, y] = RD_X / norm
        rays[4, x, y] = RD_Y / norm
        rays[5, x, y] = RD_Z / norm


@cuda.jit(func_or_sig=None, device=True)
def trace_reflection_1(ray_origin, ray_dir, spheres, lights, planes):
    """ Trace the ray and return the R, G, B values"""

    # Ambient light:
    ambient_intensity = 0.3

    # Start with black
    R, G, B = (0.0, 0.0, 0.0)

    # Declare the color of the object
    R_obj, G_obj, B_obj = (0.0, 0.0, 0.0)

    # Check whether the ray hits any of the spheres or planes:
    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    # If no intersection: return black
    if obj_type == 404:
        return 0., 0., 0.

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

    else:
        return 0., 0., 0.

    # Add ambient light
    R = R + R_obj * ambient_intensity
    G = G + G_obj * ambient_intensity
    B = B + B_obj * ambient_intensity

    # >>> 2) SHADOWS AND LAMBERT SHADING:

    # Shift point P along the normal vector to avoid shadow acne:
    BIAS = 0.001
    P0 = P0 + BIAS * N0
    P1 = P1 + BIAS * N1
    P2 = P2 + BIAS * N2

    # Get unit vector L from intersection point P to the light (only one light for now):
    L0, L1, L2 = get_vector_to_light((P0, P1, P2), lights)

    # If there is a line of sight to the light source, do the lambert shading:
    _, _, shadow_type = get_intersection((P0, P1, P2), (L0, L1, L2), spheres, planes)
    if shadow_type == 404:
        lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2
        if lambert_intensity > 0:
            R = R + R_obj * lambert_intensity
            G = G + G_obj * lambert_intensity
            B = B + B_obj * lambert_intensity

    # >>> 3) REFLECTIONS:

    # Reflection direction:
    R0, R1, R2 = get_reflection(ray_dir, (N0, N1, N2))

    # Shift point P along reflection vector to avoid mirror acne:
    P0 = P0 + BIAS * R0
    P1 = P1 + BIAS * R1
    P2 = P2 + BIAS * R2

    # Calculate reflected light by casting an additional ray from intersection point P:
    R_refl, G_refl, B_refl = trace_reflection_2((P0, P1, P2), (R0, R1, R2), spheres, lights, planes)

    reflection_intensity = 0.5
    R = R + R_refl * reflection_intensity
    G = G + G_refl * reflection_intensity
    B = B + B_refl * reflection_intensity

    return R, G, B


@cuda.jit(func_or_sig=None, device=True)
def trace_reflection_2(ray_origin, ray_dir, spheres, lights, planes):
    """ Trace the ray and return the R, G, B values"""

    # Ambient light:
    ambient_intensity = 0.3

    # Start with black
    R, G, B = (0.0, 0.0, 0.0)

    # Declare the color of the object
    R_obj, G_obj, B_obj = (0.0, 0.0, 0.0)

    # Check whether the ray hits any of the spheres or planes:
    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    # If no intersection: return black
    if obj_type == 404:
        return 0., 0., 0.

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

    else:
        return 0., 0., 0.

    # Add ambient light
    R = R + R_obj * ambient_intensity
    G = G + G_obj * ambient_intensity
    B = B + B_obj * ambient_intensity

    # >>> 2) SHADOWS AND LAMBERT SHADING:

    # Shift point P along the normal vector to avoid shadow acne:
    BIAS = 0.001
    P0 = P0 + BIAS * N0
    P1 = P1 + BIAS * N1
    P2 = P2 + BIAS * N2

    # Get unit vector L from intersection point P to the light (only one light for now):
    L0, L1, L2 = get_vector_to_light((P0, P1, P2), lights)

    # If there is a line of sight to the light source, do the lambert shading:
    _, _, shadow_type = get_intersection((P0, P1, P2), (L0, L1, L2), spheres, planes)
    if shadow_type == 404:
        lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2
        if lambert_intensity > 0:
            R = R + R_obj * lambert_intensity
            G = G + G_obj * lambert_intensity
            B = B + B_obj * lambert_intensity

    return R, G, B