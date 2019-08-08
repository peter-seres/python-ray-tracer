from numba import cuda
import math


@cuda.jit(device=True)
def clip(color):
    return min(max(0, int(round(color))), 255)


@cuda.jit(device=True)
def intersect_ray_sphere(ray_origin, ray_direction, sphere_origin, sphere_radius):
    """ Returns the distance to the sphere. Returns -999 if does not intersect. """

    L0 = sphere_origin[0] - ray_origin[0]
    L1 = sphere_origin[1] - ray_origin[1]
    L2 = sphere_origin[2] - ray_origin[2]

    R0 = ray_direction[0]
    R1 = ray_direction[1]
    R2 = ray_direction[2]

    t_ca = L0 * R0 + L1 * R1 + L2 * R2

    if t_ca < 0:
        return -999.0

    d = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2 - t_ca**2)

    if d < 0 or d > sphere_radius:
        return -999.0

    t_hc = math.sqrt(sphere_radius ** 2 - d ** 2)

    t_0 = t_ca - t_hc
    t_1 = t_ca + t_hc

    if t_0 < 0:
        t_0 = t_1
        if t_0 < 0:
            return -999.0

    return t_0 * 0.999


@cuda.jit(device=True)
def intersect_ray_plane(ray_origin, ray_dir, plane_origin, plane_normal):
    """ Returns the distance to the plane. Returns -999.0 if does not intersect. """
    EPS = 0.001

    L0 = ray_dir[0]
    L1 = ray_dir[1]
    L2 = ray_dir[2]

    N0 = plane_normal[0]
    N1 = plane_normal[1]
    N2 = plane_normal[2]

    denom = L0 * N0 + L1 * N1 + L2 * N2

    if abs(denom) < EPS:
        return -999.0

    LP_0 = plane_origin[0] - ray_origin[0]
    LP_1 = plane_origin[1] - ray_origin[1]
    LP_2 = plane_origin[2] - ray_origin[2]

    nominator = LP_0 * N0 + LP_1 * N1 + LP_2 * N2

    dist = nominator / denom

    if dist > 0:
        return dist * 0.999
    else:
        return -999.0


@cuda.jit(device=True)
def get_intersection(ray_origin, ray_dir, spheres, planes):

    intersect_dist = 999.0
    obj_index = -999
    obj_type = 404

    for idx in range(spheres.shape[1]):
        dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])

        # If it hits the sphere and is closer than the closest one:
        if intersect_dist > dist > 0:
            intersect_dist = dist
            obj_index = idx
            obj_type = 0

    for idx in range(planes.shape[1]):
        dist = intersect_ray_plane(ray_origin, ray_dir, planes[0:3, idx], planes[3:6, idx])

        # If it hits the sphere and is closer than the closest one:
        if intersect_dist > dist > 0:
            intersect_dist = dist
            obj_index = idx
            obj_type = 1

    return intersect_dist, obj_index, obj_type


@cuda.jit(device=True)
def get_refl_intersection(ray_origin, ray_dir, spheres, planes, self_index, self_type):

    intersect_dist = 999.0
    obj_index = -999
    obj_type = 404

    for idx in range(spheres.shape[1]):
        if self_type == 0 and self_index == idx:
            continue
        else:
            dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])
            # If it hits the sphere and is closer than the closest one:
            if intersect_dist > dist > 0:
                intersect_dist = dist
                obj_index = idx
                obj_type = 0

    for idx in range(planes.shape[1]):
        if self_type == 1 and self_index == idx:
            continue
        else:
            dist = intersect_ray_plane(ray_origin, ray_dir, planes[0:3, idx], planes[3:6, idx])
            # If it hits the sphere and is closer than the closest one:
            if intersect_dist > dist > 0:
                intersect_dist = dist
                obj_index = idx
                obj_type = 1

    return intersect_dist, obj_index, obj_type


@cuda.jit(device=True)
def get_shadow_intersection(ray_origin, ray_dir, spheres, planes, obj_index, obj_type):

    in_shadow = False
    intersect_dist = 999.0

    for idx in range(spheres.shape[1]):
        if obj_type == 0 and obj_index == idx:
            continue
        else:
            dist = intersect_ray_sphere(ray_origin, ray_dir, spheres[0:3, idx], spheres[3, idx])
            # If it hits the sphere and is closer than the closest one:
            if intersect_dist > dist > 0:
                in_shadow = True
                return in_shadow

    for idx in range(planes.shape[1]):
        if obj_type == 1 and obj_index == idx:
            continue
        else:
            dist = intersect_ray_plane(ray_origin, ray_dir, planes[0:3, idx], planes[3:6, idx])
            # If it hits the sphere and is closer than the closest one:
            if intersect_dist > dist > 0:
                in_shadow = True
                return in_shadow

    return in_shadow


@cuda.jit(func_or_sig=None, device=True)
def trace_reflection(ray_origin, ray_dir, spheres, lights, planes, self_type, self_index):
    # Ambient light:
    ambient_intensity = 0.2
    lambert_shading = 1.0

    # Start with black
    R = 0.0
    G = 0.0
    B = 0.0

    # Declare the color of the object
    R_obj = 0.0
    G_obj = 0.0
    B_obj = 0.0

    # Check whether the ray hits any of the spheres or planes:
    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    # If not: return black
    if obj_type == 404:
        return clip(R), clip(G), clip(B)

    if obj_type == self_type and obj_index == self_index:   # Fake intersection with thyself.
        return 0, 0, 0

    # Get point of intersection P
    P0 = ray_origin[0] + ray_dir[0] * intersect_dist
    P1 = ray_origin[1] + ray_dir[1] * intersect_dist
    P2 = ray_origin[2] + ray_dir[2] * intersect_dist

    # Get unit vector point from intersection point P to the light L
    L0 = lights[0, 0] - P0
    L1 = lights[1, 0] - P1
    L2 = lights[2, 0] - P2
    norm = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2)
    L0 = L0 / norm
    L1 = L1 / norm
    L2 = L2 / norm

    if obj_type == 0:       # (if it's a sphere)
        # The color of the sphere:
        R_obj = spheres[4, obj_index]
        G_obj = spheres[5, obj_index]
        B_obj = spheres[6, obj_index]

        N0 = P0 - spheres[0, obj_index]
        N1 = P1 - spheres[1, obj_index]
        N2 = P2 - spheres[2, obj_index]
        norm = math.sqrt(N0 * N0 + N1 * N1 + N2 * N2)
        N0 = N0 / norm
        N1 = N1 / norm
        N2 = N2 / norm

    elif obj_type == 1:     # (if it's a plane)

        # Color of the plane
        R_obj = planes[6, obj_index]
        G_obj = planes[7, obj_index]
        B_obj = planes[8, obj_index]

        # Normal vector at intersection point
        N0 = planes[3, obj_index]
        N1 = planes[4, obj_index]
        N2 = planes[5, obj_index]

    else:
        return R, G, B

    # Add ambient light
    R = R + R_obj * ambient_intensity
    G = G + G_obj * ambient_intensity
    B = B + B_obj * ambient_intensity

    # If there is a line of sight to the light source, do the lambert shading:
    in_shadow = get_shadow_intersection((P0, P1, P2), (L0, L1, L2), spheres, planes, obj_type, obj_index)
    if not in_shadow:
        lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2
        if lambert_intensity > 0:
            R = R + R_obj * lambert_intensity * lambert_shading
            G = G + G_obj * lambert_intensity * lambert_shading
            B = B + B_obj * lambert_intensity * lambert_shading

    return R, G, B


@cuda.jit(func_or_sig=None, device=True)
def trace(ray_origin, ray_dir, spheres, lights, planes):
    """ Trace the ray and return the R, G, B values"""

    # Ambient light:
    ambient_intensity = 0.3
    lambert_shading = 1.0
    specular_shading = 0.8

    # Start with black
    R = 0.0
    G = 0.0
    B = 0.0

    # Declare the color of the object
    R_obj = 0.0
    G_obj = 0.0
    B_obj = 0.0

    # Check whether the ray hits any of the spheres or planes:
    intersect_dist, obj_index, obj_type = get_intersection(ray_origin, ray_dir, spheres, planes)

    # If not: return black
    if obj_type == 404:
        return clip(R), clip(G), clip(B)

    # Get point of intersection P
    P0 = ray_origin[0] + ray_dir[0] * intersect_dist * 0.999
    P1 = ray_origin[1] + ray_dir[1] * intersect_dist * 0.999
    P2 = ray_origin[2] + ray_dir[2] * intersect_dist * 0.999

    # Get unit vector point from intersection point P to the light L
    L0 = lights[0, 0] - P0
    L1 = lights[1, 0] - P1
    L2 = lights[2, 0] - P2
    norm = math.sqrt(L0 * L0 + L1 * L1 + L2 * L2)
    L0 = L0 / norm
    L1 = L1 / norm
    L2 = L2 / norm

    if obj_type == 0:       # (if it's a sphere)

        # The color of the sphere:
        R_obj = spheres[4, obj_index]
        G_obj = spheres[5, obj_index]
        B_obj = spheres[6, obj_index]

        N0 = P0 - spheres[0, obj_index]
        N1 = P1 - spheres[1, obj_index]
        N2 = P2 - spheres[2, obj_index]
        norm = math.sqrt(N0 * N0 + N1 * N1 + N2 * N2)
        N0 = N0 / norm
        N1 = N1 / norm
        N2 = N2 / norm

    elif obj_type == 1:     # (if it's a plane)

        # Color of the plane
        R_obj = planes[6, obj_index]
        G_obj = planes[7, obj_index]
        B_obj = planes[8, obj_index]

        # Normal vector at intersection point
        N0 = planes[3, obj_index]
        N1 = planes[4, obj_index]
        N2 = planes[5, obj_index]

    else:
        return clip(R), clip(G), clip(B)

    # Add ambient light
    R = R + R_obj * ambient_intensity
    G = G + G_obj * ambient_intensity
    B = B + B_obj * ambient_intensity

    # If there is a line of sight to the light source, do the lambert shading:
    in_shadow = get_shadow_intersection((P0, P1, P2), (L0, L1, L2), spheres, planes, obj_type, obj_index)
    if not in_shadow:
        lambert_intensity = L0 * N0 + L1 * N1 + L2 * N2
        if lambert_intensity > 0:
            R = R + R_obj * lambert_intensity * lambert_shading
            G = G + G_obj * lambert_intensity * lambert_shading
            B = B + B_obj * lambert_intensity * lambert_shading

    # Reflect direction:
    R0 = ray_dir[0] - 2 * (ray_dir[0] * N0) * N0
    R1 = ray_dir[1] - 2 * (ray_dir[1] * N1) * N1
    R2 = ray_dir[2] - 2 * (ray_dir[2] * N2) * N2
    norm = math.sqrt(R0 * R0 + R1 * R1 + R2 * R2)
    R0 = R0 / norm
    R1 = R1 / norm
    R2 = R2 / norm

    # Calculate reflected light by calling trace again recursively
    if obj_type == 1:
        R_refl, G_refl, B_refl = trace_reflection((P0, P1, P2), (R0, R1, R2), spheres, lights, planes, obj_type, obj_index)

        R = R + R_refl * specular_shading
        G = G + G_refl * specular_shading
        B = B + B_refl * specular_shading

    return clip(R), clip(G), clip(B)


@cuda.jit
def ray_dir_kernel(pixel_locations, rays, O, R):

    """ Calculates the unit ray direction vector for the pixel at x, y for a camera origin O and camera rotation R.
        The output is then written into a 6 by w by h 3D array called rays.
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


@cuda.jit()
def render_kernel(pixel_array, rays, spheres, lights, planes):

    # Location of pixel
    x, y = cuda.grid(2)

    # Check if the thread is within the bounds and run the ray trace for this pixel:
    if x <= pixel_array.shape[1] and y <= pixel_array.shape[2]:

        # Get the ray's origin
        R0_X = rays[0, x, y]
        R0_Y = rays[1, x, y]
        R0_Z = rays[2, x, y]

        # Get the ray's direction
        RD_X = rays[3, x, y]
        RD_Y = rays[4, x, y]
        RD_Z = rays[5, x, y]

        # Run the tracing for this pixel to get R, G, B values
        R, G, B = trace((R0_X, R0_Y, R0_Z), (RD_X, RD_Y, RD_Z), spheres, lights, planes)

        pixel_array[0, x, y] = R
        pixel_array[1, x, y] = G
        pixel_array[2, x, y] = B